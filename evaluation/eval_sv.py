"""Evaluate single-view CAD alignment on the ScanNet25k val split.

Two phases:

  predictions: walk val (scene, frame, instance) entries, run single-view
               alignment via sufleca.sv_align.align_single_view, decompose
               the predicted (cad_orig -> camera) matrix into (t, q, s),
               and write to <output_dir>/predictions.csv (resumable).
               Entries whose instance covers < 0.1 % of the image are
               skipped before the model runs.

  metrics:     Evaluate predictions.csv with scene-level NMS
               (0.4 m / 60 deg / 0.6 ratio), apply the Scan2CAD count cap,
               then compare against scene-level GT from full_annotations.json.

Usage:
  python evaluation/eval_sv.py --config configs/sufleca.yaml
  python evaluation/eval_sv.py --config configs/sufleca.yaml --phase metrics
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import re
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import munch
import numpy as np
import quaternion
import torch
import yaml

from sufleca.config import get_config_value
from sufleca.featurizer import load_sufleca_model
from sufleca.sv_align import align_single_view

RENDER_ROTATION = np.array(
    [
        [0.0, 0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


# ── Scan2CAD evaluation thresholds ────────────────────────────────────────────

NMS_TRANS = 0.4
NMS_ROT = 60.0
NMS_SCALE = 0.6

TRANS_THRESH = 0.2
ROT_THRESH = 20.0
SCALE_THRESH = 0.2

# CSV column layout. sx/sy/sz are TRUE OBJECT SIZES in metres (cad-orig axis
# ordering): the world-frame axis extents of the predicted CAD
# (= s_unitless · bbox_size), so metrics can compare scales across
# heterogeneous retrieved CADs without any per-row conversion.
PREDICTIONS_FIELDNAMES = [
    "id_scan", "frame", "inst_id",
    "objectCategory", "alignedModelId",
    "tx", "ty", "tz", "qw", "qx", "qy", "qz", "sx", "sy", "sz",
    "object_score", "roca_score",
]

CAD_TAXONOMY = {
    "02747177": "bin",
    "02808440": "bathtub",
    "02818832": "bed",
    "02871439": "bookcase",
    "02933112": "cabinet",
    "03001627": "chair",
    "03211117": "display",
    "04256520": "sofa",
    "04379243": "table",
}

# ── tqs <-> 4x4 helpers ───────────────────────────────────────────────────────

def make_M_from_tqs(t, q, s) -> np.ndarray:
    """Compose a 4x4 transform ``T @ R @ S`` from translation, quaternion, and scale."""
    if not isinstance(q, np.quaternion):
        q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4); T[:3, 3] = t
    R = np.eye(4); R[:3, :3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4); S[:3, :3] = np.diag(s)
    return T @ R @ S


def decompose_mat4(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor a 4x4 into ``(t, q, s)``: translation, rotation quaternion, and
    per-axis (column-norm) scales, rejecting reflections."""
    A = M[:3, :3].astype(np.float64)
    # Per-axis scales from column norms (A = R @ diag(s) convention).
    s = np.linalg.norm(A, axis=0)
    # Rotation via polar decomposition (nearest orthonormal matrix); the det<0
    # guard rejects reflections rather than scoring a mirror as a rotation.
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    q = quaternion.as_float_array(quaternion.from_rotation_matrix(R))
    t = M[:3, 3]
    return t, q, s


# ── Alignment error metrics ───────────────────────────────────────────────────

def _calc_rotation_diff(q: np.quaternion, q_gt: np.quaternion) -> float:
    """Geodesic angle (degrees) between two unit quaternions."""
    dot = float(np.dot(quaternion.as_float_array(q_gt),
                       quaternion.as_float_array(q)))
    dot = min(abs(dot), 1.0)
    return float(np.rad2deg(2 * np.arccos(dot)))


def rotation_diff(q, q_gt, sym: str = "") -> float:
    """Rotation error (degrees), minimised over the object's up-axis symmetry
    group (``sym`` selects a 2/4/inf-fold rotation set)."""
    sym_m = {"__SYM_ROTATE_UP_2": 2, "__SYM_ROTATE_UP_4": 4, "__SYM_ROTATE_UP_INF": 36}
    if sym in sym_m:
        m = sym_m[sym]
        return float(np.min([
            _calc_rotation_diff(
                q,
                q_gt * quaternion.from_rotation_vector([0.0, (i * 2.0 / m) * np.pi, 0.0]),
            )
            for i in range(m)
        ]))
    return _calc_rotation_diff(q, q_gt)


def scale_ratio(pred_s, gt_s) -> float:
    """Mean absolute relative per-axis scale error ``mean(|pred/gt - 1|)``."""
    return float(np.mean(np.abs(np.array(pred_s) / np.array(gt_s) - 1.0)))


def translation_diff(pred_t, gt_t) -> float:
    """Euclidean translation error in metres."""
    return float(np.linalg.norm(np.asarray(pred_t) - np.asarray(gt_t)))


# ── Render-frame helper ──────────────────────────────────────────────────────

def build_T_render_from_cad(scale: float, center_orig: np.ndarray) -> np.ndarray:
    """4x4 mapping cad_orig -> render-pool pointmap.

    Render pipeline: vertices -= center;
    apply ROTATION; rescale by `scale`. So pointmap = scale * R_rot @ (cad - center).
    """
    R = RENDER_ROTATION[:3, :3]
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = -scale * R @ np.asarray(center_orig, dtype=np.float64)
    return T


# ── Phase A: Predictions ─────────────────────────────────────────────────────

def load_val_scene_ids(val_scenes_path: str) -> set[str]:
    """Load the Scan2CAD val split scene ids from Dataset JSON."""
    with open(val_scenes_path) as f:
        return set(json.load(f).keys())


def _resolve_category_filter(category: str | None) -> set[str] | None:
    """Return a set of synset ids to keep, or None for no filtering."""
    if not category:
        return None
    key = category.strip().lower()
    if not key:
        return None
    if key in CAD_TAXONOMY:
        return {key}
    if key.isdigit():
        cat_id = key.zfill(8)
        if cat_id in CAD_TAXONOMY:
            return {cat_id}
    for cat_id, name in CAD_TAXONOMY.items():
        if name.lower() == key:
            return {cat_id}
    names = ", ".join(sorted(CAD_TAXONOMY.values()))
    raise ValueError(
        f"Unknown category '{category}'. Use a synset id or one of: {names}"
    )


def build_targets_roca(
    roca_per_frame_path: str,
    val_scenes_path: str,
    renderings_root: str,
    render_pool: str,
    category_filter: set[str] | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> tuple[list, int, int]:
    """Build targets from an external detector's per-frame detections and its
    retrieved CAD models.

    Instance IDs are the 1-based detection index, matching the IDs stored in the
    per-frame masks under ``roca_sam_masks/``. Deduplicates by
    (scene, frame, inst_id), keeping the highest-scoring detection per slot.

    Returns (targets, skipped_no_render, skipped_no_mask_file).
    """
    with open(roca_per_frame_path) as f:
        roca_data = json.load(f)

    val_scenes = load_val_scene_ids(val_scenes_path)
    cad_ok: set = set()
    cad_missing: set = set()

    best_per_slot: dict[tuple, tuple] = {}  # (scene, frame, inst_id) -> (score, al)

    skipped_no_render = 0
    skipped_no_mask_file = 0

    for key, detections in roca_data.items():
        # key: "scene/color/000000.jpg"
        parts = key.split("/")
        scene = parts[0]
        if scene not in val_scenes:
            continue
        frame = os.path.splitext(parts[-1])[0]  # strip ".jpg"

        if not detections:
            continue

        # roca_sam_masks uses 1-based detection indices as instance IDs
        mask_path = os.path.join(renderings_root, scene, "roca_sam_masks", f"{frame}.png")
        if not os.path.exists(mask_path):
            skipped_no_mask_file += 1
            continue

        for det_idx, det in enumerate(detections):
            inst_id = det_idx + 1  # 1-based, matching roca_sam_masks
            scene_cad_id = det.get("scene_cad_id")
            if not scene_cad_id:
                continue
            cat, mid = scene_cad_id
            if cat not in CAD_TAXONOMY:
                continue
            if category_filter is not None and cat not in category_filter:
                continue

            cad_key = f"{cat}/{mid}"
            if cad_key in cad_missing:
                skipped_no_render += 1
                continue
            if cad_key not in cad_ok:
                if os.path.isdir(os.path.join(render_pool, cat, mid)):
                    cad_ok.add(cad_key)
                else:
                    cad_missing.add(cad_key)
                    skipped_no_render += 1
                    continue

            slot = (scene, frame, inst_id)
            # Semantic score = detection confidence x CAD retrieval score; the
            # 'mixed' scoring mode fuses its per-scene percentile rank with the
            # alignment logdet's rank.
            retr_score = float(det.get("score", 0.0))
            det_score = float(det.get("det_score", 1.0))
            score = det_score * retr_score
            al = {"catid_cad": cat, "id_cad": mid, "sym": "", "roca_score": score}
            prev = best_per_slot.get(slot)
            if prev is None or score > prev[0]:
                best_per_slot[slot] = (score, al)

    targets = [
        (scene, frame, inst_id, al)
        for (scene, frame, inst_id), (_, al) in best_per_slot.items()
    ]

    print(f"Detector targets: {len(targets)} unique slots "
          f"(skipped {skipped_no_render} no-render, "
          f"{skipped_no_mask_file} no-mask-file)")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(targets)

    return targets, skipped_no_render, skipped_no_mask_file


_WORKER_ARGS = None
_WORKER_CONFIG = None
_WORKER_CAD_CENTERS = None
_WORKER_MODELS = None
_WORKER_CACHES = None


def _init_worker(args: argparse.Namespace) -> None:
    """Load models and config in each worker process."""
    global _WORKER_ARGS, _WORKER_CONFIG, _WORKER_CAD_CENTERS, _WORKER_MODELS, _WORKER_CACHES
    _WORKER_ARGS = args
    with open(args.config) as f:
        config = munch.munchify(yaml.safe_load(f))
    _WORKER_CONFIG = config
    with open(args.cad_centers) as f:
        _WORKER_CAD_CENTERS = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    featurizer = load_sufleca_model(checkpoint=args.checkpoint, device=device)
    _WORKER_MODELS = {
        "featurizer": featurizer,
        "device": device,
    }
    _WORKER_CACHES = {"intrinsics": {}, "render_paths": {}}


def _predict_one_target(
    target,
    args: argparse.Namespace,
    config,
    cad_centers: dict,
    models_pack: dict,
    caches: dict,
) -> tuple:
    """Align one ``(scene, frame, inst_id, alignment)`` target and return
    ``(row_or_None, attempted, succeeded, skipped)`` counters for aggregation."""
    scene, frame, inst_id, al = target
    cat = al["catid_cad"]
    mid = al["id_cad"]
    cad_key = f"{cat}/{mid}"
    if cad_key not in cad_centers:
        return None, 0, 0, 1

    scene_dir = os.path.join(args.images_root, scene)
    color_path = os.path.join(scene_dir, "color", f"{frame}.jpg")
    depth_subdir = "true_depth" if args.true_depth else "finetuned_depth"
    depth_path = os.path.join(scene_dir, depth_subdir, f"{frame}.png")
    sam_path = os.path.join(args.renderings, scene, "roca_sam_masks", f"{frame}.png")

    intrinsics_cache = caches["intrinsics"]
    if scene not in intrinsics_cache:
        ipath = os.path.join(scene_dir, "intrinsics_color.txt")
        if not os.path.exists(ipath):
            intrinsics_cache[scene] = None
        else:
            intrinsics_cache[scene] = np.loadtxt(ipath)
    intrinsic = intrinsics_cache[scene]
    if intrinsic is None:
        return None, 0, 0, 0

    render_model_dir = os.path.join(args.render_pool, cat, mid)
    if not os.path.exists(os.path.join(render_model_dir, "precomputed", "scores.npz")):
        return None, 0, 0, 0

    attempted = 1
    try:
        res = align_single_view(
            color_path=color_path,
            sam_mask_path=sam_path,
            inst_id=inst_id,
            intrinsic=intrinsic,
            render_model_dir=render_model_dir,
            featurizer_model=models_pack["featurizer"],
            config=config,
            depth_path=depth_path,
            device=models_pack["device"],
        )
    except Exception as exc:
        print(
            f"ERROR aligning {scene}/{frame} instance {inst_id} "
            f"({cat}/{mid}): {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None, attempted, 0, 0
    if res is None:
        return None, attempted, 0, 0

    pose_path = os.path.join(args.images_root, scene, "pose", f"{frame}.txt")
    if not os.path.exists(pose_path):
        return None, attempted, 0, 0
    pose = np.loadtxt(pose_path)
    if not np.isfinite(pose).all():
        return None, attempted, 0, 0

    T_v = np.eye(4)
    T_v[:3, :3] = res["R"] @ res["S"]
    T_v[:3, 3] = res["t"]
    rscale = _read_render_scale(args.render_pool, cat, mid)
    center = np.asarray(cad_centers[cad_key], dtype=np.float64)
    M_pred_cam = T_v @ build_T_render_from_cad(rscale, center)
    t_w, q_w, s_w = decompose_mat4(pose @ M_pred_cam)

    # Convert unitless cad_orig→world scale to true object size (metres).
    size_metric = _true_object_size(args.render_pool, cat, mid, s_w)
    if size_metric is None:
        return None, attempted, 0, 0
    s_w = np.asarray(size_metric, dtype=np.float64)

    roca_val = al.get("roca_score", "")
    row = {
        "id_scan": scene, "frame": frame, "inst_id": inst_id,
        "objectCategory": cat, "alignedModelId": mid,
        "tx": t_w[0], "ty": t_w[1], "tz": t_w[2],
        "qw": q_w[0], "qx": q_w[1], "qy": q_w[2], "qz": q_w[3],
        "sx": s_w[0], "sy": s_w[1], "sz": s_w[2],
        "object_score": float(res["logdet_info"]),
        "roca_score": float(roca_val) if roca_val not in ("", None) else "",
    }
    return row, attempted, 1, 0


def _predict_one_target_worker(target) -> dict:
    """Pool-worker wrapper around :func:`_predict_one_target` using process globals."""
    row, attempted, success, skipped_no_center = _predict_one_target(
        target,
        _WORKER_ARGS,
        _WORKER_CONFIG,
        _WORKER_CAD_CENTERS,
        _WORKER_MODELS,
        _WORKER_CACHES,
    )
    return {
        "row": row,
        "attempted": attempted,
        "success": success,
        "skipped_no_center": skipped_no_center,
    }


def run_predictions(args, models_pack=None) -> None:
    """Predictions phase: build targets, align each (single- or multi-process),
    and stream rows to ``<output_dir>/predictions.csv`` (resumable)."""
    out_path = Path(args.output_dir) / "predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = munch.munchify(yaml.safe_load(f))

    # Save resolved config so results are self-describing.
    config_path = Path(args.output_dir) / "config.json"
    config_path.write_text(
        json.dumps(_run_record(args, config), indent=2, default=str)
    )

    # Resume support: skip rows already present.
    done = set()
    if out_path.exists() and not args.overwrite:
        with out_path.open(newline="") as f:
            first_line = f.readline().strip()
            expected_header = ",".join(PREDICTIONS_FIELDNAMES)
            if first_line != expected_header:
                print(
                    f"WARNING: existing {out_path} has an incompatible header "
                    f"(expected our {len(PREDICTIONS_FIELDNAMES)}-column format). "
                    f"Cannot resume — use --overwrite to replace it."
                )
            else:
                f.seek(0)
                for row in csv.DictReader(f):
                    try:
                        done.add((row["id_scan"], row["frame"], int(row["inst_id"])))
                    except (KeyError, ValueError):
                        pass
        print(f"Resuming: {len(done)} predictions already on disk.")

    category_filter = _resolve_category_filter(args.category)
    targets, skipped_no_render, skipped_no_mask_instance = build_targets_roca(
        args.roca_per_frame, args.val_scenes,
        renderings_root=args.renderings,
        render_pool=args.render_pool,
        category_filter=category_filter,
        shuffle=args.shuffle, seed=args.seed,
    )

    if category_filter is not None:
        labels = ", ".join(f"{c}:{CAD_TAXONOMY[c]}" for c in sorted(category_filter))
        print(f"Filtering to category: {labels}")

    if args.limit is not None:
        seen_scenes: list[str] = []
        seen_set: set[str] = set()
        for scene, *_ in targets:
            if scene not in seen_set:
                seen_set.add(scene)
                seen_scenes.append(scene)
            if len(seen_scenes) >= args.limit:
                break
        limited_scenes = set(seen_scenes)
        targets = [t for t in targets if t[0] in limited_scenes]
        print(f"Limiting to first {len(limited_scenes)} scenes "
              f"({len(targets)} entries).")

    if args.max_targets is not None and len(targets) > args.max_targets:
        targets = targets[: args.max_targets]
        print(f"Limiting to first {len(targets)} target entries.")

    if args.workers <= 1:
        if models_pack is None:
            raise ValueError("models_pack must be provided when workers <= 1")
        with open(args.cad_centers) as f:
            cad_centers = json.load(f)

    file_has_content = out_path.exists() and out_path.stat().st_size > 0
    if file_has_content and not args.overwrite:
        with out_path.open(newline="") as _f:
            existing_header = _f.readline().strip()
        if existing_header != ",".join(PREDICTIONS_FIELDNAMES):
            raise RuntimeError(
                f"{out_path} exists with an incompatible format. "
                f"Use --overwrite to replace it, or choose a different --output-dir."
            )
    write_header = args.overwrite or not file_has_content
    mode = "w" if args.overwrite else "a"
    fout = out_path.open(mode, newline="")
    writer = csv.DictWriter(fout, fieldnames=PREDICTIONS_FIELDNAMES, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    targets_to_run = [t for t in targets if (t[0], t[1], t[2]) not in done]
    if not targets_to_run:
        print("No pending targets to run.")
        fout.close()
        return

    t0 = time.time()
    n_attempted = n_success = 0
    skipped_no_center = 0

    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=_init_worker, initargs=(args,)) as pool:
            for idx, result in enumerate(
                pool.imap_unordered(_predict_one_target_worker, targets_to_run), start=1
            ):
                skipped_no_center += result["skipped_no_center"]
                n_attempted += result["attempted"]
                n_success += result["success"]
                row = result["row"]
                if row is not None:
                    writer.writerow(row)
                if idx % 25 == 0:
                    elapsed = time.time() - t0
                    rate = idx / max(elapsed, 1e-6)
                    print(f"[{idx}/{len(targets_to_run)}] success={n_success}/{n_attempted} "
                          f"({100*n_success/max(n_attempted,1):.1f}%) "
                          f"rate={rate:.2f} it/s ETA={(len(targets_to_run)-idx)/max(rate,1e-6)/60:.1f} min")
    else:
        caches = {"intrinsics": {}, "render_paths": {}}
        for idx, target in enumerate(targets_to_run, start=1):
            row, attempted, success, skipped = _predict_one_target(
                target, args, config, cad_centers, models_pack, caches
            )
            skipped_no_center += skipped
            n_attempted += attempted
            n_success += success
            if row is not None:
                writer.writerow(row)
            if idx % 25 == 0:
                elapsed = time.time() - t0
                rate = idx / max(elapsed, 1e-6)
                print(f"[{idx}/{len(targets_to_run)}] success={n_success}/{n_attempted} "
                      f"({100*n_success/max(n_attempted,1):.1f}%) "
                      f"rate={rate:.2f} it/s ETA={(len(targets_to_run)-idx)/max(rate,1e-6)/60:.1f} min")

    fout.close()
    elapsed = time.time() - t0
    print(f"\nFinished. attempted={n_attempted}, succeeded={n_success}, "
          f"skipped_no_center={skipped_no_center}")
    print(f"Prediction time: {elapsed:.1f}s ({elapsed/60:.2f} min)")


_render_scale_cache: dict[str, float] = {}

def _read_render_scale(render_pool: str, cat: str, mid: str) -> float:
    """Read (and cache) a CAD's render-frame ``scale`` from its ``metadata.json``."""
    key = f"{cat}/{mid}"
    if key in _render_scale_cache:
        return _render_scale_cache[key]
    meta = json.load(open(os.path.join(render_pool, cat, mid, "metadata.json")))
    s = float(meta["scale"])
    _render_scale_cache[key] = s
    return s


_cad_bbox_size_cache: dict[str, tuple | None] = {}

def _read_cad_bbox_size(render_pool: str, cat: str, mid: str) -> tuple | None:
    """Return (bbox_extent[dx, dy, dz] in cad-orig axis order, render_scale), or None."""
    key = f"{cat}/{mid}"
    if key in _cad_bbox_size_cache:
        return _cad_bbox_size_cache[key]
    meta_path = os.path.join(render_pool, cat, mid, "metadata.json")
    if not os.path.exists(meta_path):
        _cad_bbox_size_cache[key] = None
        return None
    meta = json.load(open(meta_path))
    size = meta.get("bounds", {}).get("size")
    if size is None:
        _cad_bbox_size_cache[key] = None
        return None
    scale = float(meta.get("scale", 1.0))
    arr = np.asarray(size, dtype=np.float64)
    arr = arr[[1, 2, 0]]
    _cad_bbox_size_cache[key] = (arr, scale)
    return arr, scale


def _true_object_size(render_pool: str, cat: str, mid: str, s: list | np.ndarray) -> np.ndarray | None:
    """Return metric object extents: s * cad_bbox_size, or None if bbox unavailable."""
    bs = _read_cad_bbox_size(render_pool, cat, mid)
    if bs is None:
        return None
    size, scale = bs
    return np.asarray(s, dtype=np.float64) * size / scale


# ── Phase B: Metrics ─────────────────────────────────────────────────────────

def _resolve_pred_path(output_dir: str) -> Path:
    """Return the predictions CSV path."""
    csv_path = Path(output_dir) / "predictions.csv"
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"No predictions.csv found in {output_dir}")


def load_predictions(path: str) -> dict[tuple[str, str, int], dict]:
    """Load predictions from predictions.csv (see PREDICTIONS_FIELDNAMES).

    sx/sy/sz are TRUE OBJECT SIZES in metres (cad-orig axis ordering). Every
    row is a successful prediction; failed alignments are never written.

    Returns a dict keyed by (scene, "", row_idx); the row index keeps keys
    unique, and per-scene NMS later collapses spatially redundant predictions.
    """
    by_key: dict[tuple[str, str, int], dict] = {}
    n_total = 0

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, raw in enumerate(reader):
            n_total += 1
            scene = raw["id_scan"]
            catid = raw["objectCategory"].zfill(8)
            score = float(raw.get("object_score") or 0)

            roca_val = raw.get("roca_score", "")
            pred: dict = {
                "scene": scene,
                "catid_cad": catid, "id_cad": raw["alignedModelId"],
                "score": score,
                "roca_score": float(roca_val) if roca_val != "" else float("-inf"),
                "t_w": [float(raw["tx"]), float(raw["ty"]), float(raw["tz"])],
                "q_w": [float(raw["qw"]), float(raw["qx"]), float(raw["qy"]), float(raw["qz"])],
                "s_w": [float(raw["sx"]), float(raw["sy"]), float(raw["sz"])],
            }
            by_key[(scene, "", row_idx)] = pred

    print(f"loaded {len(by_key)} predictions from {n_total} rows")
    return by_key


# ── World-space NMS + matching (Scan2CAD protocol) ────────────────────────────

def nms_per_scene(preds: list, gt_alignments: list,
                   render_pool: str = "render_pool",
                   scale_mode: str = "unitless") -> list:
    """Score-descending NMS using the Scan2CAD thresholds (0.4 m / 60 deg / 0.6 ratio).

    `preds` must already be sorted by score descending.

    scale_mode:
      - "unitless" (default): each prediction's s is divided by its CAD's
        bbox/scale before scale_ratio — comparing the cad_orig→world rescaling
        factor.
      - "metric": each prediction's s is taken as-is (world-space physical
        extent) and compared directly.
    """
    if not preds:
        return preds
    n = len(preds)
    keep = [True] * n
    quats = [np.quaternion(*p["q_w"]) for p in preds]

    def _scale_repr(p) -> list:
        """Scale vector for comparison: raw metric, or unitless cad->world factor."""
        if scale_mode == "metric":
            return p["s_w"]
        bs = _read_cad_bbox_size(render_pool, p["catid_cad"], p["id_cad"])
        if bs is None:
            return p["s_w"]
        size, sc = bs
        return (np.asarray(p["s_w"], dtype=np.float64) / (size / sc)).tolist()
    unitless = [_scale_repr(p) for p in preds]

    def _sym(p) -> str:
        """Look up the GT up-axis symmetry tag for this prediction's CAD."""
        for g in gt_alignments:
            if g["catid_cad"] == p["catid_cad"] and g["id_cad"] == p["id_cad"]:
                return g.get("sym", "")
        return ""

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            if preds[i]["catid_cad"] != preds[j]["catid_cad"]:
                continue
            sym = _sym(preds[i])
            if (
                translation_diff(preds[i]["t_w"], preds[j]["t_w"]) <= NMS_TRANS
                and scale_ratio(unitless[i], unitless[j]) <= NMS_SCALE
                and rotation_diff(quats[i], quats[j], sym) <= NMS_ROT
            ):
                keep[j] = False
    return [p for p, k in zip(preds, keep) if k]


def apply_count_cap(preds: list, gt_counts: Counter) -> list:
    """Keep at most GT-count predictions per category (already sorted by score)."""
    pred_counts: Counter = Counter()
    out = []
    for p in preds:
        cat = p["catid_cad"]
        if pred_counts[cat] < gt_counts.get(cat, 0):
            pred_counts[cat] += 1
            out.append(p)
    return out


def load_world_gt(
    full_annot_path: str, val_scenes_path: str, render_pool: str = ""
) -> tuple[dict, dict]:
    """Load scene-level GT in world frame from full_annotations.json.

    Returns (gt_per_scene, gt_counts_per_scene) where
      gt_per_scene[scene]  = [{'t', 'q', 's', 'catid_cad', 'id_cad', 'sym'}, ...]
      gt_counts_per_scene[scene] = Counter{catid_cad: count}
    both restricted to the CAD_TAXONOMY categories.

    The returned `s` is in METRES (per-axis world-frame extent in cad-orig
    ordering), matching the prediction convention. Falls back to the unitless
    decomposed scale if the CAD's metadata.json is missing under `render_pool`.
    """
    val_scenes = set(json.load(open(val_scenes_path)).keys())
    annots = json.load(open(full_annot_path))
    gt_per_scene: dict[str, list] = {}
    gt_counts_per_scene: dict[str, Counter] = {}
    for entry in annots:
        scene = entry["id_scan"]
        if scene not in val_scenes:
            continue
        trs = entry["trs"]
        to_scene = np.linalg.inv(
            make_M_from_tqs(trs["translation"], trs["rotation"], trs["scale"])
        )
        gts = []
        counts: Counter = Counter()
        for m in entry["aligned_models"]:
            cat = m["catid_cad"]
            if cat not in CAD_TAXONOMY:
                continue
            mtrs = m["trs"]
            to_s2c = make_M_from_tqs(
                mtrs["translation"], mtrs["rotation"], mtrs["scale"]
            )
            t, q, s = decompose_mat4(to_scene @ to_s2c)
            # Convert unitless scale → metric true-object-size if metadata available.
            if render_pool:
                ms = _true_object_size(render_pool, cat, m["id_cad"], s)
                if ms is not None:
                    s = ms
            gts.append({
                "t": t.tolist(), "q": q.tolist(), "s": np.asarray(s).tolist(),
                "catid_cad": cat, "id_cad": m["id_cad"], "sym": m.get("sym", ""),
            })
            counts[cat] += 1
        gt_per_scene[scene] = gts
        gt_counts_per_scene[scene] = counts
    return gt_per_scene, gt_counts_per_scene


def count_corrects_scene(
    preds: list, gts: list, render_pool: str = "",
    scale_mode: str = "unitless",
) -> Counter:
    """Match predictions to GT using TRANS/ROT/SCALE thresholds.

    In the default "unitless" mode the scale comparison is done in per-CAD
    scale-factor space: each side's metric `s` is divided by its CAD's
    `bbox/scale` to recover the cad_orig→world decomposition factor, so scales
    are comparable across heterogeneous retrieved CADs.

    Returns per-category correct counts.
    """
    corrects: Counter = Counter()
    covered = [False] * len(gts)

    def _scale_repr(cat: str, mid: str, s_metric) -> list:
        """Scale vector for comparison: raw metric, or unitless cad->world factor."""
        if scale_mode == "metric":
            return list(s_metric)
        bs = _read_cad_bbox_size(render_pool, cat, mid)
        if bs is None:
            return list(s_metric)
        size, sc = bs
        return (np.asarray(s_metric, dtype=np.float64) / (size / sc)).tolist()

    # "unitless": each side divided by its own CAD's bbox/scale; "metric": both
    # sides stay in world-space physical extents.
    gt_unitless = [_scale_repr(g["catid_cad"], g["id_cad"], g["s"]) for g in gts]

    for p in preds:
        p_unitless = _scale_repr(p["catid_cad"], p["id_cad"], p["s_w"])
        pq = np.quaternion(*p["q_w"])
        # sym is a property of the CAD model, looked up from the GT alignments.
        pred_sym = next(
            (g["sym"] for g in gts
             if g["catid_cad"] == p["catid_cad"] and g["id_cad"] == p["id_cad"]),
            ""
        )
        for j, g in enumerate(gts):
            if covered[j]:
                continue
            if g["catid_cad"] != p["catid_cad"]:
                continue
            sym = pred_sym if pred_sym == g["sym"] else ""
            s_err = scale_ratio(p_unitless, gt_unitless[j])
            s_ok = s_err <= SCALE_THRESH
            t_err = translation_diff(p["t_w"], g["t"])
            r_err = rotation_diff(pq, np.quaternion(*g["q"]), sym)
            ok = t_err <= TRANS_THRESH and r_err <= ROT_THRESH and s_ok
            if ok:
                corrects[p["catid_cad"]] += 1
                covered[j] = True
                break
    return corrects


def run_metrics(args) -> None:
    """Scan2CAD protocol: world-space NMS, count-cap, and scene-level GT matching."""
    pred_path = _resolve_pred_path(args.output_dir)

    preds_by_key = load_predictions(str(pred_path))
    gt_per_scene, gt_counts_per_scene = load_world_gt(
        args.full_annotations, args.val_scenes, render_pool=args.render_pool,
    )

    by_scene_world: dict[str, list] = defaultdict(list)
    scenes_with_predictions: set[str] = set()
    for pred in preds_by_key.values():
        scenes_with_predictions.add(pred["scene"])
        by_scene_world[pred["scene"]].append(pred)

    scoring = getattr(args, "scoring", "mixed")
    if scoring == "mixed":
        # `logdet` is only a monotonic ranking key: its sign and scale are
        # units-dependent (the info matrix mixes rotation/translation/scale
        # parameters), so multiplying it by `roca` is ill-posed. Instead fuse
        # per-scene percentile ranks of ROCA confidence and geometric logdet;
        # the product keeps AND semantics (a candidate must rank well on both).
        def _pct_ranks(values) -> np.ndarray:
            """Per-scene percentile ranks in ``(0, 1)``; non-finite entries score 0."""
            arr = np.asarray(values, dtype=float)
            out = np.zeros(len(arr))
            finite = np.isfinite(arr)
            n = int(finite.sum())
            if n:
                out[finite] = (np.argsort(np.argsort(arr[finite])) + 1) / (n + 1)
            return out

        for preds in by_scene_world.values():
            roca_rank = _pct_ranks([p.get("roca_score", float("-inf")) for p in preds])
            logdet_rank = _pct_ranks([p.get("score", float("-inf")) for p in preds])
            for p, rr, lr in zip(preds, roca_rank, logdet_rank):
                finite = np.isfinite(float(p.get("roca_score", float("-inf")))) and \
                    np.isfinite(float(p.get("score", float("-inf"))))
                p["mixed_score"] = float(rr * lr) if finite else float("-inf")
    score_keys = {"logdet": "score", "roca": "roca_score", "mixed": "mixed_score"}
    score_field = score_keys.get(scoring, "score")
    n_scenes_skipped = sum(1 for s in gt_per_scene if s not in scenes_with_predictions)
    print(f"Per-scene NMS (ranked by {scoring} / '{score_field}') + count-cap + accuracy …")
    if n_scenes_skipped:
        print(f"  skipping {n_scenes_skipped} val scenes with no predictions")
        print(f" the scenes: {sorted(s for s in gt_per_scene if s not in scenes_with_predictions)}")
    total_corrects: Counter = Counter()
    total_counts: Counter = Counter()

    for scene, gts in gt_per_scene.items():
        if scene not in scenes_with_predictions:
            continue
        for g in gts:
            total_counts[g["catid_cad"]] += 1
        preds = by_scene_world.get(scene, [])
        if preds:
            preds = sorted(preds, key=lambda p: -p.get(score_field, float("-inf")))
            preds = nms_per_scene(preds, gts, render_pool=args.render_pool,
                                   scale_mode=args.scale)
            preds = apply_count_cap(preds, gt_counts_per_scene.get(scene, Counter()))
        corrects = count_corrects_scene(
            preds, gts, render_pool=args.render_pool, scale_mode=args.scale,
        )
        total_corrects.update(corrects)

    _print_metrics(total_corrects, total_counts, Path(args.output_dir) / "metrics_nms.json")


def _print_metrics(total_corrects: Counter, total_counts: Counter, out_path: Path) -> None:
    """Compute, print, and save per-class + aggregate accuracy to ``out_path``."""
    rows = []
    cat_accs = []
    sum_correct = sum_count = 0
    for cat_id, name in sorted(CAD_TAXONOMY.items(), key=lambda x: x[1]):
        c = total_corrects[cat_id]
        n = total_counts[cat_id]
        acc = (100.0 * c / n) if n else float("nan")
        rows.append((name, c, n, acc))
        if n:
            cat_accs.append(acc)
            sum_correct += c
            sum_count += n

    print("\nPer-class accuracy")
    print(f"{'class':<10} {'correct':>8} {'gt':>6} {'acc%':>7}")
    for name, c, n, acc in rows:
        print(f"{name:<10} {c:>8d} {n:>6d} {acc:>6.2f}")
    print()
    cat_avg = float(np.mean(cat_accs)) if cat_accs else float("nan")
    inst_avg = float(100.0 * sum_correct / sum_count) if sum_count else float("nan")
    print(f"category-averaged accuracy: {cat_avg:.2f}")
    print(f"instance-averaged accuracy: {inst_avg:.2f}")

    summary = {
        "per_class": [
            {"class": n, "correct": int(c), "gt": int(g), "accuracy": float(a)}
            for (n, c, g, a) in rows
        ],
        "category_avg": cat_avg,
        "instance_avg": inst_avg,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse the eval CLI (config, paths, phase, worker count, scoring, filters)."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sufleca.yaml",
                   help="Algorithm config (checkpoint + image_processing/ransac/correspondence)")

    # Run settings
    p.add_argument("--phase", choices=["predictions", "metrics", "all"], default="all")
    p.add_argument("--output-dir", default="runs/eval_sv")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument("--category", default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--true-depth", action="store_true")
    p.add_argument("--roca-per-frame", default=None,
                   help="Path to ROCA per-frame detections JSON (defaults to "
                        "<scannet25k>/Dataset/roca_per_frame.json)")
    p.add_argument("--scoring", choices=["logdet", "roca", "mixed"], default="mixed",
                   help="NMS ranking score; 'mixed' = per-scene rank-fusion of "
                        "roca_score and logdet (default)")
    p.add_argument("--scale", choices=["unitless", "metric"], default="metric")

    # Data paths (everything ScanNet25k-internal is derived from --scannet25k)
    p.add_argument("--scannet25k", default="data/ScanNet25k",
                   help="ScanNet25k dataset root (contains Images/ and Dataset/)")
    p.add_argument("--render-pool", default=None,
                   help="Alignment pool (default: data/render_pool_<config checkpoint>)")
    p.add_argument("--cad-centers", default="data/cad_orig_centers.json")
    return p.parse_args()


def _resolve_runtime_args(cli_args: argparse.Namespace, config) -> argparse.Namespace:
    """Build the runtime namespace from CLI args + the checkpoint from config.

    Run/data settings come from the CLI. The checkpoint is the only run setting
    kept in the config file. All ScanNet25k-internal paths (image root, masks,
    alignments, val split, full annotations) are derived from --scannet25k.
    """
    args = argparse.Namespace(**vars(cli_args))
    args.checkpoint = get_config_value(config, "run.checkpoint", "sufleca")
    if args.render_pool is None:
        checkpoint_path = Path(str(args.checkpoint))
        checkpoint_name = (
            checkpoint_path.parent.name
            if checkpoint_path.name in {"best.pt", "checkpoint.pt"}
            else checkpoint_path.name
        )
        checkpoint_name = re.sub(r"[^A-Za-z0-9._-]+", "_", checkpoint_name).strip("._-")
        if not checkpoint_name:
            raise ValueError(f"cannot derive render-pool name from checkpoint {args.checkpoint!r}")
        args.render_pool = os.path.join("data", f"render_pool_{checkpoint_name}")

    if args.max_targets is not None and args.max_targets <= 0:
        raise ValueError("--max-targets must be a positive integer")

    root = args.scannet25k
    # Per-frame images/masks live under <root>/Images/<scene>/...
    args.images_root = os.path.join(root, "Images")
    args.renderings = args.images_root  # sam_masks / roca_sam_masks live with images
    args.val_scenes = os.path.join(root, "Dataset", "scan2cad_val_scenes.json")
    args.full_annotations = os.path.join(root, "full_annotations.json")
    if args.roca_per_frame is None:
        args.roca_per_frame = os.path.join(root, "Dataset", "roca_per_frame.json")

    return args


def _run_record(args: argparse.Namespace, config) -> dict:
    """Serializable record of the resolved runtime args and config for provenance."""
    return {
        "runtime": vars(args),
        "config": munch.unmunchify(config),
    }


def main() -> None:
    """Entry point: resolve args/config and run the requested phase(s)."""
    cli_args = parse_args()
    with open(cli_args.config) as f:
        config = munch.munchify(yaml.safe_load(f))
    args = _resolve_runtime_args(cli_args, config)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.phase in ("predictions", "all"):
        if args.workers > 1:
            if torch.cuda.is_available():
                print("Warning: CUDA + multi-process may oversubscribe the GPU. "
                      "Consider run.workers: 1 for GPU runs.")
            run_predictions(args)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading models on {device} …")
            featurizer = load_sufleca_model(checkpoint=args.checkpoint, device=device)
            models_pack = {
                "featurizer": featurizer,
                "device": device,
            }
            run_predictions(args, models_pack)

    if args.phase in ("metrics", "all"):
        if not os.path.exists(args.full_annotations):
            raise ValueError(f"metrics requires full_annotations at {args.full_annotations}")
        run_metrics(args)


if __name__ == "__main__":
    main()
