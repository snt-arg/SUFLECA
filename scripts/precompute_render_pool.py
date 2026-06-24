#!/usr/bin/env python3
"""Precompute the cached SUFLECA features used during render-target selection.

Consumes the per-view assets written by ``render_cads.py`` and writes, per CAD:

    <cat>/<mid>/precomputed/scores.npz
        view_ids       (V,)          str
        score_features (V, N_FPS, D) float32
        meta           JSON metadata

    <cat>/<mid>/precomputed/clean_XX.npz   (one per render)
        clean_points        (M, 3) float32
        clean_features      (M, D) float32
        meta                JSON metadata

At eval time SUFLECA reads one ``scores.npz`` to rank a CAD's renders, then the
single ``clean_<best>.npz`` for the chosen view. ``--workers N`` shards CADs
across N processes, each loading its own featurizer.

Example:
    python scripts/precompute_render_pool.py --checkpoint sufleca --workers 4
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import shutil
import tempfile
from itertools import islice

import cv2
import fpsample
import numpy as np
from PIL import Image

from sufleca.featurizer import extract_sufleca_features, load_sufleca_model, resolve_checkpoint
from sufleca.geometry import voxel_clean_indices
from sufleca.render_cache import clean_path_for, scores_path_for


def _view_suffix(rpath: str) -> str:
    """Extract the view id (e.g. ``"03"``) from a ``render_03.png`` path."""
    return os.path.splitext(os.path.basename(rpath))[0].split("_")[-1]


def _load_raw(rpath: str, image_size: int) -> tuple | None:
    """Load a render's RGB, object-frame pointmap, and foreground mask."""
    view = _view_suffix(rpath)
    model_dir = os.path.dirname(os.path.dirname(rpath))
    pmap_path = os.path.join(model_dir, "pointmaps", f"pointmap_{view}.npy")
    rmask_path = os.path.join(model_dir, "masks", f"mask_{view}.png")
    if not (os.path.exists(pmap_path) and os.path.exists(rmask_path)):
        return None
    rpil = Image.open(rpath).convert("RGB").resize((image_size, image_size))
    rpts = cv2.resize(np.load(pmap_path).astype(np.float32), (image_size, image_size),
                      interpolation=cv2.INTER_LINEAR)
    rmask = cv2.resize((np.array(Image.open(rmask_path).convert("L")) > 0).astype(np.uint8),
                       (image_size, image_size), interpolation=cv2.INTER_NEAREST).astype(bool)
    rmask = rmask & np.isfinite(rpts).all(axis=-1)
    if rmask.sum() < 10:
        return None
    return rpil, rpts, rmask


def _fps_score_features(rpts_m: np.ndarray, rfeat_m: np.ndarray, n_fps: int) -> np.ndarray:
    """Return N_FPS features sampled using their corresponding 3D points."""
    n_src = len(rpts_m)
    if n_src > n_fps:
        idx = fpsample.bucket_fps_kdline_sampling(rpts_m, n_fps, h=3)
    else:
        idx = np.arange(n_src)
        if n_src < n_fps:
            idx = np.concatenate([idx, np.zeros(n_fps - n_src, dtype=idx.dtype)])
    return rfeat_m[idx].astype(np.float32)


def _voxel_clean(
    rpts: np.ndarray,
    rmask: np.ndarray,
    rfeat: np.ndarray,
    voxel_grid: int,
    max_fps_clean: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-clean masked render points/features, then FPS-cap to ``max_fps_clean``."""
    rpts_m, rfeat_m = rpts[rmask], rfeat[rmask]
    vc = voxel_clean_indices(rpts_m, grid_size=voxel_grid, min_count_per_voxel=1)
    pts, feat = rpts_m[vc], rfeat_m[vc]
    if max_fps_clean > 0 and len(pts) > max_fps_clean:
        idx = fpsample.bucket_fps_kdline_sampling(pts, max_fps_clean, h=3)
        pts, feat = pts[idx], feat[idx]
    return pts.astype(np.float32), feat.astype(np.float32)


def _savez_atomic(path: str, **arrays) -> None:
    """Write a cache completely before making it visible to readers."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".cache_", suffix=".npz", dir=os.path.dirname(path))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _expected_view_ids(model_dir: str, rpaths: list[str]) -> set[str]:
    """The view ids a CAD should yield, from ``metadata.json`` or its render files."""
    meta_path = os.path.join(model_dir, "metadata.json")
    try:
        with open(meta_path) as f:
            expected = json.load(f).get("view_ids")
    except (OSError, ValueError):
        expected = None
    return set(map(str, expected)) if expected else {_view_suffix(p) for p in rpaths}


def _delete_raw_assets(model_dir: str) -> None:
    """Remove a CAD's raw renders/pointmaps/masks once its caches are written."""
    for name in ("renders", "pointmaps", "masks"):
        shutil.rmtree(os.path.join(model_dir, name), ignore_errors=True)


def _worker(worker_id: int, cad_chunk: list, cfg: dict, meta_json: str) -> None:
    """Featurize and cache one chunk of CADs: per-view clean points + score features."""
    import torch

    n_gpus = torch.cuda.device_count() if cfg["device"].startswith("cuda") else 0
    device = f"cuda:{worker_id % n_gpus}" if n_gpus > 1 else cfg["device"]
    model = load_sufleca_model(checkpoint=cfg["checkpoint"], device=device)
    image_size, batch = cfg["image_size"], cfg["batch_size"]

    for model_dir, rpaths in cad_chunk:
        view_ids, score_feats = [], []
        it = iter(rpaths)
        while True:
            chunk = list(islice(it, batch))
            if not chunk:
                break
            loaded = [_load_raw(rp, image_size) for rp in chunk]
            keep = [i for i, l in enumerate(loaded) if l is not None]
            if not keep:
                continue
            pils = [loaded[i][0] for i in keep]
            feats = extract_sufleca_features(pils, model, resize_to=image_size).cpu().numpy()
            for j, i in enumerate(keep):
                rpath = chunk[i]
                _, rpts, rmask = loaded[i]
                rfeat = feats[j]
                cp, cf = _voxel_clean(rpts, rmask, rfeat, cfg["voxel_grid"], cfg["max_fps_clean"])
                _savez_atomic(clean_path_for(rpath), meta=np.array(meta_json),
                              clean_points=cp, clean_features=cf)
                sf = _fps_score_features(rpts[rmask], rfeat[rmask], cfg["n_fps"])
                view_ids.append(_view_suffix(rpath))
                score_feats.append(sf)
        expected = _expected_view_ids(model_dir, rpaths)
        if set(view_ids) != expected:
            missing = sorted(expected - set(view_ids))
            raise RuntimeError(f"incomplete raw views for {model_dir}; missing/invalid: {missing}")
        _savez_atomic(scores_path_for(model_dir), meta=np.array(meta_json),
                      view_ids=np.array(view_ids), score_features=np.stack(score_feats))
        if cfg["delete_raw"]:
            _delete_raw_assets(model_dir)
        print(f"[w{worker_id}] done {model_dir}", flush=True)


def main() -> None:
    """Precompute the SUFLECA render-pool caches (clean points + score features)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-pool", default="data/render_pool")
    ap.add_argument("--checkpoint", default="sufleca", help="Variant name or path to best.pt")
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--n-fps", type=int, default=512, help="Scoring FPS sample size")
    ap.add_argument("--voxel-grid", type=int, default=32)
    ap.add_argument("--max-fps-clean", type=int, default=2048,
                    help="Cap on saved cleaned points (0 = no cap)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--delete-raw", action="store_true",
                    help="Delete renders, masks, and pointmaps after each CAD is cached")
    ap.add_argument("--limit-cads", type=int, default=0)
    args = ap.parse_args()

    pool = sorted(glob.glob(os.path.join(args.render_pool, "**", "renders", "render_*.png"),
                            recursive=True))
    by_cad: dict = {}
    for rp in pool:
        by_cad.setdefault(os.path.dirname(os.path.dirname(rp)), []).append(rp)

    work = [(md, rps) for md, rps in sorted(by_cad.items())
            if args.overwrite or not os.path.exists(scores_path_for(md))]
    if args.limit_cads > 0:
        work = work[: args.limit_cads]
    print(f"{len(work)} CADs need features ({len(by_cad) - len(work)} already done).")
    if not work:
        return

    meta_json = json.dumps({
        "checkpoint": os.path.abspath(resolve_checkpoint(args.checkpoint)),
        "image_size": args.image_size,
        "n_fps": args.n_fps,
        "voxel_grid": args.voxel_grid,
        "max_fps_clean": args.max_fps_clean,
    })
    cfg = {
        "checkpoint": args.checkpoint, "device": args.device, "image_size": args.image_size,
        "n_fps": args.n_fps, "voxel_grid": args.voxel_grid,
        "max_fps_clean": args.max_fps_clean, "batch_size": args.batch_size,
        "delete_raw": args.delete_raw,
    }

    n = max(1, args.workers)
    if n == 1:
        _worker(0, work, cfg, meta_json)
        return
    ctx = mp.get_context("spawn")
    procs = []
    for wid in range(n):
        chunk = work[wid::n]
        if not chunk:
            continue
        p = ctx.Process(target=_worker, args=(wid, chunk, cfg, meta_json))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    failed = [p.pid for p in procs if p.exitcode != 0]
    if failed:
        raise SystemExit(f"precompute workers failed: {failed}")


if __name__ == "__main__":
    main()
