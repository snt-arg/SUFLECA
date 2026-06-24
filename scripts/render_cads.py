#!/usr/bin/env python3
"""Render ShapeNet CADs into the SUFLECA render-pool layout (SAPIEN).

For each ``synset/model_id`` in ``--model-names`` this writes, under
``<output-root>/<synset>/<model_id>/``:

    renders/render_XX.png      RGB render of view XX
    pointmaps/pointmap_XX.npy  (H, W, 3) object-frame points, NaN outside mask
    masks/mask_XX.png          binary foreground mask
    metadata.json              {scale, center_orig, bounds:{min,max,size}, ...}

The object-frame pointmap equals ``scale * R @ (vertex - center_orig)`` with the
fixed render rotation ``R`` below, which is exactly the mapping SUFLECA's
``build_T_render_from_cad`` inverts at eval time. ``metadata.json`` records the
per-axis ``bounds.size`` (= ``scale * original_bbox_extent``) and the original
mesh bbox centre (``center_orig``); ``build_cad_centers.py`` collects the latter
into ``cad_orig_centers.json``.

Run ``precompute_render_pool.py`` afterwards to add the cached SUFLECA features.

With ``--zoom-views N`` it instead renders N zoomed-in, partial-object views
(wide elevation, off-centre crops) for the DINOv3 retrieval template index; pass
``--no-pointmaps`` since those templates need only ``renders/`` + ``masks/``.

Example:
    # Alignment render pool (canonical 6 views + pointmaps):
    python scripts/render_cads.py \
        --model-names data/model_names.txt \
        --shapenet-root /path/to/ShapeNetCore.v2 \
        --output-root data/render_pool --workers 4

    # Zoom render set for retrieval templates (48 views, no pointmaps):
    python scripts/render_cads.py \
        --model-names data/model_names.txt \
        --shapenet-root /path/to/ShapeNetCore.v2 \
        --output-root data/zero_render_templates \
        --zoom-views 48 --no-pointmaps --workers 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import numpy as np
import sapien.core as sapien
import trimesh
from PIL import Image

# Render intrinsics / clipping (square 448 px, ~42.5 deg vertical FOV).
WIDTH = HEIGHT = 448
FOVY = np.deg2rad(42.5)
NEAR, FAR = 0.1, 5.0
BBOX3D_PADDING = 0.10

# Fixed cad_orig -> render-frame rotation (must match build_T_render_from_cad).
ROTATION = np.array(
    [[0.0, 0.0, -1.0, 0.0],
     [-1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)

# The nine Scan2CAD eval categories.
SYNSET_TO_CATEGORY = {
    "02747177": "trash_bin",
    "02808440": "bathtub",
    "02818832": "bed",
    "02871439": "bookshelf",
    "02933112": "cabinet",
    "03001627": "chair",
    "03211117": "display",
    "04256520": "sofa",
    "04379243": "table",
}

# Canonical per-category extents (metres) used to set a sensible render scale.
# Categories without an entry fall back to unit max-extent normalization.
MEAN_SIZES = {
    "trash_bin": [0.250, 0.250, 0.400],
    "bed": [2.000, 1.600, 0.700],
    "bookshelf": [0.800, 0.300, 1.800],
    "cabinet": [0.800, 0.400, 1.800],
    "chair": [0.500, 0.500, 1.100],
    "display": [0.194, 0.612, 0.439],
    "sofa": [1.250, 0.500, 0.600],
    "table": [0.600, 1.000, 0.700],
}


def spherical_direction(theta_deg: float, phi_deg: float) -> np.ndarray:
    """Unit direction from azimuth ``theta`` and polar angle ``phi`` (degrees)."""
    theta, phi = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.array(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        dtype=np.float64,
    )


# Six deterministic, slightly top-down views (elevations ~54-63 deg).
VIEW_DIRECTIONS = [
    spherical_direction(18.0, 58.0),
    spherical_direction(82.0, 63.0),
    spherical_direction(146.0, 56.0),
    spherical_direction(211.0, 61.0),
    spherical_direction(287.0, 54.0),
    spherical_direction(332.0, 60.0),
]


def compute_view_directions(num_views: int | None) -> list[np.ndarray]:
    """Canonical 6 views, or `num_views` Fibonacci-spiral viewpoints."""
    if not num_views or num_views <= 0:
        return list(VIEW_DIRECTIONS)
    golden = 180.0 * (3.0 - np.sqrt(5.0))
    out = []
    for i in range(num_views):
        phi = 58.0 if num_views == 1 else 54.0 + 9.0 * (i / (num_views - 1))
        out.append(spherical_direction((i * golden) % 360.0, phi))
    return out


def compute_zoom_view_directions(num_views: int) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """`num_views` zoomed-in, partial-object viewpoints for retrieval templates.

    Each spec is ``(direction, radius_scale, target_offset_frac)``:
      - direction: golden-angle azimuth with elevation swept over ≈30°–80° (much
        wider than the canonical 54°–63° band), so views range from near side-on
        to steeply top-down.
      - radius_scale (< 1): pulls the camera closer than the framing radius so the
        object overflows the frame (partial / cropped object).
      - target_offset_frac: look-at point shifted off the object centre (fraction
        of the object's max extent) so the crop is asymmetric across views.

    Deterministic (fixed seed) so every model sees the same camera trajectory,
    matching how the canonical view set is fixed across models.
    """
    golden = 180.0 * (3.0 - np.sqrt(5.0))
    rng = np.random.default_rng(20260602)
    specs = []
    for i in range(num_views):
        theta = (i * golden + 41.0) % 360.0
        phi = 30.0 if num_views == 1 else 30.0 + 50.0 * (i / (num_views - 1))
        radius_scale = float(rng.uniform(0.45, 0.80))
        offset = rng.uniform(-0.25, 0.25, size=3)
        offset[2] *= 0.4  # keep most of the shift in the horizontal plane
        specs.append((spherical_direction(theta, phi), radius_scale, offset.astype(np.float64)))
    return specs


def preprocess_mesh(
    mesh_path: Path, category: str, temp_dir: Path
) -> tuple[Path, dict, float, np.ndarray, np.ndarray]:
    """Centre, rotate, and scale the mesh; return its render parameters
    ``(processed_obj, bbox_dict, scale, lengths, center)``."""
    mesh = trimesh.load(mesh_path, force="mesh")
    bbox = mesh.bounds.copy()
    center = (bbox[0] + bbox[1]) / 2.0   # original-frame bbox centre
    mesh.vertices -= center
    mesh.apply_transform(ROTATION)

    bounds = mesh.bounds.copy()
    lengths = np.abs(bounds[1] - bounds[0])
    safe = np.where(lengths > 1e-8, lengths, 1.0)
    if category in MEAN_SIZES:
        scale = float(np.min(np.array(MEAN_SIZES[category], np.float64) / safe))
    else:
        scale = float(1.0 / safe.max())

    processed_obj = temp_dir / "object.obj"
    mesh.export(processed_obj)
    bbox_dict = {"min": bounds[0].tolist(), "max": bounds[1].tolist()}
    return processed_obj, bbox_dict, scale, lengths, center


def build_object_pose(bbox_dict: dict, scale: float) -> np.ndarray:
    """4x4 pose centering the scaled, padded object bbox at the world origin."""
    lo = np.array(bbox_dict["min"])[:3] * scale * (1.0 + BBOX3D_PADDING)
    hi = np.array(bbox_dict["max"])[:3] * scale * (1.0 + BBOX3D_PADDING)
    lo += np.where(np.abs(lo) > 1e-8, np.sign(lo), -1.0) * 0.005
    hi += np.where(np.abs(hi) > 1e-8, np.sign(hi), 1.0) * 0.005
    Tso = np.eye(4)
    Tso[:3, 3] = (hi + lo) / 2.0
    return Tso


def object_frame_bounds(bbox_dict: dict, scale: float, object_pose: np.ndarray) -> dict:
    """Object-frame bbox ``{min, max, size}`` after centering by ``object_pose``."""
    lo = np.array(bbox_dict["min"])[:3] * scale - object_pose[:3, 3]
    hi = np.array(bbox_dict["max"])[:3] * scale - object_pose[:3, 3]
    return {"min": lo.tolist(), "max": hi.tolist(), "size": (hi - lo).tolist()}


def sample_background_rgba(foreground_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pick a muted background RGBA complementary to the object's median colour."""
    if np.any(mask):
        median = np.median(foreground_rgb[mask].astype(np.float32) / 255.0, axis=0)
        bg = 0.42 + 0.40 * (1.0 - median) + np.random.normal(0.0, 0.025, 3).astype(np.float32)
        bg = np.clip(bg, 0.56, 0.86)
    else:
        bg = np.array([0.92, 0.92, 0.92], np.float32)
    return np.concatenate([bg, [1.0]]) * 255.0


def calculate_cam_ext(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build a 4x4 look-at camera extrinsic from a position and target point."""
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0])
    left = np.cross(world_up, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    mat = np.eye(4)
    mat[:3, :3] = np.stack([forward, left, up], axis=1)
    mat[:3, 3] = cam_pos
    return mat


def render_rgb_and_pointmap(
    scene, camera, cam_pos: np.ndarray, target: np.ndarray
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Render one view: return the composited RGB, the position map, and the mask."""
    camera.set_pose(sapien.Pose.from_transformation_matrix(calculate_cam_ext(cam_pos, target)))
    scene.step()
    scene.update_render()
    camera.take_picture()

    rgba_img = (camera.get_float_texture("Color") * 255).clip(0, 255).astype(np.uint8)
    position = camera.get_float_texture("Position").astype(np.float32)
    seg = camera.get_uint32_texture("Segmentation")
    mask = (seg.sum(axis=-1) > 0) & (position[..., 3] < 1.0)
    if np.any(mask):
        rgba_img[~mask] = sample_background_rgba(rgba_img[..., :3], mask).astype(np.uint8)
    else:
        rgba_img[~mask] = np.array([235, 235, 235, 255], np.uint8)
    return Image.fromarray(rgba_img[..., :3], "RGB"), position, mask.astype(np.uint8)


def render_model(engine, mesh_path: Path, synset: str, model_id: str,
                 output_dir: Path, views: list, zoom: bool = False,
                 write_pointmaps: bool = True) -> None:
    """Render `views` for one CAD.

    When `zoom` is set, each entry of `views` is a `(direction, radius_scale,
    target_offset_frac)` spec from `compute_zoom_view_directions`; otherwise each
    entry is a unit view direction. `write_pointmaps=False` skips the pointmap
    output (retrieval templates only need renders + masks).
    """
    render_dir = output_dir / "renders"
    pointmap_dir = output_dir / "pointmaps"
    mask_dir = output_dir / "masks"
    for d in (render_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)
    if write_pointmaps:
        pointmap_dir.mkdir(parents=True, exist_ok=True)
    category = SYNSET_TO_CATEGORY[synset]

    with tempfile.TemporaryDirectory(prefix="render_") as tmp:
        processed_obj, bbox_dict, scale, lengths, center = preprocess_mesh(
            mesh_path, category, Path(tmp)
        )
        Tso = build_object_pose(bbox_dict, scale)
        object_center = Tso[:3, 3].copy()
        bounds = object_frame_bounds(bbox_dict, scale, Tso)
        (output_dir / "metadata.json").write_text(json.dumps({
            "synset": synset, "model_id": model_id, "category": category,
            "scale": float(scale), "center_orig": center.tolist(), "bounds": bounds,
            "view_ids": [f"{i:02d}" for i in range(len(views))],
        }, indent=2))

        max_length = float(np.max(lengths * scale))
        render_radius = 1.75 * max_length + 0.04

        scene = engine.create_scene()
        scene.set_timestep(1 / 100.0)
        builder = scene.create_actor_builder()
        builder.add_collision_from_file(str(processed_obj), scale=np.array([scale] * 3))
        builder.add_visual_from_file(str(processed_obj), scale=np.array([scale] * 3))
        asset = builder.build_static()
        asset.set_pose(sapien.Pose.from_transformation_matrix(Tso))
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        for pos in ([1, 2, 2], [1, -2, 2], [-1, 0, 1]):
            scene.add_point_light(pos, [1, 1, 1], shadow=True)
        camera = scene.add_camera("camera", width=WIDTH, height=HEIGHT, fovy=FOVY, near=NEAR, far=FAR)

        try:
            object_pose_inv = np.linalg.inv(Tso.astype(np.float32))
            for view_idx, view in enumerate(views):
                if zoom:
                    direction, radius_scale, offset_frac = view
                    target = object_center + offset_frac * max_length
                    cam_pos = target + render_radius * radius_scale * direction
                else:
                    direction = view
                    target = object_center
                    cam_pos = object_center + render_radius * direction
                render_pil, position, mask = render_rgb_and_pointmap(
                    scene, camera, cam_pos, target
                )
                render_pil.save(render_dir / f"render_{view_idx:02d}.png")
                Image.fromarray(mask * 255, mode="L").save(mask_dir / f"mask_{view_idx:02d}.png")
                if write_pointmaps:
                    model_matrix = camera.get_model_matrix().astype(np.float32)
                    points_world = position[..., :3] @ model_matrix[:3, :3].T + model_matrix[:3, 3]
                    pointmap = points_world @ object_pose_inv[:3, :3].T + object_pose_inv[:3, 3]
                    pointmap[mask == 0] = np.nan
                    np.save(pointmap_dir / f"pointmap_{view_idx:02d}.npy", pointmap.astype(np.float32))
        except Exception as e:
            print(f"Error rendering {synset}/{model_id}: {e}")
        finally:
            del camera, asset, builder, scene


def model_is_complete(output_dir: Path, n_views: int, write_pointmaps: bool = True,
                      accept_compact_cache: bool = False) -> bool:
    """Whether a CAD's outputs already exist (renders/masks/pointmaps, or the
    compact precomputed cache when ``accept_compact_cache``) so it can be skipped."""
    if accept_compact_cache:
        compact = (
            (output_dir / "metadata.json").exists()
            and (output_dir / "precomputed" / "scores.npz").exists()
            and len(list((output_dir / "precomputed").glob("clean_*.npz"))) >= n_views
        )
        if compact:
            return True
    ok = (
        (output_dir / "metadata.json").exists()
        and len(list((output_dir / "renders").glob("render_*.png"))) >= n_views
        and len(list((output_dir / "masks").glob("mask_*.png"))) >= n_views
    )
    if write_pointmaps:
        ok = ok and len(list((output_dir / "pointmaps").glob("pointmap_*.npy"))) >= n_views
    return ok


def load_model_entries(path: Path) -> list[tuple[str, str]]:
    """Parse ``synset/model_id`` lines, keeping only supported synsets."""
    entries = []
    skipped: dict[str, int] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or "/" not in line:
            continue
        synset, model_id = line.split("/", 1)
        if synset in SYNSET_TO_CATEGORY:
            entries.append((synset, model_id))
        else:
            skipped[synset] = skipped.get(synset, 0) + 1
    if skipped:
        summary = ", ".join(f"{synset} ({count})" for synset, count in sorted(skipped.items()))
        print(f"Skipping unsupported synsets: {summary}")
    return entries


def _worker(entries, shapenet_root, output_root, num_views, overwrite,
            zoom_views=None, write_pointmaps=True, accept_compact_cache=False) -> None:
    """Render each assigned CAD's views (skipping already-complete models)."""
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer(offscreen_only=True)
    engine.set_renderer(renderer)
    zoom = bool(zoom_views)
    views = compute_zoom_view_directions(zoom_views) if zoom else compute_view_directions(num_views)
    for synset, model_id in entries:
        mesh_path = Path(shapenet_root) / synset / model_id / "models" / "model_normalized.obj"
        output_dir = Path(output_root) / synset / model_id
        if not mesh_path.exists():
            print(f"missing mesh: {synset}/{model_id}")
            continue
        if not overwrite and model_is_complete(
            output_dir, len(views), write_pointmaps, accept_compact_cache
        ):
            continue
        render_model(engine, mesh_path, synset, model_id, output_dir, views,
                     zoom=zoom, write_pointmaps=write_pointmaps)


def main() -> None:
    """Render the CAD pool (multi-view RGB, pointmaps, masks) across worker processes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-names", type=Path, default=Path("data/model_names.txt"))
    ap.add_argument("--shapenet-root", type=Path, required=True,
                    help="Root of ShapeNetCore.v2 (<synset>/<model_id>/models/...).")
    ap.add_argument("--output-root", type=Path, default=Path("data/render_pool"))
    ap.add_argument("--num-views", type=int, default=None,
                    help="Override the canonical 6 views with N spiral viewpoints.")
    ap.add_argument("--zoom-views", type=int, default=None,
                    help="Render N zoomed-in, partial-object views for retrieval "
                         "templates (wide elevation, off-centre crop). Mutually "
                         "exclusive with --num-views.")
    ap.add_argument("--no-pointmaps", action="store_true",
                    help="Skip pointmap output (retrieval templates need only "
                         "renders + masks); bounds the disk footprint.")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="Render at most N CADs (debug).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--accept-compact-cache", action="store_true",
                    help="Treat metadata + precomputed caches as complete even when raw assets were deleted")
    args = ap.parse_args()

    if args.zoom_views and args.num_views:
        ap.error("--zoom-views and --num-views are mutually exclusive")
    write_pointmaps = not args.no_pointmaps

    entries = load_model_entries(args.model_names)
    if args.limit:
        entries = entries[: args.limit]
    mode = f"{args.zoom_views} zoom views" if args.zoom_views else (
        f"{args.num_views} spiral views" if args.num_views else "6 canonical views")
    print(f"{len(entries)} CADs to render ({mode}) into {args.output_root}")
    if not entries:
        return

    n = max(1, args.workers)
    if n == 1:
        _worker(entries, args.shapenet_root, args.output_root, args.num_views,
                args.overwrite, args.zoom_views, write_pointmaps, args.accept_compact_cache)
        return
    chunks = [entries[i::n] for i in range(n)]
    ctx = mp.get_context("spawn")
    procs = []
    for chunk in chunks:
        if not chunk:
            continue
        p = ctx.Process(target=_worker, args=(
            chunk, str(args.shapenet_root), str(args.output_root),
            args.num_views, args.overwrite, args.zoom_views, write_pointmaps,
            args.accept_compact_cache,
        ))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    failed = [p.pid for p in procs if p.exitcode != 0]
    if failed:
        raise SystemExit(f"render workers failed: {failed}")


if __name__ == "__main__":
    main()
