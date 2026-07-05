# =============================================================================
# SUFLECA
#
# SPDX-FileCopyrightText: 2023-2026 University of Luxembourg
# SPDX-License-Identifier: Apache-2.0
#
# File: sufleca/demo_utils.py
#
# Copyright © 2023-2026 University of Luxembourg
# Developed by Saad Ejaz at SnT/ARG.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# =============================================================================

"""Demo helpers: model loading (depth/SAM/Grounding DINO), and the
visualizations used by ``demo.ipynb`` (CAD re-render, correspondences, CAD
overlay, and the 3D alignment figure)."""
from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
MOGE_MODEL_ID = "Ruicheng/moge-2-vitl"
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

# Fixed cad_orig -> render-frame rotation used by scripts/render_cads.py; the
# render-frame points equal scale * R @ (vertex - center_orig).
_RENDER_ROT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass(frozen=True)
class DepthPrediction:
    """Pointmap output: saved depth PNG, pointmap ``.npy``, and pixel intrinsics."""

    depth_path: str
    pointmap_path: str
    intrinsic_px: np.ndarray


@dataclass(frozen=True)
class MaskPrediction:
    """SAM2 output: saved instance-mask PNG, its ``inst_id``, and the prompt box."""

    mask_path: str
    inst_id: int
    bbox_xyxy: list[float]


@dataclass(frozen=True)
class GroundingDinoPrediction:
    """A detected box: ``bbox_xyxy``, the matched ``label``, and confidence ``score``."""

    bbox_xyxy: list[float]
    label: str
    score: float


def _device(device: str) -> str:
    """Resolve a requested device, downgrading to CPU when CUDA is unavailable."""
    return device if device == "cpu" or torch.cuda.is_available() else "cpu"


def load_pointmap_model(model_path: str | Path = MOGE_MODEL_ID, device: str = "cuda"):
    """Load the pointmap model (MoGe-2) used to produce metric demo depth."""
    from moge.model.v2 import MoGeModel

    try:
        return MoGeModel.from_pretrained(str(model_path)).to(_device(device)).eval()
    except Exception as exc:
        raise RuntimeError(
            f"could not load the MoGe-2 pointmap model from {model_path!s}; "
            "use a valid local model directory or an accessible Hugging Face model ID"
        ) from exc


def load_grounding_dino(
    model_id: str = GROUNDING_DINO_MODEL_ID,
    device: str = "cuda",
) -> dict[str, Any]:
    """Load Grounding DINO for text-prompted bounding-box detection."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = _device(device)
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    except Exception as exc:
        raise RuntimeError(
            f"could not load Grounding DINO model {model_id!r}; "
            "check network access or the Hugging Face cache"
        ) from exc
    return {"processor": processor, "model": model, "device": device, "model_id": model_id}


@torch.inference_mode()
def detect_bbox_with_grounding_dino(
    color_path: str | Path,
    text_prompt: str,
    detector: dict[str, Any],
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> GroundingDinoPrediction:
    """Return the highest-confidence Grounding DINO box for a text prompt."""
    text_prompt = str(text_prompt).strip()
    if not text_prompt:
        raise ValueError("Grounding DINO requires a non-empty text prompt")

    image = Image.open(color_path).convert("RGB")
    processor = detector["processor"]
    model = detector["model"]
    inputs = processor(images=image, text=[[text_prompt]], return_tensors="pt").to(detector["device"])
    outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        target_sizes=[image.size[::-1]],
    )[0]
    if len(results["boxes"]) == 0:
        raise ValueError(
            f"Grounding DINO found no {text_prompt!r} above box threshold {box_threshold}; "
            "lower the threshold or use the interactive box mode"
        )

    best = int(torch.argmax(results["scores"]).item())
    label = results.get("text_labels", results.get("labels", [text_prompt]))[best]
    return GroundingDinoPrediction(
        bbox_xyxy=[float(v) for v in results["boxes"][best].detach().cpu().tolist()],
        label=str(label),
        score=float(results["scores"][best].detach().cpu().item()),
    )


def load_sam_predictor(model_id: str = SAM2_MODEL_ID, device: str = "cuda"):
    """Load SAM2 for box-prompted demo segmentation."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    try:
        predictor = SAM2ImagePredictor.from_pretrained(model_id)
    except Exception as exc:
        raise RuntimeError(
            f"could not load SAM2 model {model_id!r}; check network access or the Hugging Face cache"
        ) from exc
    if hasattr(predictor, "model"):
        predictor.model.to(_device(device))
    return predictor


def _intrinsic_to_pixels(intrinsic: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale a normalized intrinsic matrix (focal <= 10) to pixel units."""
    intrinsic = np.asarray(intrinsic, dtype=np.float64).copy()
    if intrinsic[0, 0] <= 10.0 and intrinsic[1, 1] <= 10.0:
        intrinsic[0, :] *= width
        intrinsic[1, :] *= height
        intrinsic[2, :] = [0.0, 0.0, 1.0]
    return intrinsic


@torch.inference_mode()
def predict_depth_with_pointmap(
    color_path: str | Path,
    pointmap_model,
    output_dir: str | Path,
    device: str = "cuda",
    use_fp16: bool = True,
) -> DepthPrediction:
    """Run the pointmap model and save a SUFLECA-compatible uint16 depth PNG."""
    device = _device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pil = Image.open(color_path).convert("RGB")
    rgb = np.asarray(pil)
    image = torch.from_numpy(rgb).to(device).float().permute(2, 0, 1) / 255.0
    pred = pointmap_model.infer(image, apply_mask=False, use_fp16=use_fp16 and device != "cpu")

    depth = pred["depth"].detach().float().cpu().numpy()
    points = pred["points"].detach().float().cpu().numpy()
    intrinsic = _intrinsic_to_pixels(
        pred["intrinsics"].detach().float().cpu().numpy(),
        width=pil.width,
        height=pil.height,
    )

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)

    depth_path = output_dir / "depth_mm.png"
    pointmap_path = output_dir / "pointmap.npy"
    Image.fromarray(depth_mm).save(depth_path)
    np.save(pointmap_path, points.astype(np.float32))
    return DepthPrediction(
        depth_path=str(depth_path),
        pointmap_path=str(pointmap_path),
        intrinsic_px=intrinsic,
    )


def _single_sam_mask(masks: np.ndarray) -> np.ndarray:
    """Reduce SAM2's (possibly multi-) mask output to the single largest-area mask."""
    masks = np.asarray(masks)
    masks = np.squeeze(masks)
    if masks.ndim == 2:
        return masks > 0
    flat = masks.reshape((-1,) + masks.shape[-2:])
    areas = flat.reshape(flat.shape[0], -1).sum(axis=1)
    return flat[int(np.argmax(areas))] > 0


@torch.inference_mode()
def segment_with_sam_box(
    color_path: str | Path,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    predictor,
    output_dir: str | Path,
    inst_id: int = 1,
    device: str = "cuda",
    pad_px: float = 10.0,
) -> MaskPrediction:
    """Segment one object with SAM2 from a box prompt and save an instance mask.

    The box is expanded by ``pad_px`` on each side (then clipped to the image) so
    a slightly tight detection still encloses the whole object for SAM2.
    """
    device = _device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(Image.open(color_path).convert("RGB"))
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bbox = np.array(
        [
            max(0.0, min(x1 - pad_px, width - 1.0)),
            max(0.0, min(y1 - pad_px, height - 1.0)),
            max(0.0, min(x2 + pad_px, width - 1.0)),
            max(0.0, min(y2 + pad_px, height - 1.0)),
        ],
        dtype=np.float32,
    )
    predictor.set_image(rgb)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device != "cpu" else nullcontext()
    with autocast:
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox,
            multimask_output=False,
        )

    mask = _single_sam_mask(masks)
    instance_map = np.zeros((height, width), dtype=np.uint8)
    instance_map[mask] = int(inst_id)
    mask_path = output_dir / "mask.png"
    Image.fromarray(instance_map).save(mask_path)
    return MaskPrediction(mask_path=str(mask_path), inst_id=int(inst_id), bbox_xyxy=bbox.tolist())


def bbox_from_json(path: str | Path, key: str = "bbox_xyxy") -> list[float]:
    """Read an ``[x1, y1, x2, y2]`` box from a JSON file (dict under ``key`` or a bare list)."""
    data = json.loads(Path(path).read_text())
    bbox = data[key] if isinstance(data, dict) else data
    if len(bbox) != 4:
        raise ValueError(f"{path} must contain four bbox values")
    return [float(v) for v in bbox]


def _view_suffix(render_path: str | Path) -> str:
    """Extract the view id (e.g. ``"03"``) from a ``render_03.png`` path."""
    return Path(render_path).stem.split("_")[-1]


def _model_dir_from_render(render_path: str | Path) -> Path:
    """Return the CAD model directory two levels above a render PNG path."""
    return Path(render_path).resolve().parent.parent


def _load_render_points(render_path: str | Path) -> np.ndarray:
    """Load a render view's sparse 3D points, preferring the dense pointmap cache
    and falling back to the precomputed clean points."""
    render_path = Path(render_path)
    view = _view_suffix(render_path)
    model_dir = _model_dir_from_render(render_path)
    pointmap_path = model_dir / "pointmaps" / f"pointmap_{view}.npy"
    if pointmap_path.exists():
        pointmap = np.load(pointmap_path).astype(np.float32)
        points = pointmap.reshape(-1, 3)
        return points[np.isfinite(points).all(axis=1)]

    clean_path = model_dir / "precomputed" / f"clean_{view}.npz"
    if clean_path.exists():
        with np.load(clean_path, allow_pickle=False) as npz:
            return np.asarray(npz["clean_points"], dtype=np.float32)
    raise FileNotFoundError(f"no pointmap or clean point cache for {render_path}")


def _load_cad_mesh_points(
    render_path: str | Path, shapenet_root: str | Path, num_samples: int = 80000
) -> np.ndarray:
    """Dense render-frame points sampled from the original ShapeNet mesh.

    Reads ``scale``/``center_orig`` from the CAD's render ``metadata.json`` and
    maps surface samples into the render frame (``scale * R @ (v - center_orig)``),
    the same frame as the precomputed ``clean_points`` the alignment is solved in.
    """
    import trimesh

    model_dir = _model_dir_from_render(render_path)
    meta = json.loads((model_dir / "metadata.json").read_text())
    mesh_path = Path(shapenet_root) / meta["synset"] / meta["model_id"] / "models" / "model_normalized.obj"
    mesh = trimesh.load(mesh_path, force="mesh")
    samples, _ = trimesh.sample.sample_surface(mesh, num_samples)
    center = np.asarray(meta["center_orig"], dtype=np.float64)
    return (float(meta["scale"]) * ((samples - center) @ _RENDER_ROT.T)).astype(np.float32)


def _cad_points(render_path: str | Path, shapenet_root: str | Path | None) -> np.ndarray:
    """Dense mesh points when a ShapeNet root is given, else the cached points."""
    if shapenet_root is not None:
        return _load_cad_mesh_points(render_path, shapenet_root)
    return _load_render_points(render_path)


def _transform_points(points: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    """Apply a predicted alignment ``q = R @ S @ p + t`` to render-frame points."""
    rotation = np.asarray(result["R"], dtype=np.float64)
    scale = np.asarray(result["S"], dtype=np.float64)
    translation = np.asarray(result["t"], dtype=np.float64).reshape(1, 3)
    return points @ scale.T @ rotation.T + translation


def _project_points(
    points_cam: np.ndarray, intrinsic: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pinhole-project camera-frame points, dropping those behind the camera.
    Returns ``(u, v, z)`` for the kept points."""
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-6)
    points_cam = points_cam[valid]
    z = points_cam[:, 2]
    u = intrinsic[0, 0] * points_cam[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * points_cam[:, 1] / z + intrinsic[1, 2]
    return u, v, z


def _spherical_direction(theta_deg: float, phi_deg: float) -> np.ndarray:
    """Unit view direction from azimuth ``theta`` and polar angle ``phi`` (degrees)."""
    theta, phi = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.array(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        dtype=np.float64,
    )


# The six canonical render viewpoints from scripts/render_cads.py. The render
# pool ships no images or camera params, so we reconstruct the viewpoint of the
# selected render from its view index to re-render the CAD at the same pose.
_RENDER_VIEW_DIRECTIONS = [
    _spherical_direction(18.0, 58.0),
    _spherical_direction(82.0, 63.0),
    _spherical_direction(146.0, 56.0),
    _spherical_direction(211.0, 61.0),
    _spherical_direction(287.0, 54.0),
    _spherical_direction(332.0, 60.0),
]


def _load_cad_mesh_surface(
    render_path: str | Path, shapenet_root: str | Path, num_samples: int = 120000
) -> tuple[np.ndarray, np.ndarray]:
    """Dense render-frame surface samples and normals from the ShapeNet mesh."""
    import trimesh

    model_dir = _model_dir_from_render(render_path)
    meta = json.loads((model_dir / "metadata.json").read_text())
    mesh_path = Path(shapenet_root) / meta["synset"] / meta["model_id"] / "models" / "model_normalized.obj"
    mesh = trimesh.load(mesh_path, force="mesh")
    samples, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
    center = np.asarray(meta["center_orig"], dtype=np.float64)
    points = (float(meta["scale"]) * ((samples - center) @ _RENDER_ROT.T)).astype(np.float32)
    normals = (np.asarray(mesh.face_normals)[face_idx] @ _RENDER_ROT.T).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    return points, normals


class _LookAtCamera:
    """Pinhole camera that looks at the render-frame origin from a view direction."""

    def __init__(self, points: np.ndarray, direction: np.ndarray, image_size: int, margin: float = 0.42):
        """Frame the point cloud: place the camera along ``direction``, build a
        right/down/forward basis (render-frame up is +z), and auto-fit the focal
        length so the points fill the image with a ``margin`` border."""
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        center = np.median(points, axis=0)
        radius = 2.6 * float(np.percentile(np.linalg.norm(points - center, axis=1), 99)) + 1e-3
        self.campos = center + radius * direction
        forward = center - self.campos
        forward /= np.linalg.norm(forward) + 1e-12
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(float(forward @ world_up)) > 0.95:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right) + 1e-12
        down = np.cross(forward, right)
        self.R_wc = np.stack([right, down, forward], axis=0)
        self.image_size = int(image_size)
        cam = (points - self.campos) @ self.R_wc.T
        front = cam[:, 2] > 1e-6
        radial = np.sqrt((cam[front, 0] / cam[front, 2]) ** 2 + (cam[front, 1] / cam[front, 2]) ** 2)
        self.focal = margin * image_size / (float(np.percentile(radial, 98)) + 1e-9)
        self.cx = self.cy = image_size / 2.0

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Project render-frame points to pixels, returning ``(u, v, z, valid)``."""
        cam = (np.asarray(points, dtype=np.float64) - self.campos) @ self.R_wc.T
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.focal * cam[:, 0] / z + self.cx
            v = self.focal * cam[:, 1] / z + self.cy
        valid = (z > 1e-6) & np.isfinite(u) & np.isfinite(v)
        return u, v, z, valid


def render_cad_view(
    render_path: str | Path,
    shapenet_root: str | Path | None,
    image_size: int = 448,
    base_color: tuple[int, int, int] = (205, 205, 214),
    background: tuple[int, int, int] = (193, 214, 232),
) -> tuple[Image.Image, "_LookAtCamera"]:
    """Re-render the CAD from the selected render's canonical viewpoint.

    With ``shapenet_root`` the CAD is sampled densely from the original ShapeNet
    mesh and Lambertian-shaded; otherwise it falls back to the sparse precomputed
    render-frame points shaded by depth. Returns the image and the camera so the
    same projection can place correspondences on the render.
    """
    if shapenet_root is not None:
        points, normals = _load_cad_mesh_surface(render_path, shapenet_root)
    else:
        points = _load_render_points(render_path)
        normals = None

    idx = int(_view_suffix(render_path)) % len(_RENDER_VIEW_DIRECTIONS)
    direction = _RENDER_VIEW_DIRECTIONS[idx]
    camera = _LookAtCamera(points, direction, image_size)

    if normals is not None:
        facing = normals @ direction
        front = facing > -0.05  # cull back-facing samples so they don't speckle through
        points, normals, facing = points[front], normals[front], facing[front]
        lit = np.clip(facing, 0.0, 1.0)
        shade = 0.35 + 0.65 * lit
    else:
        shade = None

    u, v, z, valid = camera.project(points)
    if shade is None:
        depth = z.copy()
        depth[~valid] = np.nan
        lo, hi = np.nanpercentile(depth, 5), np.nanpercentile(depth, 95)
        lit = np.clip(1.0 - (depth - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        shade = 0.35 + 0.65 * np.nan_to_num(lit)
    colors = np.clip(np.asarray(base_color, dtype=np.float64)[None, :] * shade[:, None], 0, 255)

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    keep = valid & (ui >= 0) & (ui < image_size) & (vi >= 0) & (vi < image_size)
    ui, vi, zk, ck = ui[keep], vi[keep], z[keep], colors[keep].astype(np.uint8)
    order = np.argsort(zk)[::-1]  # painter's order: far points first
    ui, vi, ck = ui[order], vi[order], ck[order]

    image = np.full((image_size, image_size, 3), background, dtype=np.uint8)
    # 1px splat for the thickness, exact pixels written last so they stay sharp.
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)):
        xx = np.clip(ui + dx, 0, image_size - 1)
        yy = np.clip(vi + dy, 0, image_size - 1)
        image[yy, xx] = ck
    return Image.fromarray(image), camera


def make_cad_overlay_image(
    color_path: str | Path,
    alignment_result: dict[str, Any],
    intrinsic: np.ndarray,
    output_path: str | Path | None = None,
    border_px: int = 2,
    point_radius: int = 2,
    shapenet_root: str | Path | None = None,
) -> Image.Image:
    """Overlay the aligned CAD on the RGB image with a red 2px border.

    With ``shapenet_root`` the overlay is sampled densely from the original
    ShapeNet mesh; otherwise it uses the sparse precomputed render-frame points.
    """
    image = np.asarray(Image.open(color_path).convert("RGB")).copy()
    height, width = image.shape[:2]
    points = _cad_points(alignment_result["render_path"], shapenet_root)
    points_cam = _transform_points(points, alignment_result)
    u, v, z = _project_points(points_cam, np.asarray(intrinsic, dtype=np.float64))

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    keep = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui, vi, z = ui[keep], vi[keep], z[keep]

    cad_mask = np.zeros((height, width), dtype=np.uint8)
    if len(ui):
        order = np.argsort(z)[::-1]
        cad_mask[vi[order], ui[order]] = 255
        if point_radius > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (point_radius, point_radius))
            cad_mask = cv2.dilate(cad_mask, kernel)

    overlay = image.astype(np.float32)
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    fill = cad_mask > 0
    overlay[fill] = overlay[fill] * 0.35 + red * 0.65

    if fill.any():
        ys, xs = np.where(fill)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=int(border_px))

    out = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    if output_path is not None:
        out.save(output_path)
    return out


def make_correspondence_image(
    color_path: str | Path,
    alignment_result: dict[str, Any],
    intrinsic: np.ndarray,
    shapenet_root: str | Path | None = None,
    output_path: str | Path | None = None,
    render_size: int = 448,
    gap: int = 20,
    max_lines: int = 120,
    seed: int = 0,
) -> Image.Image:
    """Draw the inlier correspondences between the input image and the CAD render.

    The left panel is the input image and the right panel is the retrieved CAD
    re-rendered at the selected viewpoint. Each retained match joins a scene point
    (green, left) to the CAD render-frame point it was matched to (red, right);
    these are the geometric-consensus inliers SupeRANSAC fit the pose to.
    """
    input_img = np.asarray(Image.open(color_path).convert("RGB"))
    height, width = input_img.shape[:2]
    render_pil, camera = render_cad_view(alignment_result["render_path"], shapenet_root, image_size=render_size)

    # Place the render to the right of the input image, scaled to the same height.
    scale_r = height / render_size
    render_w = int(round(render_size * scale_r))
    render_resized = cv2.resize(np.asarray(render_pil), (render_w, height), interpolation=cv2.INTER_AREA)
    x_off = width + gap
    canvas = np.full((height, x_off + render_w, 3), 255, dtype=np.uint8)
    canvas[:, :width] = input_img
    canvas[:, x_off:x_off + render_w] = render_resized

    src = alignment_result.get("match_source_points")
    tgt = alignment_result.get("match_target_points")
    if src is not None and tgt is not None and len(src):
        k = np.asarray(intrinsic, dtype=np.float64)
        tgt_cam = np.asarray(tgt, dtype=np.float64)
        tz = tgt_cam[:, 2]
        tu = k[0, 0] * tgt_cam[:, 0] / tz + k[0, 2]
        tv = k[1, 1] * tgt_cam[:, 1] / tz + k[1, 2]
        su, sv, sz, svalid = camera.project(np.asarray(src, dtype=np.float64))
        su = su * scale_r + x_off
        sv = sv * scale_r

        valid = (
            svalid & (tz > 1e-6)
            & np.isfinite(tu) & np.isfinite(tv) & np.isfinite(su) & np.isfinite(sv)
        )
        idx = np.where(valid)[0]
        if len(idx) > max_lines:
            idx = np.random.default_rng(seed).choice(idx, size=max_lines, replace=False)

        for i in idx:
            p_t = (int(round(tu[i])), int(round(tv[i])))
            p_s = (int(round(su[i])), int(round(sv[i])))
            cv2.line(canvas, p_t, p_s, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, p_t, 2, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, p_s, 2, (255, 0, 0), -1, cv2.LINE_AA)

    out = Image.fromarray(canvas)
    if output_path is not None:
        out.save(output_path)
    return out


def _backproject_depth(depth_m: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Back-project masked, valid depth pixels to a camera-frame point cloud."""
    height, width = depth_m.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_m.astype(np.float64)
    valid = mask & np.isfinite(z) & (z > 1e-6)
    x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
    points = np.stack([x, y, z], axis=-1)
    return points[valid]


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """Evenly subsample a point cloud to at most ``max_points`` for plotting."""
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[idx]


def make_alignment_3d_figure(
    color_path: str | Path,
    depth_path: str | Path,
    mask_path: str | Path,
    inst_id: int,
    alignment_result: dict[str, Any],
    intrinsic: np.ndarray,
    output_html: str | Path | None = None,
    max_points: int = 5000,
    shapenet_root: str | Path | None = None,
):
    """Create a 3D scene/CAD alignment visualization with camera-frame axes.

    With ``shapenet_root`` the CAD is sampled densely from the original ShapeNet
    mesh; otherwise it uses the sparse precomputed render-frame points.
    """
    import plotly.graph_objects as go

    rgb = np.asarray(Image.open(color_path).convert("RGB"))
    depth_m = np.asarray(Image.open(depth_path)).astype(np.float32) / 1000.0
    mask = np.asarray(Image.open(mask_path).convert("L")) == int(inst_id)
    if depth_m.shape != mask.shape:
        depth_m = cv2.resize(depth_m, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    scene_points = _backproject_depth(depth_m, mask, np.asarray(intrinsic, dtype=np.float64))

    colors = rgb.reshape(-1, 3)
    valid_flat = (mask & np.isfinite(depth_m) & (depth_m > 1e-6)).reshape(-1)
    scene_colors = colors[valid_flat]
    if len(scene_points) > max_points:
        idx = np.linspace(0, len(scene_points) - 1, max_points).astype(np.int64)
        scene_points = scene_points[idx]
        scene_colors = scene_colors[idx]

    cad_points = _transform_points(_cad_points(alignment_result["render_path"], shapenet_root), alignment_result)
    cad_points = _sample_points(cad_points[np.isfinite(cad_points).all(axis=1)], max_points)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=scene_points[:, 0],
            y=scene_points[:, 1],
            z=scene_points[:, 2],
            mode="markers",
            marker={"size": 2, "color": [f"rgb({r},{g},{b})" for r, g, b in scene_colors]},
            name="RGB-D object",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=cad_points[:, 0],
            y=cad_points[:, 1],
            z=cad_points[:, 2],
            mode="markers",
            marker={"size": 2, "color": "red", "opacity": 0.65},
            name="Aligned CAD",
        )
    )

    origin = np.asarray(alignment_result["t"], dtype=np.float64).reshape(3)
    rotation = np.asarray(alignment_result["R"], dtype=np.float64)
    scale = np.asarray(alignment_result["S"], dtype=np.float64)
    axis_len = float(np.nanpercentile(np.linalg.norm(cad_points - origin[None, :], axis=1), 85))
    axis_len = max(axis_len, 0.1)
    axes = [
        ("X", "red", rotation @ scale @ np.array([axis_len, 0.0, 0.0])),
        ("Y", "green", rotation @ scale @ np.array([0.0, axis_len, 0.0])),
        ("Z", "blue", rotation @ scale @ np.array([0.0, 0.0, axis_len])),
    ]
    for name, color, vec in axes:
        end = origin + vec
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines",
                line={"color": color, "width": 8},
                name=f"{name} axis",
            )
        )

    fig.update_layout(
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
            # Camera frame has +y down, so world-up is -y; orient the view that
            # way (a rotation, not a mirror) instead of plotly's default +z up.
            "camera": {
                "up": {"x": 0, "y": -1, "z": 0},
                "eye": {"x": 1.2, "y": -1.2, "z": -1.4},
            },
        },
        margin={"l": 0, "r": 0, "t": 28, "b": 0},
        legend={"orientation": "h"},
    )
    if output_html is not None:
        fig.write_html(str(output_html))
    return fig
