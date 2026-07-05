# =============================================================================
# SUFLECA
#
# SPDX-FileCopyrightText: 2023-2026 University of Luxembourg
# SPDX-License-Identifier: Apache-2.0
#
# File: sufleca/sv_align.py
#
# Copyright © 2023-2026 University of Luxembourg
# Developed by Saad Ejaz at SnT/ARG.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# =============================================================================

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import cv2
import fpsample
import numpy as np
import torch
from PIL import Image

from .config import get_config_value
from .featurizer import extract_sufleca_features
from .geometry import (
    are_scales_valid,
    compute_information_matrix,
    decompose_transform,
    find_correspondences_mknn,
    geometric_consensus_mask,
    run_ransac_registration,
    solve_anisotropic_procrustes,
    voxel_clean_indices,
)
from .render_cache import load_clean_view, load_scores


logger = logging.getLogger(__name__)

# Target points are farthest-point-sampled to at most this many before retrieval
# scoring and correspondence search, bounding the per-instance cost.
N_FPS = 512


def _skip(reason: str) -> None:
    """Log why an alignment was abandoned (at DEBUG level) and return ``None``.

    :func:`align_single_view` bails out for many distinct early-exit conditions;
    routing them through here makes downstream eval skips diagnosable. Enable
    with ``logging.getLogger("sufleca.sv_align").setLevel(logging.DEBUG)``.
    """
    logger.debug("align_single_view skipped: %s", reason)
    return None


def _load_depth(depth_path: str, image_size: int) -> np.ndarray:
    """Load a 16-bit millimetre depth PNG and resize it to ``image_size``
    square (nearest-neighbour), returning metres as float32."""
    depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
    depth = depth_raw / 1000.0
    return cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_NEAREST)


def align_single_view(
    *,
    color_path: str,
    sam_mask_path: str,
    inst_id: int,
    intrinsic: np.ndarray,
    render_model_dir: Optional[str] = None,
    render_paths: Optional[list[str]] = None,
    featurizer_model: dict[str, Any],
    config: Any,
    depth_path: str,
    image_size: Optional[int] = None,
    min_mask_pixels: int = 500,
    device: str = "cuda",
) -> Optional[dict]:
    """Align one masked RGB-D (machine-depth) instance to a CAD render pool.

    Back-projects the masked object to a 3D point cloud, picks the best of the
    pool's precomputed render views by feature score, matches SUFLECA features to
    that view, and fits a rigid + anisotropic-scale transform with SupeRANSAC
    (falling back to anisotropic Procrustes when RANSAC degenerates).

    :param featurizer_model: bundle from :func:`sufleca.load_sufleca_model`.
    :param config: dict or namespace with ``image_processing``/``ransac``/
        ``correspondence`` fields (see ``configs/sufleca.yaml``).
    :param render_model_dir: render-pool model directory; if ``None`` it is
        derived from ``render_paths``.
    :returns: a result dict with predicted ``R``/``S``/``t``, the parameter
        ``info`` matrix and its ``logdet_info``, the chosen ``render_*`` fields,
        and the matched 3D point arrays; or ``None`` if alignment is abandoned
        (set the module logger to DEBUG to see why).
    """
    image_size = image_size or get_config_value(config, "image_processing.image_size")
    ransac_pct_multiplier = get_config_value(config, "ransac.pct_multiplier")
    ransac_inlier_threshold = get_config_value(config, "ransac.inlier_threshold")
    mknn_n_points = get_config_value(config, "correspondence.mknn.n_points")
    mknn_max_correspondences = get_config_value(config, "correspondence.mknn.max_correspondences")
    mknn_k = int(get_config_value(config, "correspondence.mknn.k"))
    geo_filter_beta = float(get_config_value(config, "correspondence.geo_filter.beta"))
    geo_filter_aniso_shrink = float(get_config_value(config, "correspondence.geo_filter.aniso_shrink"))
    geo_filter_rel_thresh = float(get_config_value(config, "correspondence.geo_filter.rel_thresh"))

    if not (os.path.exists(color_path) and os.path.exists(sam_mask_path) and os.path.exists(depth_path)):
        return _skip("missing color, mask, or depth file")
    if render_model_dir is None and render_paths:
        render_model_dir = os.path.dirname(os.path.dirname(render_paths[0]))
    if render_model_dir is None:
        return _skip("no render_model_dir or render_paths given")

    orig_pil = Image.open(color_path).convert("RGB")
    orig_w, orig_h = orig_pil.width, orig_pil.height
    precomp_mask = np.array(Image.open(sam_mask_path))
    if inst_id not in np.unique(precomp_mask):
        return _skip(f"inst_id {inst_id} absent from mask")

    fx = intrinsic[0, 0] * image_size / orig_w
    fy = intrinsic[1, 1] * image_size / orig_h
    cx = intrinsic[0, 2] * image_size / orig_w
    cy = intrinsic[1, 2] * image_size / orig_h

    mask = cv2.resize(
        (precomp_mask == inst_id).astype(np.uint8),
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    if mask.sum() < min_mask_pixels:
        return _skip(f"mask below min_mask_pixels ({int(mask.sum())} < {min_mask_pixels})")

    depth = _load_depth(depth_path, image_size)
    H, W = depth.shape
    ug, vg = np.meshgrid(np.arange(W), np.arange(H))
    x = (ug - cx) * depth / fx
    y = (vg - cy) * depth / fy
    points = np.stack([x, y, depth], axis=-1)

    max_depth = get_config_value(config, "image_processing.max_depth", 6.0)
    iqr_limit = get_config_value(config, "image_processing.iqr_limit")
    point_mask = (depth > 1e-6) & (depth < max_depth)
    masked_depths = points[mask & point_mask, 2]
    if len(masked_depths) < 10:
        return _skip("fewer than 10 valid masked-depth points")
    if iqr_limit and iqr_limit > 0:
        q1, q3 = np.percentile(masked_depths, [25, 75])
        iqr = q3 - q1
        depth_mask = (depth >= q1 - iqr_limit * iqr) & (depth <= q3 + iqr_limit * iqr)
    else:
        depth_mask = np.ones_like(depth, dtype=bool)
    tgt_final_mask = mask & point_mask & depth_mask

    pil_image_tgt = orig_pil.resize((image_size, image_size))
    tgt_feat_dense = extract_sufleca_features(
        pil_image_tgt,
        featurizer_model,
        resize_to=image_size,
    ).squeeze(0)

    mask_dev = torch.as_tensor(tgt_final_mask, device=tgt_feat_dense.device)
    target_features = tgt_feat_dense[mask_dev].cpu().numpy()
    target_points = points[tgt_final_mask]
    finite = np.isfinite(target_points).all(axis=1)
    target_points = target_points[finite]
    target_features = target_features[finite]
    if len(target_points) < 20:
        return _skip("fewer than 20 finite target points")

    ind = voxel_clean_indices(target_points, grid_size=32, min_count_per_voxel=2)
    target_points = target_points[ind]
    target_features = target_features[ind]
    if len(target_points) < 20:
        return _skip("fewer than 20 target points after voxel cleaning")

    n_tgt = min(N_FPS, len(target_points))
    tgt_fps_idx = (
        fpsample.bucket_fps_kdline_sampling(target_points, n_tgt, h=3)
        if len(target_points) > n_tgt
        else np.arange(len(target_points))
    )
    tgt_feat_s = target_features[tgt_fps_idx]

    scores_bundle = load_scores(
        render_model_dir,
        image_size=image_size,
        checkpoint=featurizer_model.get("checkpoint"),
    )
    if scores_bundle is None:
        return _skip("no render score bundle for model")

    render_scores = []
    for view_id, entry in scores_bundle.items():
        if view_id == "meta":
            continue
        score_val = float((tgt_feat_s @ entry["score_features"].T).max(axis=1).mean())
        render_scores.append((score_val, view_id))

    if not render_scores:
        return _skip("no scorable render views")
    render_scores.sort(key=lambda x: x[0], reverse=True)
    _, view_id = render_scores[0]

    clean = load_clean_view(
        render_model_dir,
        view_id,
        image_size=image_size,
        checkpoint=featurizer_model.get("checkpoint"),
    )
    if clean is None:
        return _skip(f"failed to load clean render view {view_id}")
    source_points = clean["clean_points"]
    source_features = clean["clean_features"]
    if len(source_points) < 20:
        return _skip("fewer than 20 source points in render view")

    final_corres, final_weights = find_correspondences_mknn(
        src_features=source_features,
        tgt_features=target_features,
        src_points=source_points,
        tgt_points=target_points,
        n_points=mknn_n_points,
        max_correspondences=mknn_max_correspondences,
        normalize_weights=True,
        k=mknn_k,
    )
    if len(final_corres) < 7:
        return _skip(f"fewer than 7 correspondences ({len(final_corres)})")

    if geo_filter_beta > 0:
        keep = geometric_consensus_mask(
            source_points[final_corres[:, 0]],
            target_points[final_corres[:, 1]],
            beta=geo_filter_beta,
            rel_thresh=geo_filter_rel_thresh,
            aniso_shrink=geo_filter_aniso_shrink,
        )
        if keep.sum() >= 7:
            final_corres = final_corres[keep]
            final_weights = final_weights[keep]
            final_weights = final_weights / (final_weights.sum() + 1e-10)

    # Object extent (radius about its own centre) - fixed, near-extremal for norm
    object_center = np.median(target_points, axis=0)
    object_size = np.percentile(np.linalg.norm(target_points - object_center, axis=1), 96.0)
    thresh = max(ransac_pct_multiplier * object_size, ransac_inlier_threshold)
    T_sr, best_mask, _ = run_ransac_registration(
        source_points,
        target_points,
        final_corres,
        final_weights,
        inlier_threshold=thresh,
        max_iterations=get_config_value(config, "ransac.max_iterations"),
    )

    fit_corres = final_corres[best_mask.ravel().astype(bool)]
    using_fallback = len(fit_corres) < 7

    # trust the ransac estimate
    if not using_fallback:
        src_in = source_points[fit_corres[:, 0]]
        tgt_in = target_points[fit_corres[:, 1]]
        R_v, S_v, t_v = decompose_transform(T_sr)
        info_v = compute_information_matrix(R_v, S_v, t_v, src_in, tgt_in)
        if not are_scales_valid(*np.diag(S_v)):
            using_fallback = True

    if using_fallback:
        # procrustes fallback: first on the ransac inliers, then on the full
        # correspondence set if those are too few or give degenerate scales
        inlier_ok = len(fit_corres) >= 7
        if inlier_ok:
            _, R_v, S_v, t_v, info_v = solve_anisotropic_procrustes(
                source_points, target_points, fit_corres,
            )
            inlier_ok = are_scales_valid(*np.diag(S_v))
        if not inlier_ok:
            fit_corres = final_corres
            _, R_v, S_v, t_v, info_v = solve_anisotropic_procrustes(
                source_points, target_points, fit_corres,
            )

    # final check: reject a fallback fit with degenerate scales or non-finite pose
    if using_fallback and not (
        are_scales_valid(*np.diag(S_v))
        and np.isfinite(R_v).all()
        and np.isfinite(t_v).all()
    ):
        return _skip("fallback produced invalid scales or non-finite R/t")

    sign, logdet_info = np.linalg.slogdet(info_v)
    result = {
        "R": R_v,
        "S": S_v,
        "t": t_v,
        "info": info_v,
        "logdet_info": float(logdet_info if sign > 0 else -np.inf),
        "render_model_dir": render_model_dir,
        "render_view": view_id,
        "render_path": os.path.join(render_model_dir, "renders", f"render_{view_id}.png"),
        "match_source_points": source_points[fit_corres[:, 0]].astype(np.float32),
        "match_target_points": target_points[fit_corres[:, 1]].astype(np.float32),
    }
    if using_fallback:
        result["fallback"] = "procrustes_all"
    return result
