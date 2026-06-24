"""Precomputed render asset loader.

Two-tier layout under each CAD's `precomputed/` directory:

    <render_pool>/<cat>/<mid>/precomputed/scores.npz
        view_ids       (V,)             str    — render suffixes, e.g. ["00", "01", ...]
        score_features (V, N_FPS, D)    float32
        meta           0-d string array (JSON config)

    <render_pool>/<cat>/<mid>/precomputed/clean_XX.npz
        clean_points        (M, 3)      float32
        clean_features      (M, D)      float32
        meta                0-d string array

An alignment call reads one `scores.npz` to rank a CAD's renders
(`load_scores`), then the single `clean_XX.npz` for the chosen render
(`load_clean`).
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


def _view_suffix(rpath: str) -> str:
    """Extract the view id (e.g. ``"03"``) from a ``render_03.png`` path."""
    return os.path.splitext(os.path.basename(rpath))[0].split("_")[-1]


def _model_dir(rpath: str) -> str:
    """Return the CAD model directory two levels above a render PNG path."""
    return os.path.dirname(os.path.dirname(rpath))


def scores_path_for(rpath_or_model_dir: str) -> str:
    """Accepts either a render PNG path or the CAD model directory."""
    if rpath_or_model_dir.endswith(".png"):
        md = _model_dir(rpath_or_model_dir)
    else:
        md = rpath_or_model_dir
    return os.path.join(md, "precomputed", "scores.npz")


def clean_path_for(rpath: str) -> str:
    """Clean-cache path for a render PNG (its model dir + view suffix)."""
    return clean_path_for_view(_model_dir(rpath), _view_suffix(rpath))


def clean_path_for_view(model_dir: str, view_id: str) -> str:
    """Return the clean-cache path without requiring a render file."""
    return os.path.join(model_dir, "precomputed", f"clean_{view_id}.npz")


def _meta_ok(meta: dict, image_size: int, checkpoint: Optional[str]) -> bool:
    """Validate a cache's metadata against the requested image size and checkpoint."""
    if meta.get("image_size") != image_size:
        return False
    if checkpoint is not None and meta.get("checkpoint"):
        if os.path.abspath(checkpoint) != meta["checkpoint"]:
            return False
    return True


def load_scores(
    rpath_or_model_dir: str,
    *,
    image_size: int,
    checkpoint: Optional[str] = None,
) -> Optional[dict]:
    """Load the per-CAD score bundle. Returns dict keyed by view suffix:

        { "00": {"score_features": (N_FPS, D)}, ... }

    plus a "meta" entry. Returns None on cache miss or metadata mismatch.
    """
    path = scores_path_for(rpath_or_model_dir)
    if not os.path.exists(path):
        return None
    try:
        npz = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if "meta" not in npz.files:
        return None
    try:
        meta = json.loads(str(npz["meta"]))
    except (TypeError, ValueError):
        return None
    if not _meta_ok(meta, image_size, checkpoint):
        return None
    view_ids = [str(v) for v in npz["view_ids"]]
    feats = np.asarray(npz["score_features"], dtype=np.float32)
    by_view = {
        v: {"score_features": feats[i]}
        for i, v in enumerate(view_ids)
    }
    by_view["meta"] = meta
    return by_view


def view_score(scores: dict, rpath: str) -> Optional[dict]:
    """Look up a single render's score arrays in a `load_scores` result."""
    return scores.get(_view_suffix(rpath))


def load_clean(
    rpath: str,
    *,
    image_size: int,
    checkpoint: Optional[str] = None,
) -> Optional[dict]:
    """Load the voxel-cleaned source arrays for a single chosen render."""
    return load_clean_view(
        _model_dir(rpath), _view_suffix(rpath),
        image_size=image_size, checkpoint=checkpoint,
    )


def load_clean_view(
    model_dir: str,
    view_id: str,
    *,
    image_size: int,
    checkpoint: Optional[str] = None,
) -> Optional[dict]:
    """Load a view cache directly; no render image needs to exist."""
    path = clean_path_for_view(model_dir, view_id)
    if not os.path.exists(path):
        return None
    try:
        npz = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if "meta" not in npz.files:
        return None
    try:
        meta = json.loads(str(npz["meta"]))
    except (TypeError, ValueError):
        return None
    if not _meta_ok(meta, image_size, checkpoint):
        return None
    return {
        "clean_points":       np.asarray(npz["clean_points"],       dtype=np.float32),
        "clean_features":     np.asarray(npz["clean_features"],     dtype=np.float32),
        "meta":               meta,
    }
