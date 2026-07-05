#!/usr/bin/env python3
# =============================================================================
# SUFLECA
#
# SPDX-FileCopyrightText: 2023-2026 University of Luxembourg
# SPDX-License-Identifier: Apache-2.0
#
# File: scripts/precompute_zero_templates.py
#
# Copyright © 2023-2026 University of Luxembourg
# Developed by Saad Ejaz at SnT/ARG.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# =============================================================================

"""Precompute DINOv3 template descriptors for zero-shot CAD retrieval.

For every ``synset/model_id`` in ``--model-names`` this reads the CAD's rendered
views (``<render-pool>/<cat>/<mid>/renders/render_*.png`` with matching
``masks/mask_*.png``), replaces the background with flat grey, featurizes each
view with DINOv3, and writes, per synset:

    <out-root>/<cat>/singles.npy            (T, D) float32, L2-normalized
    <out-root>/<cat>/index.json             [[model_id, view], ...] aligned to rows
    <out-root>/<cat>/dense/<mid>__<view>.npy (K, D) float16, in-mask patch feats
    <out-root>/<cat>/meta.json              {featurizer, patch_size, size, grid, dim}

``singles.npy`` is loaded fully for the coarse top-K stage; the dense files are
read on demand only for the top-K survivors during the fine re-ranking stage
(see ``sufleca.zero_shot``).

Example:
    python scripts/precompute_zero_templates.py --size 512 --batch-size 16
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sufleca.zero_shot import (
    DINOV3_MODEL_ID,
    featurize_dinov3_dense,
    load_dinov3_retriever,
    mask_to_grid,
    masked_mean_embedding,
)

# Default synsets = the nine Scan2CAD eval categories.
DEFAULT_SYNSETS = ("02747177", "02808440", "02818832", "02871439", "02933112",
                   "03001627", "03211117", "04256520", "04379243")


def _view_id(rpath: str) -> str:
    """Extract the view id (e.g. ``"03"``) from a ``render_03.png`` path."""
    return os.path.splitext(os.path.basename(rpath))[0].split("_")[-1]


def _load_template(render_path: str, mask_path: str) -> tuple | None:
    """Load a render and its mask, filling the background with flat grey (128)."""
    if not os.path.exists(mask_path):
        return None
    try:
        rgb = np.array(Image.open(render_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) > 0
    except OSError:
        return None
    if mask.shape != rgb.shape[:2] or mask.sum() < 16:
        return None
    rgb[~mask] = 128
    return Image.fromarray(rgb), mask


def _cad_views(model_dir: Path) -> list[tuple[str, str, str]]:
    """List ``(view_id, render_path, mask_path)`` triples for one CAD's renders."""
    out = []
    for rp in sorted(glob.glob(str(model_dir / "renders" / "render_*.png"))):
        view = _view_id(rp)
        out.append((view, rp, str(model_dir / "masks" / f"mask_{view}.png")))
    return out


def main() -> None:
    """Build the DINOv3 zero-shot template index (coarse + dense) over the CAD pool."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-names", default="data/model_names.txt")
    ap.add_argument("--render-pool", default="data/render_pool")
    ap.add_argument("--out-root", default="data/zero_templates/dinov3")
    ap.add_argument("--model-id", default=DINOV3_MODEL_ID)
    ap.add_argument("--synsets", nargs="*", default=list(DEFAULT_SYNSETS))
    ap.add_argument("--size", type=int, default=512, help="Square featurize size (multiple of patch size)")
    ap.add_argument("--mask-frac", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    retriever = load_dinov3_retriever(device=device, model_id=args.model_id)
    ps = int(retriever["patch_size"])
    if args.size % ps != 0:
        raise SystemExit(f"--size {args.size} is not a multiple of patch size {ps}")
    g = args.size // ps

    pool: dict[str, list[str]] = {}
    for line in Path(args.model_names).read_text().splitlines():
        line = line.strip()
        if "/" in line:
            cat, mid = line.split("/", 1)
            if cat in args.synsets:
                pool.setdefault(cat, []).append(mid)
    print(f"Pool: {sum(len(v) for v in pool.values())} CADs across {len(pool)} synsets")

    for cat in args.synsets:
        mids = pool.get(cat, [])
        out_dir = Path(args.out_root) / cat
        if not mids:
            print(f"[{cat}] no CADs — skipping")
            continue
        if (out_dir / "singles.npy").exists() and not args.overwrite:
            print(f"[{cat}] already done — skipping (use --overwrite)")
            continue
        (out_dir / "dense").mkdir(parents=True, exist_ok=True)

        singles, index, n_cad, n_tpl, t0 = [], [], 0, 0, time.time()
        for mid in mids:
            views = _cad_views(Path(args.render_pool) / cat / mid)
            pils, masks, vids = [], [], []
            for view, rp, mp in views:
                loaded = _load_template(rp, mp)
                if loaded is not None:
                    pils.append(loaded[0])
                    masks.append(loaded[1])
                    vids.append(view)
            if not pils:
                continue
            feats = featurize_dinov3_dense(pils, retriever, size=args.size, batch_size=args.batch_size)
            for k, view in enumerate(vids):
                grid = mask_to_grid(masks[k], g, frac=args.mask_frac)
                singles.append(masked_mean_embedding(feats[k], grid))
                in_mask = feats[k][torch.from_numpy(grid).to(feats.device)]
                np.save(out_dir / "dense" / f"{mid}__{view}.npy",
                        in_mask.detach().cpu().numpy().astype(np.float16))
                index.append([mid, view])
                n_tpl += 1
            n_cad += 1
            if n_cad % 50 == 0:
                print(f"  [{cat}] {n_cad}/{len(mids)} cads, {n_tpl} templates")

        if not singles:
            print(f"[{cat}] no renders found under {args.render_pool} — skipping")
            continue
        np.save(out_dir / "singles.npy", np.stack(singles).astype(np.float32))
        (out_dir / "index.json").write_text(json.dumps(index))
        (out_dir / "meta.json").write_text(json.dumps({
            "featurizer": "dinov3", "patch_size": ps, "size": args.size,
            "grid": g, "dim": int(singles[0].shape[0]), "model_id": args.model_id,
        }))
        print(f"[{cat}] {n_cad} cads, {n_tpl} templates -> {out_dir} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
