#!/usr/bin/env python3
# =============================================================================
# SUFLECA
#
# SPDX-FileCopyrightText: 2023-2026 University of Luxembourg
# SPDX-License-Identifier: Apache-2.0
#
# File: scripts/build_cad_centers.py
#
# Copyright © 2023-2026 University of Luxembourg
# Developed by Saad Ejaz at SnT/ARG.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# =============================================================================

"""Collect per-CAD original bbox centres into a single cad_orig_centers.json.

Walks a render pool produced by ``render_cads.py`` and reads ``center_orig``
from each ``<cat>/<mid>/metadata.json``, writing a flat mapping

    {"<cat>/<mid>": [cx, cy, cz], ...}

This is the ``data.cad_centers`` file consumed by ``evaluation/eval_sv.py``.

Example:
    python scripts/build_cad_centers.py --render-pool data/render_pool \
        --output data/cad_orig_centers.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> None:
    """Scan the render pool and write each CAD's render-frame center to disk."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-pool", default="data/render_pool")
    ap.add_argument("--output", default="data/cad_orig_centers.json")
    args = ap.parse_args()

    centers = {}
    missing = 0
    for meta_path in sorted(glob.glob(os.path.join(args.render_pool, "*", "*", "metadata.json"))):
        meta = json.load(open(meta_path))
        center = meta.get("center_orig")
        if center is None:
            missing += 1
            continue
        cat, mid = meta_path.split(os.sep)[-3:-1]
        centers[f"{cat}/{mid}"] = [float(v) for v in center]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(centers, open(args.output, "w"), indent=0)
    print(f"Wrote {len(centers)} centres to {args.output}"
          + (f" ({missing} metadata files lacked center_orig)" if missing else ""))


if __name__ == "__main__":
    main()
