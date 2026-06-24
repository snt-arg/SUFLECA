#!/usr/bin/env bash
# Build the Scan2CAD render pool end-to-end (the three steps that were
# previously run by hand):
#   1. render_cads.py            RGB + object-frame pointmaps + masks + metadata
#   2. precompute_render_pool.py cache the SUFLECA features the evaluator reads
#   3. build_cad_centers.py      collect per-CAD bbox centres JSON for evaluation/eval_sv.py
#
# Usage:
#   scripts/build_render_pool.sh --shapenet-root /path/to/ShapeNetCore.v2 [options]
#
# Options (all optional except --shapenet-root):
#   --shapenet-root DIR   ShapeNetCore.v2 root (required)
#   --model-names FILE    CAD list                  (default data/model_names.txt)
#   --render-pool DIR     output override (default data/render_pool_<checkpoint>)
#   --cad-centers FILE    centres JSON              (default data/cad_orig_centers.json)
#   --checkpoint NAME     featurizer checkpoint      (default sufleca)
#   --workers N           workers for steps 1 & 2    (default 4)
#   --overwrite           re-render / re-cache existing CADs
#   --keep-raw            retain renders, masks, and pointmaps after caching
# Environment:
#   PYTHON                 interpreter to use (default: python)
set -euo pipefail

MODEL_NAMES="data/model_names.txt"
RENDER_POOL=""
CAD_CENTERS="data/cad_orig_centers.json"
CHECKPOINT="sufleca"
WORKERS=4
SHAPENET_ROOT=""
OVERWRITE=""
DELETE_RAW="--delete-raw"
PYTHON_BIN="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shapenet-root) SHAPENET_ROOT="$2"; shift 2;;
    --model-names)   MODEL_NAMES="$2";   shift 2;;
    --render-pool)   RENDER_POOL="$2";   shift 2;;
    --cad-centers)   CAD_CENTERS="$2";   shift 2;;
    --checkpoint)    CHECKPOINT="$2";    shift 2;;
    --workers)       WORKERS="$2";       shift 2;;
    --overwrite)     OVERWRITE="--overwrite"; shift;;
    --keep-raw)      DELETE_RAW=""; shift;;
    -h|--help) sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d'; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

if [[ -z "$RENDER_POOL" ]]; then
  CHECKPOINT_NAME="${CHECKPOINT%/}"
  CHECKPOINT_NAME="${CHECKPOINT_NAME##*/}"
  if [[ "$CHECKPOINT_NAME" == "best.pt" || "$CHECKPOINT_NAME" == "checkpoint.pt" ]]; then
    CHECKPOINT_PARENT="${CHECKPOINT%/*}"
    CHECKPOINT_NAME="${CHECKPOINT_PARENT##*/}"
  fi
  CHECKPOINT_NAME="$(printf '%s' "$CHECKPOINT_NAME" | sed 's/[^A-Za-z0-9._-]/_/g; s/^[._-]*//; s/[._-]*$//')"
  if [[ -z "$CHECKPOINT_NAME" ]]; then
    echo "error: cannot derive render-pool name from checkpoint '$CHECKPOINT'" >&2
    exit 2
  fi
  RENDER_POOL="data/render_pool_${CHECKPOINT_NAME}"
fi

if [[ -z "$SHAPENET_ROOT" ]]; then
  echo "error: --shapenet-root is required" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Checkpoint contract: '$CHECKPOINT' caches and evaluates only with '$RENDER_POOL'."
echo "eval_sv.py derives this pool automatically when --render-pool is omitted."

echo "[1/3] Rendering CADs -> $RENDER_POOL"
"$PYTHON_BIN" "$SCRIPT_DIR/render_cads.py" \
    --model-names "$MODEL_NAMES" \
    --shapenet-root "$SHAPENET_ROOT" \
    --output-root "$RENDER_POOL" \
    --workers "$WORKERS" --accept-compact-cache $OVERWRITE

echo "[2/3] Caching SUFLECA features ($CHECKPOINT)"
"$PYTHON_BIN" "$SCRIPT_DIR/precompute_render_pool.py" \
    --render-pool "$RENDER_POOL" \
    --checkpoint "$CHECKPOINT" \
    --workers "$WORKERS" $OVERWRITE $DELETE_RAW

echo "[3/3] Collecting CAD bbox centres -> $CAD_CENTERS"
"$PYTHON_BIN" "$SCRIPT_DIR/build_cad_centers.py" \
    --render-pool "$RENDER_POOL" \
    --output "$CAD_CENTERS"

echo "Render pool ready: $RENDER_POOL (centres: $CAD_CENTERS)"
