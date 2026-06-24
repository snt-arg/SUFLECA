#!/usr/bin/env bash
# Build the DINOv3 zero-shot retrieval template index ("zoom" templates): per
# synset, render N zoomed-in partial-object views, featurize them, then delete
# that synset's staging renders before the next one so peak disk stays bounded to
# a single category.
#
# Output: data/zero_templates/dinov3/<synset>/{singles.npy,index.json,dense/,meta.json}
#
# Usage:
#   scripts/build_zoom_templates.sh --shapenet-root /path/to/ShapeNetCore.v2 \
#       [--synsets "03001627 04379243"] [--zoom-views 48] [--size 256] \
#       [--workers 6] [--keep-renders]
set -euo pipefail
cd "$(dirname "$0")/.."

SHAPENET_ROOT=""
MODEL_NAMES="data/model_names.txt"
STAGING="data/zero_render_templates"
OUT_ROOT="data/zero_templates/dinov3"
SYNSETS="02747177 02808440 02818832 02871439 02933112 03001627 03211117 04256520 04379243"
ZOOM_VIEWS=48
SIZE=256
WORKERS=6
KEEP_RENDERS=0
OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shapenet-root) SHAPENET_ROOT="$2"; shift 2 ;;
    --model-names)   MODEL_NAMES="$2"; shift 2 ;;
    --staging)       STAGING="$2"; shift 2 ;;
    --out-root)      OUT_ROOT="$2"; shift 2 ;;
    --synsets)       SYNSETS="$2"; shift 2 ;;
    --zoom-views)    ZOOM_VIEWS="$2"; shift 2 ;;
    --size)          SIZE="$2"; shift 2 ;;
    --workers)       WORKERS="$2"; shift 2 ;;
    --keep-renders)  KEEP_RENDERS=1; shift ;;
    --overwrite)     OVERWRITE=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$SHAPENET_ROOT" ]]; then
  echo "ERROR: --shapenet-root is required" >&2
  exit 2
fi

PY=python
OVERWRITE_FLAG=""
[[ "$OVERWRITE" == 1 ]] && OVERWRITE_FLAG="--overwrite"

for SYN in $SYNSETS; do
  echo "=== $(date '+%H:%M:%S') synset $SYN: render $ZOOM_VIEWS zoom views ==="
  # Per-synset model list keeps each render+featurize pass scoped to one category.
  SYN_MODELS="$(mktemp)"
  grep "^${SYN}/" "$MODEL_NAMES" > "$SYN_MODELS" || true
  if [[ ! -s "$SYN_MODELS" ]]; then
    echo "  [$SYN] no models in $MODEL_NAMES — skipping"
    rm -f "$SYN_MODELS"
    continue
  fi

  $PY scripts/render_cads.py \
      --model-names "$SYN_MODELS" \
      --shapenet-root "$SHAPENET_ROOT" \
      --output-root "$STAGING" \
      --zoom-views "$ZOOM_VIEWS" --no-pointmaps \
      --workers "$WORKERS" $OVERWRITE_FLAG

  echo "=== $(date '+%H:%M:%S') synset $SYN: featurize -> $OUT_ROOT ==="
  $PY scripts/precompute_zero_templates.py \
      --model-names "$SYN_MODELS" \
      --render-pool "$STAGING" \
      --out-root "$OUT_ROOT" \
      --synsets "$SYN" --size "$SIZE" $OVERWRITE_FLAG

  if [[ "$KEEP_RENDERS" == 0 ]]; then
    echo "    cleaning staging renders for $SYN (peak disk bounded to one synset)"
    rm -rf "${STAGING:?}/${SYN}"
  fi
  rm -f "$SYN_MODELS"
  echo "    staging size now: $(du -sh "$STAGING" 2>/dev/null | cut -f1 || echo 0)"
done

# Drop the staging root if we emptied it.
[[ "$KEEP_RENDERS" == 0 ]] && rmdir "$STAGING" 2>/dev/null || true
echo "=== $(date '+%H:%M:%S') done. templates -> $OUT_ROOT ==="
