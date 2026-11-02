#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SCENE="${CLIPGAUSSIAN_SCENE:?Set CLIPGAUSSIAN_SCENE to the Gaussian scene/data directory}"
HUGS_PLY="${HUGS_PLY:?Set HUGS_PLY to an ML-HUGS-exported PLY file}"
CLIPGAUSSIAN_ROOT="${CLIPGAUSSIAN_ROOT:?Set CLIPGAUSSIAN_ROOT to the CLIPGaussian/3D checkout}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/clipgaussian}"
STYLE_PROMPT="${STYLE_PROMPT:-Starry Night by Vincent van Gogh}"

mkdir -p "$OUTPUT_DIR"
"$PYTHON" "$REPO_ROOT/clipgaussian_core/train_style.py" \
  -s "$SCENE" \
  -m "$OUTPUT_DIR" \
  --ply_path "$HUGS_PLY" \
  --iterations "${ITERATIONS:-5000}" \
  --style_prompt "$STYLE_PROMPT"
