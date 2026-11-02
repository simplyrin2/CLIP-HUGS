#!/usr/bin/env bash
set -euo pipefail

# Configure these externally, for example:
#   export HUMAN_CHECKPOINT=/data/ml-hugs/.../human_final.pth
#   export SCENE_CHECKPOINT=/data/ml-hugs/.../scene_final.pth
#   export DATASET_PATH=/data/ml-hugs/data/neuman
#   export SEQUENCE=bike

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
HUMAN_CHECKPOINT="${HUMAN_CHECKPOINT:?Set HUMAN_CHECKPOINT to an ML-HUGS human checkpoint}"
SCENE_CHECKPOINT="${SCENE_CHECKPOINT:?Set SCENE_CHECKPOINT to an ML-HUGS scene checkpoint}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the Neuman dataset root}"
SEQUENCE="${SEQUENCE:-bike}"
OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"

export PYTHONPATH="${ML_HUGS_ROOT:-}:$REPO_ROOT:${PYTHONPATH:-}"

"$PYTHON" "$REPO_ROOT/pipeline/ml_hugs_style_transfer.py" \
  --human_checkpoint "$HUMAN_CHECKPOINT" \
  --scene_checkpoint "$SCENE_CHECKPOINT" \
  --dataset_path "$DATASET_PATH" \
  --sequence "$SEQUENCE" \
  --style_prompt "${STYLE_PROMPT:-a Van Gogh painting with swirling brushstrokes}" \
  --output_dir "$OUTPUT_BASE/${SEQUENCE}_style" \
  --iterations "${ITERATIONS:-8000}" \
  --render_test

"$PYTHON" "$REPO_ROOT/pipeline/ml_hugs_style_render.py" \
  --human_checkpoint "$HUMAN_CHECKPOINT" \
  --stylized_scene_checkpoint "$OUTPUT_BASE/${SEQUENCE}_style/checkpoints/scene_stylized_final.pth" \
  --dataset_path "$DATASET_PATH" \
  --sequence "$SEQUENCE" \
  --output_dir "$OUTPUT_BASE/${SEQUENCE}_style/renders" \
  --render_comparison \
  --create_video

echo "Completed. Outputs are in: $OUTPUT_BASE/${SEQUENCE}_style"
