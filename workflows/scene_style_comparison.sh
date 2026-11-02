#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SCENE_CHECKPOINT="${SCENE_CHECKPOINT:?Set SCENE_CHECKPOINT}"
HUMAN_CHECKPOINT="${HUMAN_CHECKPOINT:?Set HUMAN_CHECKPOINT}"
DATASET_SEQ="${DATASET_SEQ:-lab}"
CFG_FILE="${CFG_FILE:-$REPO_ROOT/configs/hugs_scene.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/scene_comparison}"
export PYTHONPATH="${ML_HUGS_ROOT:-}:$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_DIR"

"$PYTHON" "$REPO_ROOT/hugs_style_scripts/style_scene_only.py" \
  --cfg_file "$CFG_FILE" \
  --dataset_seq "$DATASET_SEQ" \
  --scene_ckpt "$SCENE_CHECKPOINT" \
  --style_text "${STYLE_TEXT:-a Van Gogh painting with swirling brushstrokes}" \
  --steps "${STEPS:-3000}" \
  --out_ckpt "$OUTPUT_DIR/scene_only.pth"

"$PYTHON" "$REPO_ROOT/hugs_style_scripts/style_scene.py" \
  --cfg_file "$CFG_FILE" \
  --dataset_seq "$DATASET_SEQ" \
  --human_ckpt "$HUMAN_CHECKPOINT" \
  --scene_ckpt "$SCENE_CHECKPOINT" \
  --style_text "${STYLE_TEXT:-a Van Gogh painting with swirling brushstrokes}" \
  --steps "${STEPS:-3000}" \
  --out_ckpt "$OUTPUT_DIR/human_scene.pth"

echo "Comparison checkpoints are in: $OUTPUT_DIR"
