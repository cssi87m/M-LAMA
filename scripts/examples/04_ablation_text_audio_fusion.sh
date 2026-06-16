#!/usr/bin/env bash
set -euo pipefail

# Example: measure text/audio fusion contribution by gate capture and alpha scaling.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CRITERION="${CRITERION:-fluency}"
CONFIG="${CONFIG:-Model/config/config_${CRITERION}.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/checkpoints/${CRITERION}/model_best.pth}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/ablations/text_audio_fusion/${CRITERION}}"

python3 -m Model.ablation_weight_text_audio.ablation_fusion \
  --config "$CONFIG" \
  --ckpt "$CHECKPOINT" \
  --split "$SPLIT" \
  --out_dir "$OUTPUT_DIR" \
  "$@"
