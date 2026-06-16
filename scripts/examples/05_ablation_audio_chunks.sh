#!/usr/bin/env bash
set -euo pipefail

# Example: measure temporal audio chunk and speaking-part importance.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CRITERION="${CRITERION:-fluency}"
CONFIG="${CONFIG:-Model/config/config_${CRITERION}.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/checkpoints/${CRITERION}/model_best.pth}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/ablations/audio_chunks/${CRITERION}}"

python3 -m Model.ablation_longaudio_weight.ablation_chunk_importance \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split "$SPLIT" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
