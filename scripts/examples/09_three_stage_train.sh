#!/usr/bin/env bash
set -euo pipefail

# Example: run the three-stage fine-tuning pipeline.
# Set RUN_STAGE1/RUN_STAGE2/RUN_STAGE3=false to skip a stage.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/Model_finetune_3_stages"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_STAGE1="${RUN_STAGE1:-true}"
RUN_STAGE2="${RUN_STAGE2:-true}"
RUN_STAGE3="${RUN_STAGE3:-true}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-64}"
STAGE1_LR="${STAGE1_LR:-5e-4}"
TEMPERATURE="${TEMPERATURE:-0.07}"

STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-model/stage1_best.pth}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-15}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-32}"
STAGE2_LR="${STAGE2_LR:-3e-4}"

STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-model/stage2_best.pth}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-30}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-32}"
STAGE3_LR="${STAGE3_LR:-1e-5}"
STAGE3_FLAGS="${STAGE3_FLAGS:-}"

if [[ "$RUN_STAGE1" == "true" ]]; then
  python3 stage_1/stage1_trainer.py \
    --epochs "$STAGE1_EPOCHS" \
    --batch_size "$STAGE1_BATCH_SIZE" \
    --lr "$STAGE1_LR" \
    --temperature "$TEMPERATURE"
fi

if [[ "$RUN_STAGE2" == "true" ]]; then
  python3 stage_2/stage2_trainer.py \
    --stage1_checkpoint "$STAGE1_CHECKPOINT" \
    --epochs "$STAGE2_EPOCHS" \
    --batch_size "$STAGE2_BATCH_SIZE" \
    --lr "$STAGE2_LR"
fi

if [[ "$RUN_STAGE3" == "true" ]]; then
  # shellcheck disable=SC2086
  python3 stage_3/stage3_trainer.py \
    --stage2_checkpoint "$STAGE2_CHECKPOINT" \
    --epochs "$STAGE3_EPOCHS" \
    --batch_size "$STAGE3_BATCH_SIZE" \
    --lr "$STAGE3_LR" \
    $STAGE3_FLAGS
fi
