#!/usr/bin/env bash
set -euo pipefail

# Example: evaluate main M-LAMA checkpoints for all five criteria.
# If CHECKPOINT_DIR is set, each checkpoint is expected at:
#   ${CHECKPOINT_DIR}/${criterion}.pth
# Otherwise Model.test resolves checkpoint.load_checkpoint or latest best in save_dir.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CRITERIA="${CRITERIA:-grammar fluency pronunciation vocabulary content}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/preds/main}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

for criterion in $CRITERIA; do
  config="Model/config/config_${criterion}.yaml"
  out_dir="${OUTPUT_ROOT}/${criterion}"
  args=(--config "$config" --output_dir "$out_dir")

  if [[ -n "$CHECKPOINT_DIR" ]]; then
    args+=(--checkpoint "${CHECKPOINT_DIR}/${criterion}.pth")
  fi

  echo "========================================================"
  echo "Evaluating main M-LAMA criterion: ${criterion}"
  echo "Config: ${config}"
  echo "Output: ${out_dir}"
  echo "========================================================"
  python3 -m Model.test "${args[@]}" "$@"
done
