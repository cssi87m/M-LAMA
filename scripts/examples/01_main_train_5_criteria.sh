#!/usr/bin/env bash
set -euo pipefail

# Example: train the main M-LAMA model for all five criteria.
# Edit the YAML files first so data/checkpoint paths match your machine.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CRITERIA="${CRITERIA:-grammar fluency pronunciation vocabulary content}"

for criterion in $CRITERIA; do
  config="Model/config/config_${criterion}.yaml"
  echo "========================================================"
  echo "Training main M-LAMA criterion: ${criterion}"
  echo "Config: ${config}"
  echo "========================================================"
  python3 -m Model.train --config "$config" "$@"
done
