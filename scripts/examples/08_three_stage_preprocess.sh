#!/usr/bin/env bash
set -euo pipefail

# Example: verify three-stage setup and optionally precompute audio features.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/Model_finetune_3_stages"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRECOMPUTE="${PRECOMPUTE:-false}"
MAX_FILES="${MAX_FILES:-}"

python3 preprocessing/verify_setup.py

if [[ "$PRECOMPUTE" == "true" ]]; then
  args=()
  if [[ -n "$MAX_FILES" ]]; then
    args+=(--max_files "$MAX_FILES")
  fi
  python3 preprocessing/precompute_audio_features.py "${args[@]}" "$@"
fi
