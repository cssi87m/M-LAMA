#!/usr/bin/env bash
set -euo pipefail

# Example: Exp2 audio-only/no-text model for all five criteria.
# MODE=train or MODE=test.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/Exp2_notext"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODE="${MODE:-test}"
CRITERIA="${CRITERIA:-grammar fluency pronunciation vocabulary content}"

if [[ "$MODE" != "train" && "$MODE" != "test" ]]; then
  echo "MODE must be train or test"
  exit 2
fi

for criterion in $CRITERIA; do
  config="config/config_${criterion}.yaml"
  echo "========================================================"
  echo "Exp2_notext ${MODE}: ${criterion}"
  echo "Config: ${config}"
  echo "========================================================"
  python3 "${MODE}.py" --config "$config" "$@"
done
