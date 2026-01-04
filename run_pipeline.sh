#!/usr/bin/env bash
# Helper script to run the sentiment pipeline with JPype native-access flag to suppress Java warnings.
# Usage examples:
#   ./run_pipeline.sh --method lexicon
#   ./run_pipeline.sh --method transformer --model skt/kobert-base-v1

set -euo pipefail

if [ ! -d .venv ]; then
  echo "[INFO] Creating virtual environment (.venv)" >&2
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

export JAVA_TOOL_OPTIONS="--enable-native-access=ALL-UNNAMED"
export TOKENIZERS_PARALLELISM=false

echo "[INFO] JAVA_TOOL_OPTIONS=$JAVA_TOOL_OPTIONS"
echo "[INFO] TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
python -m src.main "$@"
