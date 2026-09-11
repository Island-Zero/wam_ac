#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_PYTHON="${EVAL_PYTHON:-python}"
exec "$EVAL_PYTHON" "$PROJECT_DIR/evaluate.py" \
  --config "$PROJECT_DIR/configs/local.json" \
  --methods adaptive 10-step 2-step --workers 1 \
  --suites libero_spatial --task-ids 0 --trials 1 \
  --output "$PROJECT_DIR/outputs/smoke" "$@"
