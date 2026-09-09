#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/../.." && pwd)"

export FASTWAM_ROOT="${FASTWAM_ROOT:-$PROJECT/repos/FastWAM}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$PROJECT/models}"

python -u "$HERE/run_single_gpu.py" --gpu "${GPU_ID:-0}" "$@"
