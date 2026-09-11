#!/usr/bin/env bash
# Compatibility entry point; see scripts/evaluate_2000.sh.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PROJECT_DIR/scripts/evaluate_2000.sh" "$@"
