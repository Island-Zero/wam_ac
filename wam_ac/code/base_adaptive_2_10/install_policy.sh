#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="${FASTWAM_ROOT:-$(cd "$HERE/../.." && pwd)/repos/FastWAM}"
ROBOTWIN_POLICY_ROOT="$FASTWAM_ROOT/third_party/RoboTwin/policy"
SOURCE="$HERE/policy/base_adaptive_2_10"
TARGET="$ROBOTWIN_POLICY_ROOT/base_adaptive_2_10"

test -f "$FASTWAM_ROOT/experiments/robotwin/fastwam_policy/deploy_policy.py" || {
  echo "Fast-WAM not found at: $FASTWAM_ROOT" >&2
  echo "Set FASTWAM_ROOT to the original Fast-WAM checkout." >&2
  exit 2
}
test -d "$ROBOTWIN_POLICY_ROOT" || { echo "RoboTwin policy directory missing: $ROBOTWIN_POLICY_ROOT" >&2; exit 2; }

if [[ -L "$TARGET" ]]; then
  [[ "$(readlink -f "$TARGET")" == "$(readlink -f "$SOURCE")" ]] || {
    echo "Conflicting symlink: $TARGET -> $(readlink "$TARGET")" >&2; exit 3;
  }
elif [[ -e "$TARGET" ]]; then
  echo "Refusing to overwrite existing path: $TARGET" >&2
  exit 3
else
  ln -s "$SOURCE" "$TARGET"
fi
echo "Policy ready: $TARGET -> $SOURCE"
