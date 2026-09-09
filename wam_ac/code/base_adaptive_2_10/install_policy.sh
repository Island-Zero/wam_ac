#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="${FASTWAM_ROOT:-$(cd "$HERE/../.." && pwd)/repos/FastWAM}"
ROBOTWIN_POLICY_ROOT="$FASTWAM_ROOT/third_party/RoboTwin/policy"
SOURCE="$HERE/policy/base_adaptive_2_10"
TARGET="$ROBOTWIN_POLICY_ROOT/base_adaptive_2_10"
UPSTREAM_SOURCE="$FASTWAM_ROOT/experiments/robotwin/fastwam_policy"
UPSTREAM_TARGET="$ROBOTWIN_POLICY_ROOT/fastwam_policy"

test -f "$FASTWAM_ROOT/experiments/robotwin/fastwam_policy/deploy_policy.py" || {
  echo "Fast-WAM not found at: $FASTWAM_ROOT" >&2
  echo "Set FASTWAM_ROOT to the original Fast-WAM checkout." >&2
  exit 2
}
test -d "$ROBOTWIN_POLICY_ROOT" || { echo "RoboTwin policy directory missing: $ROBOTWIN_POLICY_ROOT" >&2; exit 2; }

ensure_link() {
  local source="$1" target="$2"
  if [[ -L "$target" ]]; then
    [[ "$(readlink -f "$target")" == "$(readlink -f "$source")" ]] || {
      echo "Conflicting symlink: $target -> $(readlink "$target")" >&2; exit 3;
    }
  elif [[ -e "$target" ]]; then
    echo "Refusing to overwrite existing path: $target" >&2
    exit 3
  else
    ln -s "$source" "$target"
  fi
  echo "Policy ready: $target -> $source"
}

ensure_link "$UPSTREAM_SOURCE" "$UPSTREAM_TARGET"
ensure_link "$SOURCE" "$TARGET"
