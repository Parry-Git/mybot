#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -n "${MYBOT_PYTHON:-}" ]]; then
    PYTHON="$MYBOT_PYTHON"
elif [[ -x "$HOME/miniconda3/envs/mybot/bin/python" ]]; then
    PYTHON="$HOME/miniconda3/envs/mybot/bin/python"
else
    PYTHON="$(command -v python3)"
fi
exec "$PYTHON" "$ROOT/scripts/service.py" "${@:-start}"
