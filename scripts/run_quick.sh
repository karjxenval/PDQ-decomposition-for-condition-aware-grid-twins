#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT="${1:-data}"
python scripts/dce_real_validation.py ponte --data-root "$DATA_ROOT" --quick
python scripts/dce_real_validation.py grideye --data-root "$DATA_ROOT" --quick
