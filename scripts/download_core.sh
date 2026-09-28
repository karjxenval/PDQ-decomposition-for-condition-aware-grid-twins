#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT="${1:-data}"
MAX_GB="${2:-5}"
python scripts/download_all.py --profile core --root "$DATA_ROOT" --max-gb "$MAX_GB"
python scripts/audit_data.py --root "$DATA_ROOT"
python scripts/prepare_era.py --root "$DATA_ROOT"
python scripts/prepare_csv_measurements.py --root "$DATA_ROOT"
