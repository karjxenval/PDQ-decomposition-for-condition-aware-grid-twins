#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT="${1:-data}"
python scripts/dce_real_validation.py ponte --data-root "$DATA_ROOT" --publication
python scripts/dce_real_validation.py grideye --data-root "$DATA_ROOT" --publication
python scripts/build_publication_evidence.py --results-root results --data-root "$DATA_ROOT" --out publication_evidence
