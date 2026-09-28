.PHONY: install check demo dry-run quick publication

install:
	python -m pip install -e . pytest

check:
	python -m compileall -q dce_data scripts tests examples
	pytest -q

demo:
	python examples/synthetic_sparse_recovery_demo.py

dry-run:
	python scripts/download_all.py --profile core --root data --max-gb 5 --dry-run

quick:
	bash scripts/run_quick.sh

publication:
	bash scripts/run_publication.sh
