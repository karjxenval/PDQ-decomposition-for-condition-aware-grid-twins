# Reproducibility protocol

## Run order

1. Install the environment from `pyproject.toml` or `requirements.txt`.
2. Download data with a named profile and an explicit size guard.
3. Preserve `data/provenance/` and `data/metadata/` for the run.
4. Audit the raw files before analysis.
5. Prepare derived data without editing raw files.
6. Run `--quick` first to verify the pipeline.
7. Freeze `configs/validation.json` before the publication run.
8. Run both primary real-data validations.
9. Build publication evidence from generated result files.
10. Archive the config, commit hash, Python version, and provenance records with the results.

## Scientific safeguards

- Raw files are immutable inputs.
- Time series are split chronologically or by named regimes; adjacent samples are not randomly mixed across train/test.
- Centering and scaling are fitted on training data only.
- Rank and ridge parameters are selected from validation data only.
- Test data are reserved for final evaluation.
- Physical PMU sites are treated as grouped sensors.
- SVD and recovery-aware comparisons use the same sensor set when testing the basis effect.
- Rank-deficient sensor maps are reported explicitly rather than hidden behind a large condition number.
- A low reconstruction error is not by itself treated as evidence of stable recovery.

## Recommended record for a final run

Record at least:

```text
git commit:
python version:
operating system:
profile:
validation config SHA-256:
source catalog SHA-256:
raw inventory SHA-256:
run date:
```

For the strongest reproducibility, create a release tag after the final publication run and archive that release together with the manuscript DOI.
