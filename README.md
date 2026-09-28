# Condition-Aware Sparse-Sensor Full-Field Recovery for Digital Twins

[![tests](https://github.com/karjxenval/PDQ-decomposition-for-condition-aware-grid-twins/actions/workflows/tests.yml/badge.svg)](https://github.com/karjxenval/PDQ-decomposition-for-condition-aware-grid-twins/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research%20code-orange)

Reproducible code for acquiring public real-world datasets and testing stable sparse-sensor full-field recovery under limited observations.

The repository is built around two questions:

> **Can a high-dimensional physical field be reconstructed from a small number of sensors?**  
> **When recovery fails, can we identify why it failed?**

The pipeline therefore reports reconstruction error together with representation, identifiability, conditioning, amplification, regularization, and distribution-shift diagnostics.

## Repository-name note

The GitHub URL keeps the earlier name `PDQ-decomposition-for-condition-aware-grid-twins`. The current code is the cleaned validation workflow for the condition-aware sparse-recovery study. In the supplied implementation, the recovery-aware basis is labelled **RASD** and is compared with standard POD/SVD under controlled sensor sets. The repository name should not be read as a claim that the algebraic factorization `A ≈ PDQ` itself defines a new rank class.

## What is included

```text
.
├── dce_data/                    # acquisition library and provenance tools
├── configs/
│   ├── sources.json             # machine-readable data catalogue
│   └── validation.json          # quick/publication settings and seed
├── scripts/
│   ├── download_all.py
│   ├── audit_data.py
│   ├── prepare_era.py
│   ├── prepare_csv_measurements.py
│   ├── dce_real_validation.py
│   ├── build_publication_evidence.py
│   ├── run_quick.sh
│   ├── run_publication.sh
│   └── windows/                 # Windows runners
├── tests/                       # static + numerical no-data tests
├── examples/                    # synthetic smoke demonstration
├── docs/                        # scientific and reproducibility documentation
├── data/                        # local runtime data; payloads are git-ignored
├── results/                     # generated experiment results
├── publication_evidence/        # generated tables/figures/summaries
├── .github/                     # CI and issue/PR templates
├── CITATION.cff
├── LICENSE
├── requirements.txt
└── pyproject.toml
```

## Scientific safeguards

The validation code is designed to prevent the most common ways a sparse-recovery experiment can look better than it really is.

- Chronological or cross-regime splits are used rather than random shuffling of adjacent time samples.
- Centering and scaling are fitted on training data only.
- Sensor selection uses training information only.
- POD/SVD and RASD are compared using controlled sensor sets.
- Rank and ridge regularization are selected without using the final test set.
- Rank-deficient sensed bases are reported explicitly.
- The pipeline reports smallest singular value, condition number when defined, projection error, exact sensor-to-field amplification, and ridge-bias diagnostics.
- Paired comparisons and bootstrap intervals are included.
- Publication aggregation avoids treating repeated configurations from one physical record as independent experiments.
- The evidence builder does not automatically claim universal superiority from one mean result.

See [`docs/SCIENTIFIC_PLAN.md`](docs/SCIENTIFIC_PLAN.md) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Data sources

| Source | Main role | Access |
|---|---|---|
| **Ponte Moesa bridge benchmark** | primary structural sparse-to-dense field recovery | ETH Research Collection |
| **GridEye PMU data** | primary real multi-location electrical validation | Zenodo |
| **KIOS real PMU data** | supporting electrical measurements | Zenodo |
| **ERA Uganda public statistics** | Uganda calibration / external realism | official public statistics |
| **LBNL micro-PMU data** | secondary distribution-level evidence | public data index |
| **PNNL/ORNL GESL** | additional event evidence | manual access |
| **Z24 bridge benchmark** | additional structural benchmark | manual / terms-sensitive |

Raw third-party datasets are **not bundled or redistributed**. The acquisition layer records provenance and leaves restricted/manual sources manual. See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Start here

### 1. Create and activate an environment

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install

```bash
python -m pip install --upgrade pip
python -m pip install -e . pytest
```

### 3. Check that the repository works before downloading anything

```bash
pytest -q
python examples/synthetic_sparse_recovery_demo.py
```

The synthetic example is only a software smoke test. It is **not** used as scientific evidence.

### 4. Inspect the core acquisition plan

```bash
python scripts/download_all.py --profile core --root data --max-gb 5 --dry-run
```

The default 5 GB guard prevents unexpected large downloads.

### 5. Download and prepare the core public data

Linux/macOS:

```bash
bash scripts/download_core.sh
```

Windows:

```bat
scripts\windows\download_core.bat
```

Or run the steps manually:

```bash
python scripts/download_all.py --profile core --root data --max-gb 5
python scripts/audit_data.py --root data
python scripts/prepare_era.py --root data
python scripts/prepare_csv_measurements.py --root data
```

## Run the validation

### Quick mode

```bash
python scripts/dce_real_validation.py ponte --data-root data --quick
python scripts/dce_real_validation.py grideye --data-root data --quick
```

or:

```bash
bash scripts/run_quick.sh
```

Windows:

```bat
scripts\windows\quick_validation.bat
```

### Publication mode

```bash
python scripts/dce_real_validation.py ponte --data-root data --publication
python scripts/dce_real_validation.py grideye --data-root data --publication
python scripts/build_publication_evidence.py --results-root results --data-root data --out publication_evidence
```

or:

```bash
bash scripts/run_publication.sh
```

Windows:

```bat
scripts\windows\publication_validation.bat
```

## What the main experiments do

### Ponte Moesa

The distributed strain field is treated as measured dense truth. A strict subset of channels is retained as sensors and the rest of the field is reconstructed. The code compares standard POD/SVD recovery with the recovery-aware basis while keeping the scientific comparison controlled.

### GridEye

Magnitude-angle PMU pairs are transformed to real-imaginary coordinates to avoid angle wrap. PMU locations are selected as grouped physical sensors. The main cross-regime experiment uses:

```text
morning  -> training
noon     -> validation
evening  -> testing
```

Adjacent samples are not randomly mixed across these blocks.

## Diagnostics and outputs

Depending on the experiment, the code records:

- relative reconstruction error;
- projection error;
- sampled latent-map rank and nullity;
- smallest and largest singular values;
- condition number when mathematically defined;
- exact sensor-to-field amplification;
- ridge-bias factor;
- validation-selected rank and ridge strength;
- selected sensor/PMU locations;
- paired POD/SVD-versus-RASD comparisons;
- bootstrap confidence intervals;
- nested fixed-rank GridEye sensor paths;
- event/OOD checks when compatible data are present;
- publication-ready CSV, JSON, figure, and LaTeX outputs.

The interpretation separates four possible limits:

1. **representation-limited** — the low-rank space does not represent the field well enough;
2. **sensor-identifiability-limited** — the retained sensors do not stably determine the latent state;
3. **noise / distribution-shift-limited** — deployment conditions differ from training conditions;
4. **regularization-limited** — stabilization introduces non-negligible bias.

## Reproducibility

For any reported result, retain:

- the Git commit hash;
- Python and operating-system versions;
- the exact `configs/validation.json`;
- acquisition provenance and hashes;
- the run mode (`quick` or `publication`);
- the frozen `publication_evidence/` bundle used for reporting.

The acquisition scripts preserve raw files and write derived files separately. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Automated GitHub checks

GitHub Actions runs on pushes and pull requests using Python 3.10 and 3.12. It compiles the code, runs the static and numerical tests, and executes the synthetic smoke example.

Run the same checks locally with:

```bash
python -m compileall -q dce_data scripts tests examples
pytest -q
python examples/synthetic_sparse_recovery_demo.py
```

## Scope

This repository contains the **acquisition, validation, diagnostics, and evidence-building code**. The full mathematical derivation and interpretation of the recovery framework belong in the associated manuscript.

## Citation

GitHub reads [`CITATION.cff`](CITATION.cff), which enables the **Cite this repository** interface. Add the final paper DOI there once it is available.

## License and data rights

The code is currently **all rights reserved**; see [`LICENSE`](LICENSE). Third-party datasets remain subject to the original providers' terms. Do not commit downloaded raw datasets to this repository.

## Replacing the old repository

Use [`docs/GITHUB_UPLOAD.md`](docs/GITHUB_UPLOAD.md). The important point is simple: **extract the ZIP locally, then upload/copy its contents. GitHub does not automatically unpack an uploaded ZIP into the repository.**
