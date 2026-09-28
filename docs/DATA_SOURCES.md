# Data sources

The authoritative machine-readable catalog is `configs/sources.json`.

| Source | Role in the study | Access |
|---|---|---|
| ERA Uganda public statistics | Uganda calibration and external realism; not node-level SCADA validation | Automatically discovers public spreadsheets from ERA statistics pages |
| KIOS real PMU data (Zenodo 8343635) | Real PMU measurements; secondary electrical evidence | Automatic through the Zenodo record |
| GridEye PMU data (Zenodo 17648863) | Primary real electrical multi-location validation | Automatic through the Zenodo record |
| Ponte Moesa bridge benchmark | Primary real structural sparse-to-dense validation | Automatic for selected open bitstreams, subject to source terms |
| LBNL open micro-PMU data | Secondary distribution-PMU evidence | Extended/full profiles only |
| PNNL/ORNL GESL | Additional event evidence | Manual access workflow; no bypassing access controls |
| Z24 bridge benchmark | Optional structural benchmark | Manual; respect the source's non-commercial/reuse terms |

## Important distinction

ERA aggregate statistics are useful for checking realism of Uganda-focused experiments, but they are not a substitute for measured feeder/node SCADA or synchronized node-level phasor data.

## Do not commit raw data

The `.gitignore` file excludes downloaded raw data, derived datasets, audit outputs and generated results. The repository should contain code and metadata, not redistributed third-party payloads.
