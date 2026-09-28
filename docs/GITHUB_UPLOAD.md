# Replace the current GitHub contents with this rebuild

This ZIP is only a delivery package. **Extract it first.** GitHub does not unpack a ZIP into repository files automatically.

## Safest method

1. Back up anything from the old repository that you may still need.
2. Clone the existing repository.
3. Delete the old project files, but keep the hidden `.git/` folder.
4. Extract this ZIP.
5. Copy everything from the extracted folder into the cloned repository root, including `.github/`.
6. Run the checks below.
7. Commit and push.

```bash
git clone https://github.com/karjxenval/PDQ-decomposition-for-condition-aware-grid-twins.git
cd PDQ-decomposition-for-condition-aware-grid-twins

python -m pip install -e . pytest
pytest -q
python examples/synthetic_sparse_recovery_demo.py

git status
git add -A
git commit -m "Rebuild reproducible sparse-sensor validation repository"
git push origin main
```

## If you use the GitHub website

Delete the old files first. Extract this ZIP on your computer, then upload the extracted files and folders. Make sure the hidden `.github/` folder is included so that GitHub Actions runs automatically.

## Check after upload

- `README.md` renders correctly.
- The **Actions** tab runs the test workflow successfully.
- GitHub shows **Cite this repository** from `CITATION.cff`.
- No raw datasets or generated large outputs were accidentally committed.

Suggested repository description:

> Reproducible real-data validation for condition-aware sparse-sensor full-field recovery in structural and electrical digital twins.

Suggested topics:

`digital-twin`, `sparse-sensing`, `full-field-reconstruction`, `reduced-order-modeling`, `pmu`, `structural-health-monitoring`, `scientific-computing`, `reproducible-research`
