# Contributing

1. Do not commit raw third-party datasets.
2. Keep data preparation separate from raw acquisition.
3. Add or change a data source through `configs/sources.json` and the appropriate adapter.
4. Do not tune hyperparameters on the test set.
5. Add a test for changes that affect acquisition, configuration, or numerical utilities.
6. Document any change that can alter published numerical results.
