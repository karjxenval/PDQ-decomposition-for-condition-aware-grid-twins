"""No-download demonstration of the sparse-to-dense recovery core.

This example is only a software smoke test. It is not a scientific benchmark.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dce_real_validation import diagnostics, greedy_dopt_rows, recover, svd_basis


def main():
    rng = np.random.default_rng(42)
    n_channels, rank = 40, 5
    n_train, n_test = 300, 80

    truth_basis, _ = np.linalg.qr(rng.normal(size=(n_channels, rank)))
    train = (
        truth_basis @ rng.normal(size=(rank, n_train))
        + 0.01 * rng.normal(size=(n_channels, n_train))
    )
    test = (
        truth_basis @ rng.normal(size=(rank, n_test))
        + 0.01 * rng.normal(size=(n_channels, n_test))
    )

    basis = svd_basis(train, rank)
    sensors = greedy_dopt_rows(basis, s=8)
    gamma = 1e-6
    reconstruction, _, _ = recover(basis, sensors, test, gamma)

    rel = np.linalg.norm(reconstruction - test, axis=0) / np.maximum(
        np.linalg.norm(test, axis=0), 1e-12
    )

    print("Selected sensor indices:", sensors.tolist())
    print("Mean relative error:", float(rel.mean()))
    print("Diagnostics:", diagnostics(basis, sensors, gamma))


if __name__ == "__main__":
    main()
