import numpy as np

from scripts.dce_real_validation import (
    Scale,
    chronological_split,
    diagnostics,
    greedy_dopt_rows,
    orth,
    recover,
    svd_basis,
)


def test_orthonormalization():
    rng = np.random.default_rng(7)
    B = orth(rng.normal(size=(12, 4)))
    assert np.allclose(B.T @ B, np.eye(4), atol=1e-10)


def test_chronological_split_preserves_order():
    X = np.arange(8 * 20, dtype=float).reshape(8, 20)
    tr, va, te = chronological_split(X)
    assert tr.shape[1] + va.shape[1] + te.shape[1] == X.shape[1]
    assert np.array_equal(np.concatenate([tr, va, te], axis=1), X)


def test_training_scaler_is_invertible_on_other_blocks():
    X = np.arange(8 * 20, dtype=float).reshape(8, 20)
    tr, va, _ = chronological_split(X)
    scaler = Scale("feature").fit(tr)
    assert np.allclose(scaler.inverse(scaler.transform(va)), va)


def test_sensor_path_is_unique_and_valid():
    rng = np.random.default_rng(8)
    B = orth(rng.normal(size=(15, 5)))
    idx = greedy_dopt_rows(B, 8)
    assert len(idx) == 8
    assert len(np.unique(idx)) == 8
    assert np.all((idx >= 0) & (idx < B.shape[0]))


def test_exact_recovery_for_in_subspace_field_with_all_rows():
    rng = np.random.default_rng(9)
    B = orth(rng.normal(size=(12, 4)))
    X = B @ rng.normal(size=(4, 20))
    idx = np.arange(B.shape[0])
    Xhat, _, _ = recover(B, idx, X, gamma=0.0)
    assert np.allclose(Xhat, X, atol=1e-9)


def test_rank_deficiency_is_reported_as_zero_sigma_min():
    rng = np.random.default_rng(10)
    B = orth(rng.normal(size=(12, 4)))
    d = diagnostics(B, idx=[0, 1], gamma=1e-6)
    assert d["rank_G"] <= 2
    assert d["sigma_min"] == 0.0
    assert d["kappa"] is None


def test_svd_basis_rank():
    rng = np.random.default_rng(11)
    P = svd_basis(rng.normal(size=(10, 30)), 5)
    assert P.shape == (10, 5)
    assert np.allclose(P.T @ P, np.eye(5), atol=1e-10)
