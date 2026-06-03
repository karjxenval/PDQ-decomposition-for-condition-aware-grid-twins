#!/usr/bin/env python3
"""
Sparse-sensor full-field recovery experiments for industrial digital twins.

This script generates a finite-element structural-response benchmark, fits
several reduced bases, performs sparse-sensor recovery, and saves a compact
evidence package for tables and figures.

Main outputs:
  outputs/tables/raw_results.csv
  outputs/tables/table_sensor_budget.csv
  outputs/tables/table_noise.csv
  outputs/tables/table_dropout.csv
  outputs/tables/table_placement.csv
  outputs/tables/table_ood.csv
  outputs/tables/summary_key_results.csv

  outputs/figures/fig_sensor_budget_relerr.png
  outputs/figures/fig_noise_relerr.png
  outputs/figures/fig_dropout_relerr.png
  outputs/figures/fig_placement_relerr.png
  outputs/figures/fig_ood_basis_vs_recovery.png

Dependencies:
  numpy, scipy, pandas, matplotlib
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import qr
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    seed: int = 123
    nx: int = 9
    ny: int = 5
    rank: int = 12

    n_train: int = 420
    n_val: int = 80
    n_test: int = 120

    young_modulus: float = 1.0
    area: float = 1.0
    damage_factor: float = 0.55

    sensor_budgets: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    noise_levels: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.10, 0.20)
    dropout_levels: Tuple[float, ...] = (0.0, 0.10, 0.20, 0.40)

    placement_modes: Tuple[str, ...] = (
        "random",
        "leverage",
        "qr",
        "physical",
        "poor",
    )

    gamma_multipliers: Tuple[float, ...] = (
        0.0,
        1e-10,
        1e-8,
        1e-6,
        1e-4,
        1e-2,
        1e-1,
        1.0,
    )

    n_random_trials: int = 5
    pdq_kappa_max: float = 50.0
    ridge_eps: float = 1e-12


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def relative_noise(x: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Generate noise with ||eta|| approximately level * ||x||."""
    if level <= 0:
        return np.zeros_like(x)
    eta = rng.standard_normal(size=x.shape)
    eta_norm = np.linalg.norm(eta)
    x_norm = max(np.linalg.norm(x), 1e-14)
    if eta_norm < 1e-14:
        return np.zeros_like(x)
    return eta / eta_norm * (level * x_norm)


def condition_number_from_svals(svals: np.ndarray, eps: float = 1e-14) -> float:
    if len(svals) == 0:
        return np.inf
    smax = float(np.max(svals))
    smin = float(np.min(svals))
    if smin <= eps:
        return np.inf
    return smax / smin


def finite_or_nan(x: float) -> float:
    if np.isfinite(x):
        return float(x)
    return np.nan


# ============================================================
# Finite-element truss benchmark
# ============================================================

def build_grid_truss(
    nx: int,
    ny: int,
    young_modulus: float = 1.0,
    area: float = 1.0,
    damaged: bool = False,
    damage_factor: float = 0.55,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """
    Build a 2D truss grid with horizontal, vertical, and diagonal bars.

    Boundary condition:
      all nodes on the left edge are fixed in x and y.

    Returns:
      Kff: free-free stiffness matrix
      nodes: node coordinates, shape (n_nodes, 2)
      free_dofs: global free dof indices
      elements: list of node index pairs
      fixed_dofs: global fixed dof indices
    """
    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append((float(i), float(j)))
    nodes = np.array(nodes, dtype=float)

    def node_id(i: int, j: int) -> int:
        return j * nx + i

    elements: List[Tuple[int, int]] = []

    # Horizontal and vertical bars
    for j in range(ny):
        for i in range(nx - 1):
            elements.append((node_id(i, j), node_id(i + 1, j)))
    for j in range(ny - 1):
        for i in range(nx):
            elements.append((node_id(i, j), node_id(i, j + 1)))

    # Diagonal bracing
    for j in range(ny - 1):
        for i in range(nx - 1):
            elements.append((node_id(i, j), node_id(i + 1, j + 1)))
            elements.append((node_id(i + 1, j), node_id(i, j + 1)))

    n_nodes = nodes.shape[0]
    n_dofs = 2 * n_nodes

    rows, cols, data = [], [], []

    # Damage selected central/right elements if requested
    damaged_set = set()
    if damaged:
        for e_idx, (a, b) in enumerate(elements):
            xa = nodes[a]
            xb = nodes[b]
            mid = 0.5 * (xa + xb)
            if (0.45 * (nx - 1) <= mid[0] <= 0.80 * (nx - 1)) and (0.25 * (ny - 1) <= mid[1] <= 0.75 * (ny - 1)):
                damaged_set.add(e_idx)

    for e_idx, (a, b) in enumerate(elements):
        xa = nodes[a]
        xb = nodes[b]
        dx, dy = xb - xa
        L = float(np.sqrt(dx * dx + dy * dy))
        c = dx / L
        s = dy / L

        EA = young_modulus * area
        if e_idx in damaged_set:
            EA *= damage_factor

        k = (EA / L) * np.array(
            [
                [c * c, c * s, -c * c, -c * s],
                [c * s, s * s, -c * s, -s * s],
                [-c * c, -c * s, c * c, c * s],
                [-c * s, -s * s, c * s, s * s],
            ],
            dtype=float,
        )

        dofs = [2 * a, 2 * a + 1, 2 * b, 2 * b + 1]
        for ii in range(4):
            for jj in range(4):
                rows.append(dofs[ii])
                cols.append(dofs[jj])
                data.append(k[ii, jj])

    K = coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()

    fixed = []
    for j in range(ny):
        nid = node_id(0, j)
        fixed.extend([2 * nid, 2 * nid + 1])
    fixed_dofs = np.array(sorted(fixed), dtype=int)

    all_dofs = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    Kff = K[free_dofs][:, free_dofs].tocsr()

    # Small numerical stabilization for pathological grids
    Kff = Kff + 1e-10 * csr_matrix(np.eye(Kff.shape[0]))

    return Kff, nodes, free_dofs, elements, fixed_dofs


def make_load_vector(
    nodes: np.ndarray,
    free_dofs: np.ndarray,
    nx: int,
    ny: int,
    regime: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create a global load vector and return its free-dof portion.
    Regimes:
      id: mixed downward/right-edge loads
      lateral: horizontal right-edge loads
      high: larger version of id loads
    """
    n_nodes = nodes.shape[0]
    f_global = np.zeros(2 * n_nodes)

    x_coords = nodes[:, 0]
    y_coords = nodes[:, 1]
    xmax = np.max(x_coords)
    ymax = np.max(y_coords)

    right_nodes = np.where(np.isclose(x_coords, xmax))[0]
    top_nodes = np.where(np.isclose(y_coords, ymax))[0]
    upper_right_nodes = np.intersect1d(right_nodes, np.where(y_coords >= 0.5 * ymax)[0])

    if regime in ("id", "damaged"):
        amp = rng.uniform(0.6, 1.4)
        for nid in right_nodes:
            f_global[2 * nid + 1] += -amp * rng.uniform(0.5, 1.5)
        for nid in top_nodes:
            f_global[2 * nid + 1] += -0.25 * amp * rng.uniform(0.0, 1.0)
        for nid in upper_right_nodes:
            f_global[2 * nid] += 0.15 * amp * rng.uniform(-1.0, 1.0)

    elif regime == "lateral":
        amp = rng.uniform(0.6, 1.4)
        for nid in right_nodes:
            f_global[2 * nid] += amp * rng.uniform(0.8, 1.6)
        for nid in top_nodes:
            f_global[2 * nid + 1] += -0.10 * amp * rng.uniform(0.0, 1.0)

    elif regime == "high":
        amp = rng.uniform(1.8, 2.8)
        for nid in right_nodes:
            f_global[2 * nid + 1] += -amp * rng.uniform(0.7, 1.7)
        for nid in upper_right_nodes:
            f_global[2 * nid] += 0.25 * amp * rng.uniform(-1.0, 1.0)

    else:
        raise ValueError(f"Unknown load regime: {regime}")

    return f_global[free_dofs]


def generate_snapshots(
    Kff: csr_matrix,
    nodes: np.ndarray,
    free_dofs: np.ndarray,
    nx: int,
    ny: int,
    n_cases: int,
    regime: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate displacement snapshots by solving K u = f."""
    n_free = len(free_dofs)
    X = np.zeros((n_free, n_cases), dtype=float)

    for j in range(n_cases):
        f = make_load_vector(nodes, free_dofs, nx, ny, regime, rng)
        u = spsolve(Kff, f)
        X[:, j] = np.asarray(u, dtype=float)

    return X


# ============================================================
# Basis construction
# ============================================================

def svd_basis(A: np.ndarray, r: int) -> np.ndarray:
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, :r]


def randomized_svd_basis(
    A: np.ndarray,
    r: int,
    rng: np.random.Generator,
    oversample: int = 8,
    n_iter: int = 1,
) -> np.ndarray:
    n, m = A.shape
    ell = min(r + oversample, min(n, m))
    Omega = rng.standard_normal(size=(m, ell))
    Y = A @ Omega

    for _ in range(n_iter):
        Y = A @ (A.T @ Y)

    Q, _ = np.linalg.qr(Y, mode="reduced")
    B_small = Q.T @ A
    U_small, _, _ = np.linalg.svd(B_small, full_matrices=False)
    U = Q @ U_small
    return U[:, :r]


def cur_column_basis(A: np.ndarray, r: int) -> np.ndarray:
    """
    Simple CUR-style selected-column basis using right singular vector leverage.
    Returns an orthonormal basis for selected snapshot columns.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    rr = min(r, Vt.shape[0])
    leverage = np.sum(Vt[:rr, :] ** 2, axis=0)
    chosen = np.argsort(leverage)[::-1][:r]
    C = A[:, chosen]

    Q, R = np.linalg.qr(C, mode="reduced")
    if Q.shape[1] < r or np.linalg.matrix_rank(R) < min(R.shape):
        return U[:, :r]

    return Q[:, :r]


def pdq_condition_aware_basis(
    A: np.ndarray,
    r: int,
    kappa_max: float = 50.0,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Practical PDQ-style basis from SVD with core singular values clipped
    to a prescribed condition threshold.

    This is not presented as a new rank class. It is a condition-aware
    fitted representation:
        A_r = U diag(s) V^T = P D Q
    with B = P D used for recovery.

    The scaling of D affects the ridge recovery coordinates, hence it is
    treated as part of the deployed surrogate.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s = s[:r]
    U = U[:, :r]

    if len(s) == 0:
        raise ValueError("Rank r must be positive.")

    smax = max(float(s[0]), eps)
    floor = smax / max(kappa_max, 1.0)
    d = np.sqrt(np.maximum(s, floor))
    d = np.maximum(d, eps)

    B = U * d.reshape(1, -1)

    diag = {
        "pdq_core_kappa": float(np.max(d) / max(np.min(d), eps)),
        "pdq_core_min": float(np.min(d)),
        "pdq_core_max": float(np.max(d)),
    }
    return B, diag


def fit_all_bases(A_train: np.ndarray, cfg: Config, rng: np.random.Generator) -> Dict[str, Dict]:
    bases: Dict[str, Dict] = {}

    t0 = time.perf_counter()
    bases["SVD"] = {
        "B": svd_basis(A_train, cfg.rank),
        "fit_time": time.perf_counter() - t0,
        "diagnostics": {},
    }

    t0 = time.perf_counter()
    bases["rSVD"] = {
        "B": randomized_svd_basis(A_train, cfg.rank, rng=rng),
        "fit_time": time.perf_counter() - t0,
        "diagnostics": {},
    }

    t0 = time.perf_counter()
    bases["CUR"] = {
        "B": cur_column_basis(A_train, cfg.rank),
        "fit_time": time.perf_counter() - t0,
        "diagnostics": {},
    }

    t0 = time.perf_counter()
    B_pdq, diag_pdq = pdq_condition_aware_basis(
        A_train,
        cfg.rank,
        kappa_max=cfg.pdq_kappa_max,
    )
    bases["PDQ"] = {
        "B": B_pdq,
        "fit_time": time.perf_counter() - t0,
        "diagnostics": diag_pdq,
    }

    return bases


# ============================================================
# Sensor selection
# ============================================================

def select_sensors(
    B: np.ndarray,
    s: int,
    mode: str,
    rng: np.random.Generator,
    train_rms: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = B.shape[0]
    s = int(max(1, min(s, n)))

    if mode == "random":
        return np.sort(rng.choice(n, size=s, replace=False))

    leverage = np.sum(B ** 2, axis=1)

    if mode == "leverage":
        return np.sort(np.argsort(leverage)[::-1][:s])

    if mode == "poor":
        return np.sort(np.argsort(leverage)[:s])

    if mode == "qr":
        try:
            _, _, piv = qr(B.T, pivoting=True, mode="economic")
            return np.sort(piv[:s])
        except Exception:
            return np.sort(np.argsort(leverage)[::-1][:s])

    if mode == "physical":
        if train_rms is None:
            return np.sort(np.argsort(leverage)[::-1][:s])
        # Physically informed proxy: select high-response coordinates with spacing.
        score = np.asarray(train_rms).copy()
        order = np.argsort(score)[::-1]
        chosen = []
        min_gap = max(1, n // (3 * s))
        for idx in order:
            if all(abs(int(idx) - int(j)) >= min_gap for j in chosen):
                chosen.append(int(idx))
            if len(chosen) == s:
                break
        if len(chosen) < s:
            for idx in order:
                if int(idx) not in chosen:
                    chosen.append(int(idx))
                if len(chosen) == s:
                    break
        return np.sort(np.array(chosen, dtype=int))

    raise ValueError(f"Unknown sensor selection mode: {mode}")


# ============================================================
# Recovery and metrics
# ============================================================

def recover_from_sensors(
    B: np.ndarray,
    x: np.ndarray,
    idx: np.ndarray,
    gamma: float,
    noise_level: float,
    rng: np.random.Generator,
    dropout: float = 0.0,
    ridge_eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Recover x from sparse noisy measurements at indices idx.
    """
    idx_eff = np.array(idx, dtype=int)

    if dropout > 0:
        keep_count = max(1, int(round((1.0 - dropout) * len(idx_eff))))
        keep = rng.choice(len(idx_eff), size=keep_count, replace=False)
        idx_eff = np.sort(idx_eff[keep])

    G = B[idx_eff, :]
    y_clean = x[idx_eff]
    eta = relative_noise(y_clean, noise_level, rng)
    y = y_clean + eta

    r = B.shape[1]
    H = G.T @ G + gamma * np.eye(r)
    b = G.T @ y

    t0 = time.perf_counter()

    try:
        if gamma > 0:
            zhat = np.linalg.solve(H, b)
        else:
            zhat, *_ = np.linalg.lstsq(G, y, rcond=None)
    except np.linalg.LinAlgError:
        zhat = np.linalg.pinv(H + ridge_eps * np.eye(r)) @ b

    online_time = time.perf_counter() - t0
    xhat = B @ zhat

    svals = np.linalg.svd(G, compute_uv=False)
    if len(svals) < r:
        sigma_min = 0.0
    else:
        sigma_min = float(np.min(svals))

    sigma_max = float(np.max(svals)) if len(svals) > 0 else 0.0
    kappa = condition_number_from_svals(
        np.concatenate([svals, np.zeros(max(0, r - len(svals)))]),
    )

    try:
        amp_mat = np.linalg.solve(H + ridge_eps * np.eye(r), G.T)
        amplification = float(np.linalg.norm(B, 2) * np.linalg.norm(amp_mat, 2))
    except Exception:
        amplification = np.nan

    diag = {
        "n_sensors_used": int(len(idx_eff)),
        "noise_norm": float(np.linalg.norm(eta)),
        "sigma_min_SB": finite_or_nan(sigma_min),
        "sigma_max_SB": finite_or_nan(sigma_max),
        "kappa_SB": finite_or_nan(kappa),
        "amplification": finite_or_nan(amplification),
        "online_time_sec": float(online_time),
    }

    return xhat, diag


def basis_projection_error(B: np.ndarray, x: np.ndarray) -> float:
    try:
        z, *_ = np.linalg.lstsq(B, x, rcond=None)
        xp = B @ z
    except Exception:
        xp = B @ (np.linalg.pinv(B) @ x)
    return float(np.linalg.norm(x - xp) / max(np.linalg.norm(x), 1e-14))


def compute_metrics(
    x: np.ndarray,
    xhat: np.ndarray,
    K_energy: Optional[csr_matrix] = None,
) -> Dict[str, float]:
    diff = xhat - x
    rel = np.linalg.norm(diff) / max(np.linalg.norm(x), 1e-14)
    maxerr = np.max(np.abs(diff))

    if K_energy is not None:
        num = float(diff.T @ (K_energy @ diff))
        den = float(x.T @ (K_energy @ x))
        energy_rel = np.sqrt(max(num, 0.0)) / max(np.sqrt(max(den, 0.0)), 1e-14)
    else:
        energy_rel = np.nan

    return {
        "RelErr": float(rel),
        "MaxErr": float(maxerr),
        "EnergyErr": float(energy_rel),
    }


def choose_gamma(
    B: np.ndarray,
    X_val: np.ndarray,
    K_val: Optional[csr_matrix],
    idx: np.ndarray,
    noise_level: float,
    cfg: Config,
    rng: np.random.Generator,
) -> float:
    """
    Choose gamma from a small grid using validation relative error.
    Gamma grid is scaled by mean diagonal of G^T G.
    """
    G = B[idx, :]
    scale = float(np.trace(G.T @ G) / max(B.shape[1], 1))
    scale = max(scale, 1e-12)

    best_gamma = None
    best_score = np.inf

    # Fixed noise samples per gamma for fair validation.
    val_cols = min(X_val.shape[1], 50)
    Xv = X_val[:, :val_cols]

    for mult in cfg.gamma_multipliers:
        gamma = float(mult * scale)
        errs = []

        local_rng = np.random.default_rng(9991)
        for j in range(val_cols):
            x = Xv[:, j]
            xhat, _ = recover_from_sensors(
                B=B,
                x=x,
                idx=idx,
                gamma=gamma,
                noise_level=noise_level,
                rng=local_rng,
                dropout=0.0,
                ridge_eps=cfg.ridge_eps,
            )
            errs.append(compute_metrics(x, xhat, K_val)["RelErr"])

        score = float(np.mean(errs))
        if score < best_score:
            best_score = score
            best_gamma = gamma

    return float(best_gamma if best_gamma is not None else 1e-8)


def evaluate_method(
    *,
    method: str,
    B: np.ndarray,
    fit_time: float,
    X_test: np.ndarray,
    K_test: Optional[csr_matrix],
    A_train: np.ndarray,
    placement: str,
    sensor_budget: float,
    noise_level: float,
    dropout: float,
    regime: str,
    experiment: str,
    cfg: Config,
    rng: np.random.Generator,
    X_val: Optional[np.ndarray] = None,
    K_val: Optional[csr_matrix] = None,
    trial: int = 0,
) -> List[Dict[str, float]]:
    n = B.shape[0]
    s = max(1, int(round(sensor_budget * n)))
    train_rms = np.sqrt(np.mean(A_train ** 2, axis=1))

    idx = select_sensors(
        B=B,
        s=s,
        mode=placement,
        rng=rng,
        train_rms=train_rms,
    )

    if X_val is None:
        X_val = X_test[:, : min(30, X_test.shape[1])]
    gamma = choose_gamma(
        B=B,
        X_val=X_val,
        K_val=K_val,
        idx=idx,
        noise_level=noise_level,
        cfg=cfg,
        rng=rng,
    )

    rows: List[Dict[str, float]] = []

    for j in range(X_test.shape[1]):
        x = X_test[:, j]
        xhat, diag = recover_from_sensors(
            B=B,
            x=x,
            idx=idx,
            gamma=gamma,
            noise_level=noise_level,
            rng=rng,
            dropout=dropout,
            ridge_eps=cfg.ridge_eps,
        )

        metrics = compute_metrics(x, xhat, K_test)
        b_err = basis_projection_error(B, x)

        row = {
            "experiment": experiment,
            "method": method,
            "regime": regime,
            "placement": placement,
            "sensor_budget": float(sensor_budget),
            "noise_level": float(noise_level),
            "dropout": float(dropout),
            "trial": int(trial),
            "snapshot_id": int(j),
            "gamma": float(gamma),
            "fit_time_sec": float(fit_time),
            "BasisErr": float(b_err),
            **metrics,
            **diag,
        }
        rows.append(row)

    return rows


# ============================================================
# Aggregation and plotting
# ============================================================

def aggregate_results(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metrics = [
        "RelErr",
        "MaxErr",
        "EnergyErr",
        "BasisErr",
        "sigma_min_SB",
        "kappa_SB",
        "amplification",
        "online_time_sec",
        "fit_time_sec",
        "gamma",
        "n_sensors_used",
    ]

    agg = df.groupby(group_cols)[metrics].agg(["mean", "std", "median"]).reset_index()
    agg.columns = [
        "_".join([str(c) for c in col if str(c) != ""]).rstrip("_")
        for col in agg.columns.values
    ]
    return agg


def save_core_tables(df: pd.DataFrame, tables_dir: Path) -> Dict[str, str]:
    outputs: Dict[str, str] = {}

    specs = {
        "table_sensor_budget.csv": (
            df[df["experiment"] == "sensor_budget"],
            ["experiment", "method", "sensor_budget"],
        ),
        "table_noise.csv": (
            df[df["experiment"] == "noise"],
            ["experiment", "method", "noise_level"],
        ),
        "table_dropout.csv": (
            df[df["experiment"] == "dropout"],
            ["experiment", "method", "dropout"],
        ),
        "table_placement.csv": (
            df[df["experiment"] == "placement"],
            ["experiment", "method", "placement"],
        ),
        "table_ood.csv": (
            df[df["experiment"] == "ood"],
            ["experiment", "method", "regime"],
        ),
    }

    for filename, (sub, group_cols) in specs.items():
        table = aggregate_results(sub, group_cols)
        path = tables_dir / filename
        table.to_csv(path, index=False)
        outputs[filename] = str(path)

    # Compact summary: best method by mean RelErr in each experiment key.
    summary_rows = []
    for exp in sorted(df["experiment"].unique()):
        sub = df[df["experiment"] == exp]
        group_cols = ["method"]
        if exp == "sensor_budget":
            group_cols.append("sensor_budget")
        elif exp == "noise":
            group_cols.append("noise_level")
        elif exp == "dropout":
            group_cols.append("dropout")
        elif exp == "placement":
            group_cols.append("placement")
        elif exp == "ood":
            group_cols.append("regime")

        tmp = sub.groupby(group_cols).agg(
            RelErr_mean=("RelErr", "mean"),
            RelErr_std=("RelErr", "std"),
            EnergyErr_mean=("EnergyErr", "mean"),
            kappa_SB_median=("kappa_SB", "median"),
            amplification_median=("amplification", "median"),
        ).reset_index()
        tmp["experiment"] = exp
        summary_rows.append(tmp)

    summary = pd.concat(summary_rows, ignore_index=True)
    path = tables_dir / "summary_key_results.csv"
    summary.to_csv(path, index=False)
    outputs["summary_key_results.csv"] = str(path)

    return outputs


def plot_metric_lines(
    table: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    plt.figure(figsize=(7.2, 4.6))

    for method, sub in table.groupby(group_col):
        sub = sub.sort_values(x_col)
        x = sub[x_col].values
        y = sub[y_col].values
        plt.plot(x, y, marker="o", label=str(method))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_placement_bars(table: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(8.0, 4.8))
    pivot = table.pivot_table(
        index="placement",
        columns="method",
        values="RelErr_mean",
        aggfunc="mean",
    )
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Mean relative recovery error")
    plt.xlabel("Sensor placement strategy")
    plt.title("Sensor placement effect on full-field recovery")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_ood_basis_recovery(table: pd.DataFrame, outpath: Path) -> None:
    # Keep the figure readable: compare SVD and PDQ if available.
    keep = table[table["method"].isin(["SVD", "PDQ"])].copy()
    if keep.empty:
        keep = table.copy()

    regimes = list(keep["regime"].unique())
    methods = list(keep["method"].unique())

    x_positions = np.arange(len(regimes))
    width = 0.35 / max(1, len(methods))

    plt.figure(figsize=(8.0, 4.8))

    for k, method in enumerate(methods):
        sub = keep[keep["method"] == method].set_index("regime").reindex(regimes)
        offsets = (k - (len(methods) - 1) / 2) * width
        plt.bar(
            x_positions + offsets,
            sub["RelErr_mean"].values,
            width=width,
            label=f"{method}: recovery",
            alpha=0.85,
        )
        plt.plot(
            x_positions + offsets,
            sub["BasisErr_mean"].values,
            marker="x",
            linestyle="none",
            label=f"{method}: basis error",
        )

    plt.xticks(x_positions, regimes, rotation=20, ha="right")
    plt.ylabel("Mean relative error")
    plt.title("OOD recovery error versus basis mismatch")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_core_figures(tables_dir: Path, figures_dir: Path) -> Dict[str, str]:
    outputs: Dict[str, str] = {}

    sensor_budget = pd.read_csv(tables_dir / "table_sensor_budget.csv")
    noise = pd.read_csv(tables_dir / "table_noise.csv")
    dropout = pd.read_csv(tables_dir / "table_dropout.csv")
    placement = pd.read_csv(tables_dir / "table_placement.csv")
    ood = pd.read_csv(tables_dir / "table_ood.csv")

    p = figures_dir / "fig_sensor_budget_relerr.png"
    plot_metric_lines(
        sensor_budget,
        x_col="sensor_budget",
        y_col="RelErr_mean",
        group_col="method",
        xlabel="Sensor budget s/n",
        ylabel="Mean relative recovery error",
        title="Full-field recovery versus sensor budget",
        outpath=p,
    )
    outputs[p.name] = str(p)

    p = figures_dir / "fig_noise_relerr.png"
    plot_metric_lines(
        noise,
        x_col="noise_level",
        y_col="RelErr_mean",
        group_col="method",
        xlabel="Relative sensor noise level",
        ylabel="Mean relative recovery error",
        title="Noise robustness of sparse-sensor recovery",
        outpath=p,
    )
    outputs[p.name] = str(p)

    p = figures_dir / "fig_dropout_relerr.png"
    plot_metric_lines(
        dropout,
        x_col="dropout",
        y_col="RelErr_mean",
        group_col="method",
        xlabel="Sensor dropout fraction",
        ylabel="Mean relative recovery error",
        title="Recovery degradation under sensor dropout",
        outpath=p,
    )
    outputs[p.name] = str(p)

    p = figures_dir / "fig_placement_relerr.png"
    plot_placement_bars(placement, p)
    outputs[p.name] = str(p)

    p = figures_dir / "fig_ood_basis_vs_recovery.png"
    plot_ood_basis_recovery(ood, p)
    outputs[p.name] = str(p)

    return outputs


# ============================================================
# Main experiment runner
# ============================================================

def run_experiments(cfg: Config, outdir: Path) -> None:
    rng = set_seed(cfg.seed)

    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    safe_mkdir(tables_dir)
    safe_mkdir(figures_dir)

    print("[1/8] Building finite-element digital-twin benchmark...")
    K_base, nodes, free_dofs, elements, fixed_dofs = build_grid_truss(
        nx=cfg.nx,
        ny=cfg.ny,
        young_modulus=cfg.young_modulus,
        area=cfg.area,
        damaged=False,
        damage_factor=cfg.damage_factor,
    )

    K_damaged, _, _, _, _ = build_grid_truss(
        nx=cfg.nx,
        ny=cfg.ny,
        young_modulus=cfg.young_modulus,
        area=cfg.area,
        damaged=True,
        damage_factor=cfg.damage_factor,
    )

    A_train = generate_snapshots(
        Kff=K_base,
        nodes=nodes,
        free_dofs=free_dofs,
        nx=cfg.nx,
        ny=cfg.ny,
        n_cases=cfg.n_train,
        regime="id",
        rng=rng,
    )

    A_val = generate_snapshots(
        Kff=K_base,
        nodes=nodes,
        free_dofs=free_dofs,
        nx=cfg.nx,
        ny=cfg.ny,
        n_cases=cfg.n_val,
        regime="id",
        rng=rng,
    )

    tests = {
        "in_distribution": (
            generate_snapshots(K_base, nodes, free_dofs, cfg.nx, cfg.ny, cfg.n_test, "id", rng),
            K_base,
        ),
        "lateral_loads": (
            generate_snapshots(K_base, nodes, free_dofs, cfg.nx, cfg.ny, cfg.n_test, "lateral", rng),
            K_base,
        ),
        "high_amplitude": (
            generate_snapshots(K_base, nodes, free_dofs, cfg.nx, cfg.ny, cfg.n_test, "high", rng),
            K_base,
        ),
        "damaged_structure": (
            generate_snapshots(K_damaged, nodes, free_dofs, cfg.nx, cfg.ny, cfg.n_test, "damaged", rng),
            K_damaged,
        ),
    }

    print("[2/8] Fitting reduced bases...")
    bases = fit_all_bases(A_train, cfg, rng)

    all_rows: List[Dict[str, float]] = []

    print("[3/8] Experiment 1: sensor budget...")
    X_id, K_id = tests["in_distribution"]
    for method, item in bases.items():
        B = item["B"]
        for budget in cfg.sensor_budgets:
            for trial in range(cfg.n_random_trials):
                rows = evaluate_method(
                    method=method,
                    B=B,
                    fit_time=item["fit_time"],
                    X_test=X_id,
                    K_test=K_id,
                    A_train=A_train,
                    placement="qr",
                    sensor_budget=budget,
                    noise_level=0.05,
                    dropout=0.0,
                    regime="in_distribution",
                    experiment="sensor_budget",
                    cfg=cfg,
                    rng=np.random.default_rng(cfg.seed + 1000 + trial),
                    X_val=A_val,
                    K_val=K_base,
                    trial=trial,
                )
                all_rows.extend(rows)

    print("[4/8] Experiment 2: sensor noise...")
    for method, item in bases.items():
        B = item["B"]
        for noise in cfg.noise_levels:
            rows = evaluate_method(
                method=method,
                B=B,
                fit_time=item["fit_time"],
                X_test=X_id,
                K_test=K_id,
                A_train=A_train,
                placement="qr",
                sensor_budget=0.20,
                noise_level=noise,
                dropout=0.0,
                regime="in_distribution",
                experiment="noise",
                cfg=cfg,
                rng=np.random.default_rng(cfg.seed + 2000 + int(noise * 1000)),
                X_val=A_val,
                K_val=K_base,
                trial=0,
            )
            all_rows.extend(rows)

    print("[5/8] Experiment 3: sensor dropout...")
    for method, item in bases.items():
        B = item["B"]
        for dropout in cfg.dropout_levels:
            for trial in range(cfg.n_random_trials):
                rows = evaluate_method(
                    method=method,
                    B=B,
                    fit_time=item["fit_time"],
                    X_test=X_id,
                    K_test=K_id,
                    A_train=A_train,
                    placement="qr",
                    sensor_budget=0.25,
                    noise_level=0.05,
                    dropout=dropout,
                    regime="in_distribution",
                    experiment="dropout",
                    cfg=cfg,
                    rng=np.random.default_rng(cfg.seed + 3000 + trial + int(dropout * 100)),
                    X_val=A_val,
                    K_val=K_base,
                    trial=trial,
                )
                all_rows.extend(rows)

    print("[6/8] Experiment 4: sensor placement...")
    for method, item in bases.items():
        B = item["B"]
        for placement in cfg.placement_modes:
            for trial in range(cfg.n_random_trials):
                rows = evaluate_method(
                    method=method,
                    B=B,
                    fit_time=item["fit_time"],
                    X_test=X_id,
                    K_test=K_id,
                    A_train=A_train,
                    placement=placement,
                    sensor_budget=0.20,
                    noise_level=0.05,
                    dropout=0.0,
                    regime="in_distribution",
                    experiment="placement",
                    cfg=cfg,
                    rng=np.random.default_rng(cfg.seed + 4000 + trial),
                    X_val=A_val,
                    K_val=K_base,
                    trial=trial,
                )
                all_rows.extend(rows)

    print("[7/8] Experiment 5: out-of-distribution regimes...")
    for method, item in bases.items():
        B = item["B"]
        for regime_name, (X_test, K_test) in tests.items():
            rows = evaluate_method(
                method=method,
                B=B,
                fit_time=item["fit_time"],
                X_test=X_test,
                K_test=K_test,
                A_train=A_train,
                placement="qr",
                sensor_budget=0.20,
                noise_level=0.05,
                dropout=0.0,
                regime=regime_name,
                experiment="ood",
                cfg=cfg,
                rng=np.random.default_rng(cfg.seed + 5000 + hash(regime_name) % 1000),
                X_val=A_val,
                K_val=K_base,
                trial=0,
            )
            all_rows.extend(rows)

    print("[8/8] Saving outputs...")
    df = pd.DataFrame(all_rows)

    # Attach PDQ diagnostics where useful.
    for method, item in bases.items():
        for key, value in item.get("diagnostics", {}).items():
            df.loc[df["method"] == method, key] = value

    raw_path = tables_dir / "raw_results.csv"
    df.to_csv(raw_path, index=False)

    table_outputs = save_core_tables(df, tables_dir)
    fig_outputs = save_core_figures(tables_dir, figures_dir)

    manifest = {
        "config": asdict(cfg),
        "n_free_dofs": int(A_train.shape[0]),
        "n_train_snapshots": int(A_train.shape[1]),
        "n_val_snapshots": int(A_val.shape[1]),
        "n_test_snapshots_each_regime": int(cfg.n_test),
        "methods": list(bases.keys()),
        "tables": {"raw_results.csv": str(raw_path), **table_outputs},
        "figures": fig_outputs,
        "notes": [
            "Finite-element benchmark is a 2D truss with left-edge fixed boundary conditions.",
            "All methods use the same ridge sparse-sensor recovery map.",
            "PDQ is implemented as a condition-aware scaled reduced surrogate, not as a claim of a new rank class.",
            "Tables report aggregated mean/std/median values from raw_results.csv.",
        ],
    }

    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDone.")
    print(f"Raw results: {raw_path}")
    print(f"Tables:      {tables_dir}")
    print(f"Figures:     {figures_dir}")
    print(f"Manifest:    {outdir / 'manifest.json'}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sparse-sensor digital-twin recovery experiments."
    )
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rank", type=int, default=12)
    parser.add_argument("--nx", type=int, default=9)
    parser.add_argument("--ny", type=int, default=5)
    parser.add_argument("--n_train", type=int, default=420)
    parser.add_argument("--n_val", type=int, default=80)
    parser.add_argument("--n_test", type=int, default=120)
    parser.add_argument("--n_random_trials", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        seed=args.seed,
        rank=args.rank,
        nx=args.nx,
        ny=args.ny,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        n_random_trials=args.n_random_trials,
    )

    outdir = Path(args.outdir)
    safe_mkdir(outdir)

    run_experiments(cfg, outdir)


if __name__ == "__main__":
    main()