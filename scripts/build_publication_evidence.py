from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon
import matplotlib.pyplot as plt


# ----------------------------- helpers ---------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x):
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def pick(d, *names, default=np.nan):
    for n in names:
        if isinstance(d, dict) and n in d:
            return d[n]
    return default


def cell_cluster(name: str) -> str:
    m = re.match(r"cell_r(\d+)_c\d+", str(name))
    return f"r{m.group(1)}" if m else str(name)


def bootstrap_ci(values, stat=np.mean, reps=20000, seed=20260924, alpha=0.05):
    """Ordinary bootstrap CI; values are assumed to be analysis units."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    for b in range(reps):
        vals[b] = stat(rng.choice(x, size=len(x), replace=True))
    return tuple(np.quantile(vals, [alpha/2, 1-alpha/2]))


def cluster_bootstrap_ci(df, value_col, cluster_col="cluster",
                         stat=np.mean, reps=20000, seed=20260924, alpha=0.05):
    """
    Conservative cluster bootstrap. For Ponte, the five M2 row groups are
    resampled as clusters; all columns within a selected row travel together.
    """
    work = df[[cluster_col, value_col]].dropna()
    clusters = list(work[cluster_col].unique())
    if not clusters:
        return (np.nan, np.nan)
    groups = {c: work.loc[work[cluster_col] == c, value_col].to_numpy(float)
              for c in clusters}
    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    for b in range(reps):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = np.concatenate([groups[c] for c in chosen])
        vals[b] = stat(sample)
    return tuple(np.quantile(vals, [alpha/2, 1-alpha/2]))


def fmt_ci(lo, hi, digits=4):
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "NA"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def holm_adjust(pvals):
    """Holm step-down family-wise error correction without extra dependencies."""
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    good_idx = np.where(np.isfinite(p))[0]
    if len(good_idx) == 0:
        return out
    vals = p[good_idx]
    order = np.argsort(vals)
    m = len(vals)
    adj_sorted = np.empty(m, float)
    running = 0.0
    for j, pos in enumerate(order):
        raw = (m - j) * vals[pos]
        running = max(running, raw)
        adj_sorted[j] = min(1.0, running)
    for j, pos in enumerate(order):
        out[good_idx[pos]] = adj_sorted[j]
    return out


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sha256_bytes(b: bytes):
    return hashlib.sha256(b).hexdigest()


def write_tex_table(df: pd.DataFrame, path: Path, caption: str, label: str,
                    float_format="%.4f"):
    latex = df.to_latex(index=False, escape=True, float_format=float_format,
                        caption=caption, label=label)
    path.write_text(latex, encoding="utf-8")


# ----------------------------- Ponte -----------------------------------

def parse_ponte(results_root: Path):
    path = results_root / "ponte" / "publication" / "ponte_results.json"
    d = load_json(path)
    rows = []
    failed = []
    for cell in d:
        if cell.get("status") != "ok":
            failed.append((cell.get("name"), cell.get("error")))
            continue
        name = cell["name"]
        for r in cell.get("results", []):
            svd = r["svd"]
            rasd = r["rasd"]
            sd = svd.get("diagnostics", {})
            rd = rasd.get("diagnostics", {})
            s_err = safe_float(svd.get("mean_relerr"))
            r_err = safe_float(rasd.get("mean_relerr"))
            change = 100.0 * (s_err - r_err) / s_err if s_err > 0 else np.nan
            rows.append({
                "record": name,
                "cluster": cell_cluster(name),
                "sensor_count": int(r["sensor_count"]),
                "n_channels": int(r.get("n_channels", np.nan)),
                "sensor_fraction": safe_float(r.get("sensor_count", 0)) / max(safe_float(r.get("n_channels", np.nan)), 1.0),
                "rank": int(r["rank"]),
                "svd_error": s_err,
                "rasd_error": r_err,
                "abs_diff_svd_minus_rasd": s_err - r_err,
                "relative_change_percent": change,
                "svd_projection_error": safe_float(svd.get("projection_error")),
                "rasd_projection_error": safe_float(rasd.get("projection_error")),
                "svd_sigma_min": safe_float(sd.get("sigma_min")),
                "rasd_sigma_min": safe_float(rd.get("sigma_min")),
                "svd_amplification": safe_float(pick(sd, "exact_amplification", "amplification")),
                "rasd_amplification": safe_float(pick(rd, "exact_amplification", "amplification")),
                "svd_bias": safe_float(sd.get("ridge_bias_factor")),
                "rasd_bias": safe_float(rd.get("ridge_bias_factor")),
                "svd_rank_G": safe_float(sd.get("rank_G")),
                "rasd_rank_G": safe_float(rd.get("rank_G")),
            })
    return pd.DataFrame(rows), failed


def ponte_budget_summary(df: pd.DataFrame):
    rows = []
    pvals = []
    for m, g in df.groupby("sensor_count", sort=True):
        x = g["abs_diff_svd_minus_rasd"].to_numpy(float)
        rc = g["relative_change_percent"].to_numpy(float)
        wins = int((rc > 0.05).sum())
        losses = int((rc < -0.05).sum())
        ties = int(len(rc) - wins - losses)

        nz = rc[np.abs(rc) > 0.05]
        sign_p = binomtest(int((nz > 0).sum()), n=len(nz), p=0.5).pvalue if len(nz) else np.nan
        try:
            w_p = wilcoxon(x, zero_method="wilcox", alternative="two-sided",
                           method="auto").pvalue
        except Exception:
            w_p = np.nan

        cilo, cihi = cluster_bootstrap_ci(g, "abs_diff_svd_minus_rasd")
        rlo, rhi = cluster_bootstrap_ci(g, "relative_change_percent", stat=np.median)

        rows.append({
            "sensor_count": int(m),
            "records": len(g),
            "median_sensor_fraction_percent": 100*np.median(g["sensor_fraction"]),
            "svd_median_error_percent": 100*np.median(g["svd_error"]),
            "rasd_median_error_percent": 100*np.median(g["rasd_error"]),
            "median_relative_change_percent": np.median(rc),
            "median_change_cluster_bootstrap_lo": rlo,
            "median_change_cluster_bootstrap_hi": rhi,
            "mean_absolute_error_difference": np.mean(x),
            "mean_diff_cluster_bootstrap_lo": cilo,
            "mean_diff_cluster_bootstrap_hi": cihi,
            "wins": wins, "ties": ties, "losses": losses,
            "wilcoxon_p": w_p, "sign_test_p": sign_p,
            "rank_deficient_svd": int((g["svd_rank_G"] < g["rank"]).sum()),
            "rank_deficient_rasd": int((g["rasd_rank_G"] < g["rank"]).sum()),
        })
        pvals.append(w_p)

    out = pd.DataFrame(rows)
    if len(out):
        out["wilcoxon_p_holm"] = holm_adjust(out["wilcoxon_p"].to_numpy(float))
    return out


def ponte_overall_summary(df: pd.DataFrame):
    # Avoid pretending 75 rows are independent: average within each record first.
    per_record = df.groupby(["record", "cluster"], as_index=False).agg(
        mean_relative_change_percent=("relative_change_percent", "mean"),
        median_relative_change_percent=("relative_change_percent", "median"),
        mean_abs_diff=("abs_diff_svd_minus_rasd", "mean"),
        configs=("sensor_count", "size")
    )
    x = per_record["mean_abs_diff"].to_numpy(float)
    try:
        wp = wilcoxon(x, zero_method="wilcox", alternative="two-sided", method="auto").pvalue
    except Exception:
        wp = np.nan
    ci = cluster_bootstrap_ci(per_record, "mean_abs_diff")
    rci = cluster_bootstrap_ci(per_record, "mean_relative_change_percent")
    return per_record, {
        "records": int(len(per_record)),
        "row_clusters": int(per_record["cluster"].nunique()),
        "mean_record_level_relative_change_percent": float(per_record["mean_relative_change_percent"].mean()),
        "median_record_level_relative_change_percent": float(per_record["mean_relative_change_percent"].median()),
        "record_level_change_cluster_bootstrap_95ci": list(rci),
        "mean_record_level_abs_diff": float(per_record["mean_abs_diff"].mean()),
        "mean_record_level_abs_diff_cluster_bootstrap_95ci": list(ci),
        "record_level_wilcoxon_p": safe_float(wp),
        "wins_records": int((per_record["mean_relative_change_percent"] > 0.05).sum()),
        "losses_records": int((per_record["mean_relative_change_percent"] < -0.05).sum()),
        "ties_records": int((per_record["mean_relative_change_percent"].abs() <= 0.05).sum()),
    }


def within_record_correlations(df: pd.DataFrame, method: str):
    """
    Descriptive only. Compute correlations within records across sensor budgets,
    then summarize the 15 record-level correlations instead of pooling 75
    non-independent points.
    """
    cols = {
        "amplification": f"{method}_amplification",
        "sigma_min": f"{method}_sigma_min",
        "bias": f"{method}_bias",
        "projection_error": f"{method}_projection_error",
    }
    rows = []
    for rec, g in df.groupby("record"):
        for metric, c in cols.items():
            z = g[[c, f"{method}_error"]].dropna()
            if len(z) >= 4 and z[c].nunique() > 1 and z[f"{method}_error"].nunique() > 1:
                rho, p = spearmanr(z[c], z[f"{method}_error"])
                rows.append({"record": rec, "metric": metric,
                             "rho": rho, "p": p, "method": method})
    return pd.DataFrame(rows)


def sensor_efficiency(df: pd.DataFrame, thresholds=(0.01, 0.02, 0.05, 0.10)):
    rows = []
    for rec, g in df.groupby("record"):
        row = {"record": rec, "n_channels": int(g["n_channels"].iloc[0])}
        for method in ["svd", "rasd"]:
            for th in thresholds:
                ok = g.loc[g[f"{method}_error"] <= th, "sensor_count"]
                row[f"{method}_min_sensors_for_{int(100*th)}pct"] = int(ok.min()) if len(ok) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------- GridEye ---------------------------------

def parse_grideye(results_root: Path):
    path = results_root / "grideye" / "publication" / "grideye_results.json"
    d = load_json(path)

    main_rows = []
    for x in d.get("results", []):
        svd = x["svd"]; rasd = x["rasd"]
        sd = svd.get("diagnostics", {}); rd = rasd.get("diagnostics", {})
        main_rows.append({
            "site_count": int(x["selected_site_count"]),
            "sites": ";".join(x["selected_sites"]),
            "rank": int(x["rank"]),
            "svd_val_error": safe_float(svd.get("validation_relerr")),
            "svd_test_error": safe_float(svd.get("mean_relerr")),
            "rasd_val_error": safe_float(rasd.get("validation_relerr")),
            "rasd_test_error": safe_float(rasd.get("mean_relerr")),
            "relative_change_percent": safe_float(x.get("paired", {}).get("relative_change_percent")),
            "svd_sigma_min": safe_float(sd.get("sigma_min")),
            "svd_amplification": safe_float(pick(sd, "exact_amplification", "amplification")),
            "svd_bias": safe_float(sd.get("ridge_bias_factor")),
            "rasd_sigma_min": safe_float(rd.get("sigma_min")),
            "rasd_amplification": safe_float(pick(rd, "exact_amplification", "amplification")),
            "rasd_bias": safe_float(rd.get("ridge_bias_factor")),
        })

    nested_rows = []
    for x in d.get("nested_fixed_rank_sensor_path", []):
        dg = x.get("diagnostics", {})
        nested_rows.append({
            "site_count": int(x["site_count"]),
            "sites": ";".join(x["sites"]),
            "validation_error": safe_float(x.get("validation_relerr")),
            "test_error": safe_float(x.get("test_mean_relerr")),
            "sigma_min": safe_float(dg.get("sigma_min")),
            "amplification": safe_float(pick(dg, "exact_amplification", "amplification")),
            "bias": safe_float(dg.get("ridge_bias_factor")),
            "rank_G": safe_float(dg.get("rank_G")),
            "r": safe_float(dg.get("r")),
        })

    event_rows = []
    for x in d.get("results", []):
        k = int(x["selected_site_count"])
        for name, v in x.get("event_ood", {}).items():
            event_rows.append({
                "site_count": k,
                "sites": ";".join(x["selected_sites"]),
                "event": name,
                "svd_error": safe_float(v.get("svd_mean_relerr")),
                "rasd_error": safe_float(v.get("rasd_mean_relerr")),
                "relative_change_percent": safe_float(v.get("paired", {}).get("relative_change_percent")),
            })

    return d, pd.DataFrame(main_rows), pd.DataFrame(nested_rows), pd.DataFrame(event_rows)


def event_hashes(data_root: Path):
    base = data_root / "raw" / "grideye_real_pmu_2025"
    rows = []
    for zpath in sorted(base.glob("*events.zip")):
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                if name.startswith("__MACOSX/") or Path(name).name.startswith("._"):
                    continue
                b = z.read(name)
                rows.append({
                    "archive": zpath.name,
                    "member": name,
                    "event_key": f"{zpath.name}::{Path(name).name}",
                    "bytes": len(b),
                    "sha256": sha256_bytes(b),
                })
    return pd.DataFrame(rows)


def grid_nested_summary(nested: pd.DataFrame):
    if nested.empty:
        return {}
    n = nested.sort_values("site_count").copy()
    def rho(a,b):
        z=n[[a,b]].dropna()
        if len(z)<3: return (np.nan,np.nan)
        r,p=spearmanr(z[a],z[b])
        return (safe_float(r),safe_float(p))
    r_k_smin = rho("site_count","sigma_min")
    r_k_amp = rho("site_count","amplification")
    r_k_test = rho("site_count","test_error")
    first=n.iloc[0]; last=n.iloc[-1]
    return {
        "site_counts": n["site_count"].astype(int).tolist(),
        "sigma_min_monotone_nondecreasing": bool(np.all(np.diff(n["sigma_min"]) >= -1e-12)),
        "amplification_monotone_nonincreasing": bool(np.all(np.diff(n["amplification"]) <= 1e-12)),
        "test_error_monotone_nondecreasing": bool(np.all(np.diff(n["test_error"]) >= -1e-12)),
        "first_sigma_min": safe_float(first["sigma_min"]),
        "last_sigma_min": safe_float(last["sigma_min"]),
        "first_amplification": safe_float(first["amplification"]),
        "last_amplification": safe_float(last["amplification"]),
        "first_test_error": safe_float(first["test_error"]),
        "last_test_error": safe_float(last["test_error"]),
        "test_error_fold_change_last_vs_first": safe_float(last["test_error"]/first["test_error"]) if first["test_error"]>0 else np.nan,
        "spearman_k_sigma_min": {"rho":r_k_smin[0],"p":r_k_smin[1]},
        "spearman_k_amplification": {"rho":r_k_amp[0],"p":r_k_amp[1]},
        "spearman_k_test_error": {"rho":r_k_test[0],"p":r_k_test[1]},
    }


# ----------------------------- figures ---------------------------------

def make_figures(outdir: Path, ponte: pd.DataFrame, budget: pd.DataFrame,
                 gmain: pd.DataFrame, gnested: pd.DataFrame):
    # 1: Ponte median error vs sensor count
    if not ponte.empty:
        agg = ponte.groupby("sensor_count").agg(
            svd=("svd_error","median"),
            rasd=("rasd_error","median")
        ).reset_index()
        fig, ax = plt.subplots(figsize=(6.2,4.2))
        ax.plot(agg["sensor_count"], 100*agg["svd"], marker="o", label="SVD/POD")
        ax.plot(agg["sensor_count"], 100*agg["rasd"], marker="s", label="RASD")
        ax.set_xlabel("Number of sensors")
        ax.set_ylabel("Median relative reconstruction error (%)")
        ax.legend()
        ax.grid(True, alpha=.25)
        fig.tight_layout()
        fig.savefig(outdir/"fig_ponte_error_vs_sensors.pdf")
        fig.savefig(outdir/"fig_ponte_error_vs_sensors.png", dpi=220)
        plt.close(fig)

        # 2: RASD relative change by budget
        fig, ax = plt.subplots(figsize=(6.2,4.2))
        groups=[ponte.loc[ponte.sensor_count==m,"relative_change_percent"].values
                for m in sorted(ponte.sensor_count.unique())]
        ax.boxplot(groups, tick_labels=[str(m) for m in sorted(ponte.sensor_count.unique())],
                   showfliers=True)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Number of sensors")
        ax.set_ylabel("RASD relative error reduction vs SVD (%)")
        ax.grid(True, axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(outdir/"fig_ponte_rasd_change.pdf")
        fig.savefig(outdir/"fig_ponte_rasd_change.png", dpi=220)
        plt.close(fig)

    # 3: GridEye fixed-rank nested: test error
    if not gnested.empty:
        n=gnested.sort_values("site_count")
        fig, ax = plt.subplots(figsize=(6.2,4.2))
        ax.plot(n["site_count"],100*n["test_error"],marker="o")
        ax.set_xlabel("Nested PMU site count")
        ax.set_ylabel("Evening deployment error (%)")
        ax.grid(True,alpha=.25)
        fig.tight_layout()
        fig.savefig(outdir/"fig_grideye_nested_test_error.pdf")
        fig.savefig(outdir/"fig_grideye_nested_test_error.png",dpi=220)
        plt.close(fig)

        # 4: GridEye sensing geometry
        fig, ax = plt.subplots(figsize=(6.2,4.2))
        ax.plot(n["site_count"],n["sigma_min"],marker="o",label=r"$\sigma_{\min}(SB)$")
        ax.plot(n["site_count"],n["amplification"],marker="s",label="Exact amplification")
        ax.set_xlabel("Nested PMU site count")
        ax.set_ylabel("Diagnostic value")
        ax.legend()
        ax.grid(True,alpha=.25)
        fig.tight_layout()
        fig.savefig(outdir/"fig_grideye_nested_geometry.pdf")
        fig.savefig(outdir/"fig_grideye_nested_geometry.png",dpi=220)
        plt.close(fig)


# ----------------------------- report ----------------------------------

def build_report(outdir: Path, ponte: pd.DataFrame, failed,
                 budget: pd.DataFrame, per_record: pd.DataFrame, overall: dict,
                 corr: pd.DataFrame, efficiency: pd.DataFrame,
                 gmain: pd.DataFrame, gnested: pd.DataFrame, gevents: pd.DataFrame,
                 hashes: pd.DataFrame, nested_summary: dict):

    lines=[]
    lines.append("# Publication Evidence Audit")
    lines.append("")
    lines.append("This report is generated directly from the publication JSON files. "
                 "It is deliberately conservative: Ponte significance is evaluated at "
                 "the record level, with an additional bootstrap clustered by M2 row, "
                 "rather than treating every sensor-budget configuration as independent.")
    lines.append("")

    lines.append("## 1. Ponte Moesa: real sparse-to-dense strain recovery")
    lines.append("")
    if failed:
        lines.append(f"Failed records: {failed}")
    else:
        lines.append(f"All {ponte['record'].nunique()} non-empty records completed successfully.")
    lines.append(f"The publication analysis contains {len(ponte)} record-budget configurations "
                 f"across sensor budgets {sorted(ponte.sensor_count.unique().tolist())}.")
    lines.append("")
    lines.append("### Budget-wise paired evidence")
    lines.append("")
    for _,r in budget.iterrows():
        lines.append(
            f"- **{int(r.sensor_count)} sensors:** median SVD error "
            f"{r.svd_median_error_percent:.3f}%, median RASD error "
            f"{r.rasd_median_error_percent:.3f}%; median relative RASD change "
            f"{r.median_relative_change_percent:.2f}% "
            f"(cluster-bootstrap 95% CI "
            f"{fmt_ci(r.median_change_cluster_bootstrap_lo, r.median_change_cluster_bootstrap_hi,2)}); "
            f"{int(r.wins)} wins / {int(r.ties)} ties / {int(r.losses)} losses; "
            f"Wilcoxon p={r.wilcoxon_p:.4g}, Holm-adjusted p={r.wilcoxon_p_holm:.4g}."
        )
    lines.append("")
    lines.append("### Across budgets without pseudo-replication")
    lines.append("")
    lines.append(
        f"Each bridge record was first collapsed across sensor budgets. Across "
        f"{overall['records']} records ({overall['row_clusters']} row clusters), "
        f"the mean record-level relative RASD change was "
        f"{overall['mean_record_level_relative_change_percent']:.2f}% and the median was "
        f"{overall['median_record_level_relative_change_percent']:.2f}%. "
        f"The row-cluster bootstrap 95% CI for the mean record-level change is "
        f"{fmt_ci(*overall['record_level_change_cluster_bootstrap_95ci'],2)}. "
        f"Record-level Wilcoxon p={overall['record_level_wilcoxon_p']:.4g}."
    )
    lines.append("")
    rankdef_svd=int((ponte.svd_rank_G < ponte["rank"]).sum())
    rankdef_rasd=int((ponte.rasd_rank_G < ponte["rank"]).sum())
    lines.append(
        f"Rank-deficient sensed bases occurred in {rankdef_svd} SVD configurations "
        f"and {rankdef_rasd} RASD configurations. In these cases the correct smallest "
        f"singular value is zero and the ridge-bias factor should be one; this directly "
        f"distinguishes regularized solvability from latent identifiability."
    )
    lines.append("")

    lines.append("## 2. GridEye: cross-regime deployment")
    lines.append("")
    if not gmain.empty:
        for _,r in gmain.sort_values("site_count").iterrows():
            lines.append(
                f"- **{int(r.site_count)} site(s)** ({r.sites}), rank {int(r['rank'])}: "
                f"SVD noon/evening error {100*r.svd_val_error:.3f}% / "
                f"{100*r.svd_test_error:.3f}%; RASD "
                f"{100*r.rasd_val_error:.3f}% / {100*r.rasd_test_error:.3f}%; "
                f"RASD relative change {r.relative_change_percent:.2f}%."
            )
    lines.append("")
    lines.append("### Fixed-rank nested sensor path")
    lines.append("")
    if nested_summary:
        lines.append(
            f"From {nested_summary['site_counts'][0]} to {nested_summary['site_counts'][-1]} sites, "
            f"$\\sigma_{{\\min}}$ changed from {nested_summary['first_sigma_min']:.4f} to "
            f"{nested_summary['last_sigma_min']:.4f}, while exact amplification changed from "
            f"{nested_summary['first_amplification']:.3f} to "
            f"{nested_summary['last_amplification']:.3f}. Evening test error changed from "
            f"{100*nested_summary['first_test_error']:.3f}% to "
            f"{100*nested_summary['last_test_error']:.3f}% "
            f"({nested_summary['test_error_fold_change_last_vs_first']:.2f}x)."
        )
        lines.append(
            f"Monotonic checks: sigma_min nondecreasing = "
            f"{nested_summary['sigma_min_monotone_nondecreasing']}; amplification "
            f"nonincreasing = {nested_summary['amplification_monotone_nonincreasing']}; "
            f"test error nondecreasing = {nested_summary['test_error_monotone_nondecreasing']}."
        )
        lines.append(
            f"Spearman trends across the six nested points: site count vs sigma_min "
            f"rho={nested_summary['spearman_k_sigma_min']['rho']:.3f}; "
            f"site count vs amplification rho={nested_summary['spearman_k_amplification']['rho']:.3f}; "
            f"site count vs test error rho={nested_summary['spearman_k_test_error']['rho']:.3f}. "
            "Because n=6, these trend statistics are descriptive, not stand-alone inferential evidence."
        )
    lines.append("")

    lines.append("## 3. Event-file integrity check")
    lines.append("")
    if hashes.empty:
        lines.append("No event hashes were computed because the raw GridEye event archives were not found.")
    else:
        dup=hashes.groupby("sha256").filter(lambda g: len(g)>1)
        lines.append(f"Hashed {len(hashes)} event CSV files; unique hashes = {hashes.sha256.nunique()}.")
        if len(dup):
            lines.append("Exact byte-identical duplicates were detected and should not be counted as independent events:")
            for h,g in dup.groupby("sha256"):
                lines.append(f"- {h[:16]}: " + "; ".join(g.event_key.tolist()))
        else:
            lines.append("No exact byte-identical event CSV duplicates were found.")
    lines.append("")

    lines.append("## 4. Claims that the data support")
    lines.append("")
    lines.append(
        "1. Real sparse-to-dense recovery can be evaluated without imputing the Ponte benchmark: "
        "all reported Ponte fields use spatial coordinates observed at every snapshot."
    )
    lines.append(
        "2. Recovery-aware basis learning is conditional rather than universally dominant. "
        "The paper should report both improvements and deteriorations."
    )
    lines.append(
        "3. Rank, smallest sensed singular value, exact sensor-to-field amplification, ridge bias, "
        "and deployment error are empirically distinct quantities."
    )
    lines.append(
        "4. The GridEye fixed-rank nested experiment directly tests whether improved nominal sensing "
        "geometry is sufficient for cross-regime deployment. The numerical sequence should be shown "
        "in full rather than reduced to a single cherry-picked comparison."
    )
    lines.append(
        "5. Event results should be described as separate event stress tests only after the hash audit "
        "confirms which files are genuinely distinct."
    )
    lines.append("")
    lines.append("## 5. Claims to avoid")
    lines.append("")
    lines.append(
        "- Do not claim that RASD universally outperforms POD/SVD."
    )
    lines.append(
        "- Do not count the 75 Ponte configurations as 75 independent experiments."
    )
    lines.append(
        "- Do not infer that lower condition number alone guarantees deployment reliability."
    )
    lines.append(
        "- Do not call ERA statistics SCADA or operational state measurements."
    )
    lines.append(
        "- Do not treat multiple event files with identical hashes as independent validation."
    )

    (outdir/"publication_evidence_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def build_combined_tex(outdir: Path, budget: pd.DataFrame, gmain: pd.DataFrame, gnested: pd.DataFrame):
    # Compact tables suitable for manuscript insertion.
    parts=[]

    p=budget.copy()
    if len(p):
        p=p[["sensor_count","svd_median_error_percent","rasd_median_error_percent",
             "median_relative_change_percent","wins","ties","losses","wilcoxon_p_holm"]]
        p.columns=["Sensors","SVD median error (\\%)","RASD median error (\\%)",
                   "Median RASD change (\\%)","Wins","Ties","Losses","Holm $p$"]
        parts.append(p.to_latex(index=False, escape=False, float_format="%.3f",
            caption="Ponte Moesa record-level paired summary across sparse sensor budgets.",
            label="tab:ponte_real_summary"))

    if len(gmain):
        g=gmain.copy()
        for c in ["svd_val_error","svd_test_error","rasd_val_error","rasd_test_error"]:
            g[c]=100*g[c]
        g=g[["site_count","sites","rank","svd_val_error","svd_test_error",
             "rasd_val_error","rasd_test_error","relative_change_percent"]]
        g.columns=["Sites","Selected PMU sites","Rank","SVD val. (\\%)","SVD test (\\%)",
                   "RASD val. (\\%)","RASD test (\\%)","RASD change (\\%)"]
        parts.append(g.to_latex(index=False, escape=False, float_format="%.3f",
            caption="GridEye morning-to-noon-to-evening cross-regime reconstruction.",
            label="tab:grideye_main"))

    if len(gnested):
        n=gnested.copy()
        n["validation_error"]=100*n["validation_error"]
        n["test_error"]=100*n["test_error"]
        n=n[["site_count","sites","validation_error","test_error",
             "sigma_min","amplification","bias"]]
        n.columns=["Sites","Nested site set","Noon error (\\%)","Evening error (\\%)",
                   "$\\sigma_{\\min}(SB)$","Exact amplification","Ridge bias"]
        parts.append(n.to_latex(index=False, escape=False, float_format="%.4f",
            caption="Fixed-rank nested GridEye sensor path. Sensing geometry and deployment error are reported separately.",
            label="tab:grideye_nested"))

    (outdir/"publication_tables.tex").write_text("\n\n".join(parts)+"\n", encoding="utf-8")


# ----------------------------- main ------------------------------------

def main():
    ap=argparse.ArgumentParser(
        description="Build conservative publication evidence from final DCE real-data results.")
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Acquisition data directory; used only for event integrity hashes.")
    ap.add_argument("--out", type=Path, default=Path("publication_evidence"))
    args=ap.parse_args()

    ensure_dir(args.out)

    ponte, failed = parse_ponte(args.results_root)
    if ponte.empty:
        raise RuntimeError("No Ponte publication results found.")
    budget = ponte_budget_summary(ponte)
    per_record, overall = ponte_overall_summary(ponte)
    corr = pd.concat([within_record_correlations(ponte,"svd"),
                      within_record_correlations(ponte,"rasd")],
                     ignore_index=True)
    efficiency = sensor_efficiency(ponte)

    gd, gmain, gnested, gevents = parse_grideye(args.results_root)
    nested_summary = grid_nested_summary(gnested)

    hashes = pd.DataFrame()
    if args.data_root is not None:
        hashes = event_hashes(args.data_root)

    # Link event rows to exact hashes when possible.
    if not gevents.empty and not hashes.empty:
        map_hash={}
        for _,r in hashes.iterrows():
            map_hash[r["event_key"]] = r["sha256"]
            # JSON uses archive::basename in current pipeline.
            map_hash[f"{r['archive']}::{Path(r['member']).name}"] = r["sha256"]
        gevents["sha256"] = gevents["event"].map(map_hash)

    ponte.to_csv(args.out/"ponte_all_configurations.csv",index=False)
    budget.to_csv(args.out/"ponte_budget_summary.csv",index=False)
    per_record.to_csv(args.out/"ponte_record_level_summary.csv",index=False)
    corr.to_csv(args.out/"ponte_within_record_diagnostic_correlations.csv",index=False)
    efficiency.to_csv(args.out/"ponte_sensor_efficiency.csv",index=False)
    gmain.to_csv(args.out/"grideye_main.csv",index=False)
    gnested.to_csv(args.out/"grideye_nested_fixed_rank.csv",index=False)
    gevents.to_csv(args.out/"grideye_event_ood.csv",index=False)
    hashes.to_csv(args.out/"grideye_event_hashes.csv",index=False)

    summary={
        "ponte_overall":overall,
        "grideye_nested":nested_summary,
        "failed_ponte":failed,
    }
    (args.out/"key_statistics.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")

    make_figures(args.out,ponte,budget,gmain,gnested)
    build_report(args.out,ponte,failed,budget,per_record,overall,corr,efficiency,
                 gmain,gnested,gevents,hashes,nested_summary)
    build_combined_tex(args.out,budget,gmain,gnested)

    print("\nPUBLICATION EVIDENCE BUILT")
    print("Output:",args.out.resolve())
    print("\nPonte budget summary:")
    print(budget[["sensor_count","svd_median_error_percent","rasd_median_error_percent",
                  "median_relative_change_percent","wins","ties","losses",
                  "wilcoxon_p_holm"]].to_string(index=False))
    print("\nPonte overall record-level summary:")
    for k,v in overall.items():
        print(f"  {k}: {v}")
    print("\nGridEye nested fixed-rank summary:")
    for k,v in nested_summary.items():
        print(f"  {k}: {v}")
    if not hashes.empty:
        print("\nGridEye event integrity:")
        print(hashes[["event_key","bytes","sha256"]].to_string(index=False))
        print("Unique event hashes:",hashes["sha256"].nunique(),"/",len(hashes))
    print("\nFiles written:")
    for p in sorted(args.out.iterdir()):
        print(" ",p.name)


if __name__ == "__main__":
    main()
