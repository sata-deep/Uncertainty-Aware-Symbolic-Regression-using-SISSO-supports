#!/usr/bin/env python3
"""Scaled full-data posterior and requested feature/support diagnostics."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path[:0] = [str(HERE), str(PROJECT)]

from bayes_so2.io import load_sisso_directory
from support_inference import exact_uniform_support_inference
from symbolic import standardize_train_validation


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def write_full_posterior(dimension: int) -> dict[str, object]:
    source_dir = PROJECT / f"SISSO-{dimension}D"
    output = HERE / "full_data_posterior" / f"d{dimension}"
    output.mkdir(parents=True, exist_ok=True)
    subspace, _ = load_sisso_directory(source_dir)
    if subspace.deterministic_model is None or subspace.deterministic_model.rmse is None:
        raise RuntimeError("deterministic model/RMSE unavailable")
    X_scaled, _, means, scales, keep = standardize_train_validation(subspace.X, subspace.X)
    posterior = exact_uniform_support_inference(
        X_scaled,
        subspace.y,
        X_scaled,
        dimension,
        sigma2=float(subspace.deterministic_model.rmse**2),
        tau2=float(np.var(subspace.y, ddof=0) / dimension),
        batch_size=50_000,
        top_n=100,
    )
    kept_indices = np.flatnonzero(keep)
    kept_names = [name for name, include in zip(subspace.feature_names, keep) if include]
    kept_ids = subspace.feature_ids[keep]
    top_rows: list[dict[str, object]] = []
    for row in posterior.top_supports:
        columns = [int(value) for value in row["columns"]]
        top_rows.append(
            {
                "rank": row["rank"],
                "posterior_probability": row["posterior_probability"],
                "cumulative_probability": row["cumulative_probability"],
                "feature_ids": " ".join(str(int(kept_ids[index])) for index in columns),
                "feature_expressions": " | ".join(kept_names[index] for index in columns),
                "log_marginal_likelihood": row["log_marginal_likelihood"],
            }
        )
    with (output / "top_100_supports.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(top_rows[0]))
        writer.writeheader()
        writer.writerows(top_rows)
    with (output / "feature_pips.csv").open("w", newline="") as handle:
        fieldnames = ["feature_id", "expression", "pip", "mean", "std"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scaled_index in np.argsort(posterior.inclusion_probabilities)[::-1]:
            original = kept_indices[scaled_index]
            writer.writerow(
                {
                    "feature_id": int(subspace.feature_ids[original]),
                    "expression": subspace.feature_names[original],
                    "pip": float(posterior.inclusion_probabilities[scaled_index]),
                    "mean": float(means[original]),
                    "std": float(scales[original]),
                }
            )
    with (output / "predictions.csv").open("w", newline="") as handle:
        fields = ["sample_name", "observed", "pred_map", "pred_bma", "bma_sd", "bma_lower95", "bma_upper95"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, name in enumerate(subspace.sample_names):
            writer.writerow(dict(sample_name=name, observed=float(subspace.y[index]),
                                 pred_map=float(posterior.map_predictions[index]),
                                 pred_bma=float(posterior.bma_predictions[index]),
                                 bma_sd=float(np.sqrt(posterior.bma_variance[index])),
                                 bma_lower95=float(posterior.bma_intervals[index, 0]),
                                 bma_upper95=float(posterior.bma_intervals[index, 1])))
    map_ids = [int(kept_ids[index]) for index in posterior.map_columns]
    summary = {
        "dimension": dimension,
        "K": int(np.sum(keep)),
        "support_count": posterior.support_count,
        "sigma2": float(subspace.deterministic_model.rmse**2),
        "tau2": float(np.var(subspace.y, ddof=0) / dimension),
        "deterministic_support": list(subspace.deterministic_model.feature_ids),
        "map_support": map_ids,
        "map_probability": posterior.map_probability,
        "supports_to_95_probability": posterior.supports_to_95,
        "leading_posterior_odds_bayes_factor": posterior.leading_posterior_odds,
        "bma_training_rmse": rmse(subspace.y, posterior.bma_predictions),
        "bma_accumulated_probability": posterior.accumulated_probability,
        "top_10_supports": top_rows[:10],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "top_10_supports.tex").open("w") as handle:
        handle.write("\\begin{tabular}{rllr}\n\\toprule\nRank & Support & $p(\\gamma\\mid\\mathcal D)$ & Cumulative \\\\\n\\midrule\n")
        for row in top_rows[:10]:
            support = row["feature_ids"].replace(" ", ",")
            handle.write(f"{row['rank']} & $({support})$ & {row['posterior_probability']:.4g} & {row['cumulative_probability']:.4g} \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")
    return summary


def feature_201_203_diagnostic() -> dict[str, object]:
    output = HERE / "feature_201_203"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
    subspace, _ = load_sisso_directory(PROJECT / "SISSO-2D")
    phi201 = subspace.X[:, 200]
    phi203 = subspace.X[:, 202]
    pearson = pearsonr(phi201, phi203)
    spearman = spearmanr(phi201, phi203)
    rank201 = np.argsort(np.argsort(phi201)) + 1
    rank203 = np.argsort(np.argsort(phi203)) + 1
    rank_displacement = np.abs(rank201 - rank203)
    rows = [
        {
            "sample_name": name,
            "phi_201": float(phi201[index]),
            "phi_203": float(phi203[index]),
            "rank_201": int(rank201[index]),
            "rank_203": int(rank203[index]),
            "absolute_rank_difference": int(rank_displacement[index]),
        }
        for index, name in enumerate(subspace.sample_names)
    ]
    with (output / "feature_201_203_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "mean_absolute_rank_difference": float(np.mean(rank_displacement)),
        "median_absolute_rank_difference": float(np.median(rank_displacement)),
        "maximum_absolute_rank_difference": int(np.max(rank_displacement)),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].scatter(phi201, phi203, s=25, alpha=0.75, edgecolor="none", color="#0072B2")
    axes[0].set_xlabel(r"$\phi_{201}$")
    axes[0].set_ylabel(r"$\phi_{203}$")
    axes[0].text(0.04, 0.96, f"Pearson $r={summary['pearson_r']:.3f}$\nSpearman $\\rho={summary['spearman_rho']:.3f}$", transform=axes[0].transAxes, va="top")
    axes[1].scatter(rank201, rank203, s=25, alpha=0.75, edgecolor="none", color="#D55E00")
    axes[1].plot([1, len(rank201)], [1, len(rank201)], "k--", lw=1)
    axes[1].set_xlabel(r"rank by $\phi_{201}$")
    axes[1].set_ylabel(r"rank by $\phi_{203}$")
    axes[1].set_title("compound ordering")
    fig.tight_layout()
    fig.savefig(output / "feature_201_vs_203.pdf", bbox_inches="tight")
    fig.savefig(output / "feature_201_vs_203.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


