#!/usr/bin/env python3
"""Known-support validation for uniform-prior Gaussian support inference."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from support_inference import exact_uniform_support_inference
from symbolic import standardize_train_validation


def main() -> None:
    output = HERE / "synthetic_test"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    seed = 20260818
    realizations = 200
    n_samples = 80
    noise_levels = (0.1, 0.5, 1.0)
    correlations = (0.0, 0.5, 0.9, 0.99)
    rows: list[dict[str, object]] = []

    for noise in noise_levels:
        for correlation in correlations:
            for realization in range(realizations):
                condition_seed = (
                    seed + realization + int(noise * 10_000) + int(correlation * 1_000_000)
                )
                rng = np.random.default_rng(condition_seed)
                x1 = rng.normal(size=n_samples)
                x2 = rng.normal(size=n_samples)
                independent = rng.normal(size=n_samples)
                decoy = correlation * x1 + np.sqrt(1.0 - correlation**2) * independent
                nuisance = rng.normal(size=n_samples)
                X = np.column_stack((x1, x2, decoy, nuisance))
                y = 1.5 * x1 - 1.0 * x2 + rng.normal(scale=noise, size=n_samples)
                X_scaled, _, _, _, _ = standardize_train_validation(X, X)
                posterior = exact_uniform_support_inference(
                    X_scaled,
                    y,
                    X_scaled[:1],
                    dimension=2,
                    sigma2=noise**2,
                    tau2=float(np.var(y, ddof=0) / 2),
                    batch_size=32,
                    top_n=6,
                    probability_threshold=0.0,
                )
                support_rows = posterior.top_supports
                true_rank = next(
                    int(row["rank"])
                    for row in support_rows
                    if tuple(row["columns"]) == (0, 1)
                )
                true_probability = next(
                    float(row["posterior_probability"])
                    for row in support_rows
                    if tuple(row["columns"]) == (0, 1)
                )
                rows.append(
                    {
                        "noise_sd": noise,
                        "true_decoy_correlation": correlation,
                        "realization": realization + 1,
                        "seed": condition_seed,
                        "pip_true_x1": float(posterior.inclusion_probabilities[0]),
                        "pip_true_x2": float(posterior.inclusion_probabilities[1]),
                        "pip_decoy": float(posterior.inclusion_probabilities[2]),
                        "pip_nuisance": float(posterior.inclusion_probabilities[3]),
                        "true_support_probability": true_probability,
                        "map_recovers_true_support": posterior.map_columns == (0, 1),
                        "true_support_in_95_model_set": true_rank <= posterior.supports_to_95,
                        "supports_to_95_probability": posterior.supports_to_95,
                    }
                )

    with (output / "synthetic_realizations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries: list[dict[str, object]] = []
    for noise in noise_levels:
        for correlation in correlations:
            selected = [
                row for row in rows
                if row["noise_sd"] == noise and row["true_decoy_correlation"] == correlation
            ]
            summaries.append(
                {
                    "noise_sd": noise,
                    "true_decoy_correlation": correlation,
                    "realizations": len(selected),
                    "mean_pip_true_x1": float(np.mean([row["pip_true_x1"] for row in selected])),
                    "mean_pip_true_x2": float(np.mean([row["pip_true_x2"] for row in selected])),
                    "mean_pip_decoy": float(np.mean([row["pip_decoy"] for row in selected])),
                    "map_support_recovery_frequency": float(np.mean([row["map_recovers_true_support"] for row in selected])),
                    "true_support_95_model_set_frequency": float(np.mean([row["true_support_in_95_model_set"] for row in selected])),
                    "mean_supports_to_95_probability": float(np.mean([row["supports_to_95_probability"] for row in selected])),
                }
            )
    with (output / "synthetic_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / "synthetic_summary.json").write_text(
        json.dumps(
            {
                "model": "y = 1.5*x1 - 1.0*x2 + Gaussian noise",
                "candidate_features": ["x1", "x2", "correlated decoy of x1", "independent nuisance"],
                "true_support": ["x1", "x2"],
                "base_seed": seed,
                "realizations_per_condition": realizations,
                "conditions": summaries,
            },
            indent=2,
        ) + "\n"
    )

    import matplotlib.pyplot as plt

    colors = {0.1: "#0072B2", 0.5: "#D55E00", 1.0: "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.25), sharex=True)
    for noise in noise_levels:
        selected = [row for row in summaries if row["noise_sd"] == noise]
        x = [row["true_decoy_correlation"] for row in selected]
        axes[0].plot(x, [row["mean_pip_true_x1"] for row in selected], "o-", color=colors[noise], label=fr"$\sigma={noise}$")
        axes[0].plot(x, [row["mean_pip_decoy"] for row in selected], "s--", color=colors[noise], alpha=0.8)
        axes[1].plot(x, [row["map_support_recovery_frequency"] for row in selected], "o-", color=colors[noise])
        axes[2].plot(x, [row["mean_supports_to_95_probability"] for row in selected], "o-", color=colors[noise])
    axes[0].set_ylabel("Mean PIP")
    axes[0].set_title("true $x_1$ (circles) / decoy (squares)")
    axes[1].set_ylabel("MAP recovery frequency")
    axes[1].set_title("exact true-support recovery")
    axes[2].set_ylabel("Supports in 95% model set")
    axes[2].set_title("posterior model-set size")
    for axis in axes:
        axis.set_xlabel(r"corr($x_1$, decoy)")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "synthetic_support_recovery.pdf", bbox_inches="tight")
    fig.savefig(output / "synthetic_support_recovery.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

