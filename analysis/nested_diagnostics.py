#!/usr/bin/env python3
"""Residual and X-family diagnostics from completed nested predictions."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    source = HERE / "predictions_all.csv"
    if not source.exists():
        raise FileNotFoundError("run nested CV first: predictions_all.csv is absent")
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if int(row["repetition"]) == 1 and int(row["seed"]) == 42]
    summaries: list[dict[str, object]] = []
    for dimension in (2, 3):
        dimension_rows = [row for row in selected if int(row["dimension"]) == dimension]
        if len(dimension_rows) != 88:
            continue
        for family in ("overall", "Co", "Fe", "Ni"):
            family_rows = dimension_rows if family == "overall" else [row for row in dimension_rows if row["x_family"] == family]
            residuals = np.array([float(row["observed"]) - float(row["pred_bma"]) for row in family_rows])
            summaries.append(
                {
                    "dimension": dimension,
                    "family": family,
                    "n": len(family_rows),
                    "residual_mean": float(np.mean(residuals)),
                    "residual_sd": float(np.std(residuals, ddof=1)),
                    "rmse_bma": float(np.sqrt(np.mean(residuals**2))),
                    "coverage_bma_95": float(np.mean([row["bma_covered95"].lower() == "true" for row in family_rows])),
                    "mean_interval_width": float(np.mean([float(row["bma_upper95"]) - float(row["bma_lower95"]) for row in family_rows])),
                }
            )
    output = HERE / "residual_subgroup_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    if not summaries:
        raise RuntimeError("both dimensions require 88 completed seed-42 predictions")
    with (output / "residual_and_coverage_by_family.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
    import matplotlib.pyplot as plt

    colors = {"Co": "#0072B2", "Fe": "#D55E00", "Ni": "#009E73"}
    markers = {"Co": "o", "Fe": "s", "Ni": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for axis, dimension in zip(axes, (2, 3)):
        dimension_rows = [row for row in selected if int(row["dimension"]) == dimension]
        for family in ("Co", "Fe", "Ni"):
            family_rows = [row for row in dimension_rows if row["x_family"] == family]
            fitted = np.array([float(row["pred_bma"]) for row in family_rows])
            residual = np.array([float(row["observed"]) for row in family_rows]) - fitted
            axis.scatter(fitted, residual, s=27, alpha=0.75, marker=markers[family], color=colors[family], label=family)
        axis.axhline(0.0, color="black", ls="--", lw=1)
        axis.set_xlabel("nested-CV BMA fitted value")
        axis.set_title(fr"$d={dimension}$")
    axes[0].set_ylabel("validation residual")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "nested_residuals_by_family.pdf", bbox_inches="tight")
    fig.savefig(output / "nested_residuals_by_family.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

