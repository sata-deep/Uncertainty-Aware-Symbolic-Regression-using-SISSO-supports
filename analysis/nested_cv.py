#!/usr/bin/env python3
"""Leakage-free nested SISSO/Bayesian-support cross-validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bayes_so2.io import load_sisso_directory, parse_train_dat
from support_inference import exact_uniform_support_inference
from symbolic import (
    canonical_expression_ids,
    evaluate_expressions,
    standardize_train_validation,
)


DEFAULT_EXECUTABLE = Path(os.environ.get("SISSO_EXECUTABLE", "SISSO"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fold-local residual-driven SISSO and uniform-prior Bayesian support inference"
    )
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--sisso-template-dir", type=Path)
    parser.add_argument("--sisso-executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="rerun completed fold directories")
    return parser.parse_args()


def json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _replace_control(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^\s*{re.escape(key)}\s*=.*$")
    replacement = f"{key}={value}"
    if not pattern.search(text):
        raise ValueError(f"missing {key} in SISSO.in template")
    return pattern.sub(replacement, text, count=1)


def write_fold_sisso_inputs(
    template_dir: Path,
    fold_sisso_dir: Path,
    train_indices: np.ndarray,
) -> None:
    fold_sisso_dir.mkdir(parents=True, exist_ok=True)
    source_lines = (template_dir / "train.dat").read_text().splitlines()
    header, data_lines = source_lines[0], source_lines[1:]
    selected_lines = [data_lines[int(index)] for index in train_indices]
    (fold_sisso_dir / "train.dat").write_text(
        "\n".join([header, *selected_lines]) + "\n"
    )
    control = (template_dir / "SISSO.in").read_text()
    control = _replace_control(control, "nsample", str(len(train_indices)))
    control = _replace_control(control, "restart", "0")
    (fold_sisso_dir / "SISSO.in").write_text(control)


def run_sisso(executable: Path, fold_sisso_dir: Path) -> float:
    if not executable.is_file():
        raise FileNotFoundError(f"SISSO executable not found: {executable}")
    environment = os.environ.copy()
    environment["I_MPI_FABRICS"] = "shm"
    environment.setdefault("OMP_NUM_THREADS", "1")
    started = time.perf_counter()
    with (fold_sisso_dir / "SISSO.out").open("w") as output:
        subprocess.run(
            [str(executable.resolve())],
            cwd=fold_sisso_dir,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.perf_counter() - started


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2)))


def support_payload(
    columns: list[int] | tuple[int, ...],
    feature_ids: np.ndarray,
    expressions: list[str],
    canonical_ids: dict[str, int],
) -> dict[str, object]:
    cols = [int(value) for value in columns]
    names = [expressions[index] for index in cols]
    return {
        "fold_local_ids": [int(feature_ids[index]) for index in cols],
        "canonical_full_data_ids": [canonical_ids.get(name) for name in names],
        "expressions": names,
    }


def same_support(first: dict[str, object], second: dict[str, object]) -> bool:
    return sorted(first["expressions"]) == sorted(second["expressions"])


def write_fold_csvs(
    fold_dir: Path,
    payload: dict[str, Any],
    feature_rows: list[dict[str, object]],
    top_rows: list[dict[str, object]],
) -> None:
    with (fold_dir / "predictions.csv").open("w", newline="") as handle:
        fieldnames = [
            "sample_name", "observed", "pred_deterministic", "pred_map", "pred_bma",
            "bma_sd", "bma_lower95", "bma_upper95", "bma_covered95", "x_family",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload["predictions"])
    with (fold_dir / "feature_scaling_and_pips.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    with (fold_dir / "top_posterior_supports.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(top_rows[0]))
        writer.writeheader()
        writer.writerows(top_rows)


def run_fold(
    *,
    dimension: int,
    repetition: int,
    seed: int,
    fold_number: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    template_dir: Path,
    executable: Path,
    run_dir: Path,
    batch_size: int,
    top_n: int,
) -> dict[str, Any]:
    fold_dir = run_dir / f"fold_{fold_number}"
    sisso_dir = fold_dir / "sisso"
    fold_dir.mkdir(parents=True, exist_ok=True)
    source = parse_train_dat(template_dir / "train.dat")
    y_all = np.asarray(source["y"], dtype=float)
    primary_all = np.asarray(source["primary_features"], dtype=float)
    primary_names = list(source["primary_feature_names"])
    sample_names = list(source["sample_names"])

    write_fold_sisso_inputs(template_dir, sisso_dir, train_indices)
    sisso_seconds = run_sisso(executable, sisso_dir)
    fold_subspace, _ = load_sisso_directory(sisso_dir)
    if fold_subspace.deterministic_model is None:
        raise RuntimeError("fold-local SISSO deterministic model was not parsed")

    validation_raw = evaluate_expressions(
        fold_subspace.feature_names,
        primary_names,
        primary_all[validation_indices],
    )
    reconstructed_train = evaluate_expressions(
        fold_subspace.feature_names,
        primary_names,
        primary_all[train_indices],
    )
    reconstruction_error = float(np.max(np.abs(reconstructed_train - fold_subspace.X)))
    if not np.allclose(reconstructed_train, fold_subspace.X, rtol=2.0e-7, atol=2.0e-8):
        raise RuntimeError(
            f"expression evaluator does not reproduce fold Uspace.dat; max error={reconstruction_error}"
        )

    deterministic = fold_subspace.deterministic_model
    deterministic_columns = np.array(deterministic.feature_ids, dtype=int) - 1
    deterministic_predictions = (
        deterministic.intercept
        + validation_raw[:, deterministic_columns] @ deterministic.coefficients
    )
    deterministic_training_predictions = (
        deterministic.intercept
        + fold_subspace.X[:, deterministic_columns] @ deterministic.coefficients
    )
    deterministic_training_rmse = rmse(fold_subspace.y, deterministic_training_predictions)
    if deterministic.rmse is not None and not math.isclose(
        deterministic_training_rmse, deterministic.rmse, rel_tol=2.0e-6, abs_tol=2.0e-7
    ):
        raise RuntimeError("parsed deterministic SISSO prediction does not reproduce its RMSE")

    X_train_scaled, X_validation_scaled, means, scales, keep = standardize_train_validation(
        fold_subspace.X, validation_raw
    )
    kept_indices = np.flatnonzero(keep)
    inverse_kept = {int(original): new for new, original in enumerate(kept_indices)}
    if any(int(column) not in inverse_kept for column in deterministic_columns):
        raise RuntimeError("fold SISSO selected a feature removed as numerically constant")

    sigma2 = max(deterministic_training_rmse**2, np.finfo(float).eps)
    tau2 = max(float(np.var(fold_subspace.y, ddof=0) / dimension), np.finfo(float).eps)
    bayes_started = time.perf_counter()
    posterior = exact_uniform_support_inference(
        X_train_scaled,
        fold_subspace.y,
        X_validation_scaled,
        dimension,
        sigma2,
        tau2,
        batch_size=batch_size,
        top_n=top_n,
    )
    bayes_seconds = time.perf_counter() - bayes_started

    full_id_map = canonical_expression_ids(
        template_dir / "SIS_subspaces" / "Uspace.expressions"
    )
    kept_feature_ids = fold_subspace.feature_ids[keep]
    kept_expressions = [
        expression for expression, include in zip(fold_subspace.feature_names, keep) if include
    ]
    deterministic_support = support_payload(
        deterministic_columns,
        fold_subspace.feature_ids,
        fold_subspace.feature_names,
        full_id_map,
    )
    map_support = support_payload(
        posterior.map_columns,
        kept_feature_ids,
        kept_expressions,
        full_id_map,
    )

    observed = y_all[validation_indices]
    intervals = posterior.bma_intervals
    covered = (observed >= intervals[:, 0]) & (observed <= intervals[:, 1])
    prediction_rows: list[dict[str, object]] = []
    for position, source_index in enumerate(validation_indices):
        name = sample_names[int(source_index)]
        family = next((value for value in ("Co", "Fe", "Ni") if name.startswith(value)), "unknown")
        prediction_rows.append(
            {
                "sample_name": name,
                "observed": float(observed[position]),
                "pred_deterministic": float(deterministic_predictions[position]),
                "pred_map": float(posterior.map_predictions[position]),
                "pred_bma": float(posterior.bma_predictions[position]),
                "bma_sd": float(math.sqrt(posterior.bma_variance[position])),
                "bma_lower95": float(intervals[position, 0]),
                "bma_upper95": float(intervals[position, 1]),
                "bma_covered95": bool(covered[position]),
                "x_family": family,
            }
        )

    top_rows: list[dict[str, object]] = []
    for row in posterior.top_supports:
        support = support_payload(
            row["columns"], kept_feature_ids, kept_expressions, full_id_map
        )
        top_rows.append(
            {
                "rank": row["rank"],
                "posterior_probability": row["posterior_probability"],
                "cumulative_probability": row["cumulative_probability"],
                "fold_local_ids": " ".join(map(str, support["fold_local_ids"])),
                "canonical_full_data_ids": " ".join(
                    "NA" if value is None else str(value)
                    for value in support["canonical_full_data_ids"]
                ),
                "expressions": " | ".join(support["expressions"]),
                "log_marginal_likelihood": row["log_marginal_likelihood"],
            }
        )
    feature_rows: list[dict[str, object]] = []
    sis_scores = np.asarray(fold_subspace.metadata["sis_scores"], dtype=float)
    for original_index, include in enumerate(keep):
        feature_rows.append(
            {
                "fold_local_id": int(fold_subspace.feature_ids[original_index]),
                "canonical_full_data_id": full_id_map.get(
                    fold_subspace.feature_names[original_index], "NA"
                ),
                "expression": fold_subspace.feature_names[original_index],
                "sis_score": float(sis_scores[original_index]),
                "train_mean": float(means[original_index]),
                "train_std": float(scales[original_index]),
                "retained_for_bayes": bool(include),
                "pip": float(
                    posterior.inclusion_probabilities[inverse_kept[original_index]]
                ) if include else 0.0,
            }
        )

    payload: dict[str, Any] = {
        "dimension": dimension,
        "repetition": repetition,
        "seed": seed,
        "fold": fold_number,
        "train_indices": [int(value) for value in train_indices],
        "validation_indices": [int(value) for value in validation_indices],
        "n_train": int(len(train_indices)),
        "n_validation": int(len(validation_indices)),
        "screened_features_before_scaling": int(fold_subspace.n_features),
        "screened_features_after_scaling": int(np.sum(keep)),
        "near_constant_features_removed": int(np.sum(~keep)),
        "support_count": posterior.support_count,
        "sigma2_definition": "fold deterministic SISSO training RMSE squared",
        "sigma2": sigma2,
        "tau2_definition": "population variance of y_train divided by dimension",
        "tau2": tau2,
        "deterministic_training_rmse": deterministic_training_rmse,
        "deterministic_support": deterministic_support,
        "map_support": map_support,
        "same_support": same_support(deterministic_support, map_support),
        "rmse_deterministic": rmse(observed, deterministic_predictions),
        "rmse_map": rmse(observed, posterior.map_predictions),
        "rmse_bma": rmse(observed, posterior.bma_predictions),
        "coverage_bma_95": float(np.mean(covered)),
        "mean_bma_interval_width": float(np.mean(intervals[:, 1] - intervals[:, 0])),
        "map_probability": posterior.map_probability,
        "supports_to_95_probability": posterior.supports_to_95,
        "leading_posterior_odds": posterior.leading_posterior_odds,
        "bma_accumulated_probability": posterior.accumulated_probability,
        "bma_omitted_probability": posterior.omitted_probability,
        "expression_reconstruction_max_abs_error": reconstruction_error,
        "sisso_runtime_seconds": sisso_seconds,
        "bayesian_runtime_seconds": bayes_seconds,
        "predictions": prediction_rows,
        "top_supports": top_rows,
    }
    write_fold_csvs(fold_dir, payload, feature_rows, top_rows)
    json_dump(fold_dir / "fold_result.json", payload)
    return payload


def flatten_metric_row(result: dict[str, Any]) -> dict[str, object]:
    def ids(support: dict[str, object]) -> str:
        return " ".join("NA" if value is None else str(value) for value in support["canonical_full_data_ids"])
    return {
        "dimension": result["dimension"],
        "repetition": result["repetition"],
        "seed": result["seed"],
        "fold": result["fold"],
        "n_train": result["n_train"],
        "n_validation": result["n_validation"],
        "K_fold": result["screened_features_after_scaling"],
        "support_count": result["support_count"],
        "deterministic_support_canonical_ids": ids(result["deterministic_support"]),
        "map_support_canonical_ids": ids(result["map_support"]),
        "deterministic_support_expressions": " | ".join(result["deterministic_support"]["expressions"]),
        "map_support_expressions": " | ".join(result["map_support"]["expressions"]),
        "same_support": result["same_support"],
        "rmse_deterministic": result["rmse_deterministic"],
        "rmse_map": result["rmse_map"],
        "rmse_bma": result["rmse_bma"],
        "coverage_bma_95": result["coverage_bma_95"],
        "sigma2": result["sigma2"],
        "tau2": result["tau2"],
        "sisso_runtime_seconds": result["sisso_runtime_seconds"],
        "bayesian_runtime_seconds": result["bayesian_runtime_seconds"],
    }


def rebuild_combined_summaries(output_root: Path) -> None:
    result_paths = sorted(output_root.glob("runs/seed_*/d*/fold_*/fold_result.json"))
    results = [json.loads(path.read_text()) for path in result_paths]
    if not results:
        return
    fold_rows = [flatten_metric_row(result) for result in results]
    with (output_root / "fold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)

    predictions: list[dict[str, object]] = []
    for result in results:
        for row in result["predictions"]:
            predictions.append(
                {
                    "dimension": result["dimension"],
                    "repetition": result["repetition"],
                    "seed": result["seed"],
                    "fold": result["fold"],
                    **row,
                }
            )
    with (output_root / "predictions_all.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    repetition_rows: list[dict[str, object]] = []
    groups = sorted({(int(row["dimension"]), int(row["repetition"]), int(row["seed"])) for row in results})
    for dimension, repetition, seed in groups:
        selected = [
            result for result in results
            if (result["dimension"], result["repetition"], result["seed"])
            == (dimension, repetition, seed)
        ]
        pred_rows = [row for result in selected for row in result["predictions"]]
        observed = np.array([row["observed"] for row in pred_rows], dtype=float)
        det = np.array([row["pred_deterministic"] for row in pred_rows], dtype=float)
        map_pred = np.array([row["pred_map"] for row in pred_rows], dtype=float)
        bma = np.array([row["pred_bma"] for row in pred_rows], dtype=float)
        repetition_rows.append(
            {
                "dimension": dimension,
                "repetition": repetition,
                "seed": seed,
                "completed_folds": len(selected),
                "n_predictions": len(pred_rows),
                "RMSE_det": rmse(observed, det),
                "RMSE_MAP": rmse(observed, map_pred),
                "RMSE_BMA": rmse(observed, bma),
                "Delta_MAP": rmse(observed, map_pred) - rmse(observed, det),
                "Delta_BMA": rmse(observed, bma) - rmse(observed, det),
                "coverage_BMA_95": float(np.mean([row["bma_covered95"] for row in pred_rows])),
                "deterministic_MAP_support_agreement_fraction": float(np.mean([result["same_support"] for result in selected])),
                "total_sisso_runtime_seconds": float(np.sum([result["sisso_runtime_seconds"] for result in selected])),
                "total_bayesian_runtime_seconds": float(np.sum([result["bayesian_runtime_seconds"] for result in selected])),
            }
        )
    with (output_root / "results_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(repetition_rows[0]))
        writer.writeheader()
        writer.writerows(repetition_rows)

    aggregate: list[dict[str, object]] = []
    for dimension in sorted({int(row["dimension"]) for row in repetition_rows}):
        selected = [row for row in repetition_rows if row["dimension"] == dimension and row["completed_folds"] == 5]
        record: dict[str, object] = {"dimension": dimension, "completed_repetitions": len(selected)}
        for metric in ("RMSE_det", "RMSE_MAP", "RMSE_BMA", "Delta_MAP", "Delta_BMA"):
            values = np.array([row[metric] for row in selected], dtype=float)
            record[f"{metric}_mean"] = float(np.mean(values)) if len(values) else None
            record[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            record[f"{metric}_se"] = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
        aggregate.append(record)
    json_dump(
        output_root / "results_summary.json",
        {
            "method": "training-fold SISSO -> residual-informed union -> standardized uniform-prior Bayesian support inference",
            "variance_definitions": {
                "sigma2": "deterministic SISSO training RMSE squared within each fold",
                "tau2": "population variance of y_train divided by d within each fold",
            },
            "repetitions": repetition_rows,
            "aggregate": aggregate,
        },
    )


def main() -> None:
    args = parse_args()
    template_dir = args.sisso_template_dir or PROJECT / f"SISSO-{args.dimension}D"
    run_dir = args.output_root / "runs" / f"seed_{args.seed}" / f"d{args.dimension}"
    run_dir.mkdir(parents=True, exist_ok=True)
    source = parse_train_dat(template_dir / "train.dat")
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    split_indices = list(splitter.split(np.asarray(source["y"])))
    for fold_number, (train_indices, validation_indices) in enumerate(split_indices, start=1):
        result_path = run_dir / f"fold_{fold_number}" / "fold_result.json"
        if result_path.exists() and not args.force:
            print(f"d={args.dimension}, seed={args.seed}, fold={fold_number}: already complete")
            continue
        print(f"d={args.dimension}, seed={args.seed}, fold={fold_number}: running fold-local SISSO")
        result = run_fold(
            dimension=args.dimension,
            repetition=args.repetition,
            seed=args.seed,
            fold_number=fold_number,
            train_indices=train_indices,
            validation_indices=validation_indices,
            template_dir=template_dir,
            executable=args.sisso_executable,
            run_dir=run_dir,
            batch_size=args.batch_size,
            top_n=args.top_n,
        )
        print(
            f"  K={result['screened_features_after_scaling']}; "
            f"RMSE det/MAP/BMA={result['rmse_deterministic']:.4f}/"
            f"{result['rmse_map']:.4f}/{result['rmse_bma']:.4f}; "
            f"SISSO={result['sisso_runtime_seconds']:.1f}s; Bayes={result['bayesian_runtime_seconds']:.1f}s"
        )
        rebuild_combined_summaries(args.output_root)
    rebuild_combined_summaries(args.output_root)


if __name__ == "__main__":
    main()

