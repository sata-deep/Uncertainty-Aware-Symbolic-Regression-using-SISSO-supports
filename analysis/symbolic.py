"""Evaluation and identity helpers for the restricted SISSO expression set."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


SAFE_EXPRESSION = re.compile(r"^[A-Za-z0-9_+*/().^ -]+$")


def evaluate_expressions(
    expressions: list[str],
    primary_feature_names: list[str],
    primary_features: np.ndarray,
) -> np.ndarray:
    """Evaluate SISSO arithmetic expressions without target information."""
    primary_features = np.asarray(primary_features, dtype=float)
    if primary_features.ndim != 2:
        raise ValueError("primary_features must be a two-dimensional array")
    if primary_features.shape[1] != len(primary_feature_names):
        raise ValueError("primary feature name/value counts differ")
    environment = {
        name: primary_features[:, index]
        for index, name in enumerate(primary_feature_names)
    }
    columns: list[np.ndarray] = []
    for expression in expressions:
        if not SAFE_EXPRESSION.fullmatch(expression):
            raise ValueError(f"unsupported token in SISSO expression: {expression}")
        python_expression = expression.replace("^", "**")
        try:
            value = eval(python_expression, {"__builtins__": {}}, environment)
        except Exception as exc:
            raise ValueError(f"failed to evaluate SISSO expression: {expression}") from exc
        column = np.asarray(value, dtype=float)
        if column.ndim == 0:
            column = np.full(primary_features.shape[0], float(column))
        if column.shape != (primary_features.shape[0],):
            raise ValueError(f"unexpected shape for expression: {expression}")
        if not np.all(np.isfinite(column)):
            raise ValueError(f"non-finite validation value for expression: {expression}")
        columns.append(column)
    return np.column_stack(columns) if columns else np.empty((primary_features.shape[0], 0))


def canonical_expression_ids(expressions_path: Path) -> dict[str, int]:
    """Map exact expression strings to one-based IDs in a saved full-data union."""
    mapping: dict[str, int] = {}
    for feature_id, line in enumerate(expressions_path.read_text().splitlines(), start=1):
        expression = line.split("SIS_score", 1)[0].strip()
        mapping[expression] = feature_id
    return mapping


def standardize_train_validation(
    X_train: np.ndarray,
    X_validation: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize from training rows and safely exclude near-constant columns."""
    X_train = np.asarray(X_train, dtype=float)
    X_validation = np.asarray(X_validation, dtype=float)
    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0, ddof=0)
    magnitude = np.maximum(np.max(np.abs(X_train), axis=0), 1.0)
    keep = np.isfinite(scale) & (scale > relative_tolerance * magnitude)
    if not np.any(keep):
        raise ValueError("all fold-specific screened features are numerically constant")
    scaled_train = (X_train[:, keep] - mean[keep]) / scale[keep]
    scaled_validation = (X_validation[:, keep] - mean[keep]) / scale[keep]
    if not np.all(np.isfinite(scaled_train)) or not np.all(np.isfinite(scaled_validation)):
        raise ValueError("non-finite value after training-fold standardization")
    return scaled_train, scaled_validation, mean, scale, keep

