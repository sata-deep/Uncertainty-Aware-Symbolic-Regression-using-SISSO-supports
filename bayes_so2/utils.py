from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


LOG_2PI = math.log(2.0 * math.pi)
Z95 = 1.959963984540054


def logsumexp(values: np.ndarray) -> float:
    """Stable log-sum-exp for one-dimensional arrays."""
    if values.size == 0:
        return -np.inf
    vmax = float(np.max(values))
    if not np.isfinite(vmax):
        return vmax
    return vmax + float(np.log(np.sum(np.exp(values - vmax))))


def normal_interval_from_mean_var(
    mean: np.ndarray,
    variance: np.ndarray,
    z_value: float = Z95,
) -> np.ndarray:
    """Return approximate symmetric intervals from moments."""
    std = np.sqrt(np.maximum(variance, 0.0))
    return np.column_stack((mean - z_value * std, mean + z_value * std))


def operator_count(expression: str) -> int:
    """Fallback complexity heuristic based on symbolic operator count."""
    tokens = re.findall(r"exp|sqrt|\^-?\d+|[+\-*/]", expression)
    return max(1, len(tokens))


def choose_support_count(n_features: int, dimension: int, fixed_size: bool) -> int:
    """Count enumerated supports for exact inference."""
    if fixed_size:
        return math.comb(n_features, dimension)
    return sum(math.comb(n_features, size) for size in range(dimension + 1))


def json_ready(value: Any) -> Any:
    """Recursively convert arrays, paths, and numpy scalars to JSON types."""
    if dataclasses.is_dataclass(value):
        return json_ready(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
