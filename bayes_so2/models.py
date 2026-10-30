from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DiscoveredSISSOFiles:
    sisso_dir: Path
    manuscript_pdf: Path | None
    sisso_in: Path | None
    sisso_out: Path | None
    train_dat: Path | None
    uspace_expressions: Path | None
    uspace_dat: Path | None
    models_ranking: Path | None
    models_coefficients: Path | None
    descriptor_data: Path | None


@dataclass
class DeterministicModel:
    dimension: int
    feature_ids: tuple[int, ...]
    feature_expressions: tuple[str, ...]
    coefficients: np.ndarray
    intercept: float
    rmse: float | None
    maxae: float | None
    rank: int = 1
    model_id: int = 1


@dataclass
class ScreenedSubspace:
    X: np.ndarray
    y: np.ndarray
    feature_ids: np.ndarray
    feature_names: list[str]
    sample_names: list[str]
    target_name: str
    complexity_vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    deterministic_model: DeterministicModel | None = None

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


@dataclass
class SupportRecord:
    column_indices: tuple[int, ...]
    feature_ids: tuple[int, ...]
    feature_names: tuple[str, ...]
    log_marginal_likelihood: float
    log_prior: float
    log_weight: float
    posterior_probability: float
    intercept_mean: float
    beta_mean: np.ndarray
    beta_cov: np.ndarray


@dataclass
class BayesianSOResult:
    dimension: int
    fixed_size: bool
    sigma2: float
    tau2: float
    prior_name: str
    fit_intercept: bool
    support_records: list[SupportRecord]
    inclusion_probabilities: np.ndarray
    map_support_feature_ids: tuple[int, ...]
    map_support_feature_names: tuple[str, ...]
    map_intercept: float
    map_coefficients: np.ndarray
    posterior_mean_intercept: float
    posterior_mean_coefficients: np.ndarray
    coefficient_variances: np.ndarray
    coefficient_intervals: np.ndarray
    predictive_mean: np.ndarray | None
    predictive_variance: np.ndarray | None
    predictive_intervals: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)
