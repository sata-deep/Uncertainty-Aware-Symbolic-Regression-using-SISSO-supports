"""Exact fixed-size Gaussian support inference for the revision calculations.

The implementation is intentionally independent of the legacy ``bayes_so2``
prediction path.  It standardizes outside this module, enumerates every support,
and accumulates validation predictive moments over the complete posterior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class ExactSupportResult:
    dimension: int
    support_count: int
    log_evidence: float
    map_columns: tuple[int, ...]
    map_probability: float
    map_beta_mean: np.ndarray
    map_beta_cov: np.ndarray
    map_predictions: np.ndarray
    bma_predictions: np.ndarray
    bma_variance: np.ndarray
    bma_intervals: np.ndarray
    inclusion_probabilities: np.ndarray
    top_supports: list[dict[str, object]]
    supports_to_95: int
    leading_posterior_odds: float
    accumulated_probability: float
    omitted_probability: float
    y_mean: float
    x_mean: np.ndarray


def enumerate_support_array(n_features: int, dimension: int) -> np.ndarray:
    """Return all fixed-size supports as a compact int32 array."""
    if dimension < 1 or dimension > n_features:
        raise ValueError("dimension must satisfy 1 <= dimension <= n_features")
    if dimension == 1:
        return np.arange(n_features, dtype=np.int32)[:, None]
    if dimension == 2:
        left, right = np.triu_indices(n_features, k=1)
        return np.column_stack((left, right)).astype(np.int32, copy=False)
    if dimension == 3:
        count = math.comb(n_features, 3)
        supports = np.empty((count, 3), dtype=np.int32)
        cursor = 0
        for first in range(n_features - 2):
            second, third = np.triu_indices(n_features - first - 1, k=1)
            block = len(second)
            supports[cursor : cursor + block, 0] = first
            supports[cursor : cursor + block, 1] = second + first + 1
            supports[cursor : cursor + block, 2] = third + first + 1
            cursor += block
        if cursor != count:
            raise RuntimeError(f"support enumeration filled {cursor} rows, expected {count}")
        return supports
    raise ValueError("revision exact enumeration currently supports d=1, 2, or 3")


def _posterior_batch(
    supports: np.ndarray,
    gram: np.ndarray,
    cross: np.ndarray,
    y_sum_squares: float,
    n_samples: int,
    sigma2: float,
    tau2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return log marginal likelihood, beta mean, and beta covariance by support."""
    dimension = supports.shape[1]
    precision = gram[supports[:, :, None], supports[:, None, :]] / sigma2
    precision += np.eye(dimension, dtype=float)[None, :, :] / tau2
    rhs = cross[supports] / sigma2
    sign, logdet_precision = np.linalg.slogdet(precision)
    if not np.all(sign > 0):
        bad = np.flatnonzero(sign <= 0)
        precision[bad] += 1.0e-10 * np.eye(dimension)[None, :, :]
        sign, logdet_precision = np.linalg.slogdet(precision)
        if not np.all(sign > 0):
            raise np.linalg.LinAlgError("non-positive posterior precision")
    beta_cov = np.linalg.inv(precision)
    beta_mean = np.einsum("bij,bj->bi", beta_cov, rhs, optimize=True)
    quadratic = y_sum_squares / sigma2 - np.einsum(
        "bi,bi->b", rhs, beta_mean, optimize=True
    )
    logdet_sigma = (
        n_samples * math.log(sigma2)
        + dimension * math.log(tau2)
        + logdet_precision
    )
    log_marginal = -0.5 * (
        n_samples * LOG_2PI + logdet_sigma + quadratic
    )
    return log_marginal, beta_mean, beta_cov


def exact_uniform_support_inference(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    dimension: int,
    sigma2: float,
    tau2: float,
    *,
    batch_size: int = 50_000,
    top_n: int = 100,
    probability_threshold: float = 1.0e-15,
) -> ExactSupportResult:
    """Enumerate the uniform-prior posterior and numerical BMA moments.

    The intercept has an improper flat prior and is handled by centering.  The
    predictive variance includes observation noise, intercept uncertainty,
    coefficient uncertainty, and between-support uncertainty. Every support is
    used for normalization and PIPs; predictive accumulation omits only
    supports below probability_threshold and reports that omitted mass.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_predict = np.asarray(X_predict, dtype=float)
    if X_train.ndim != 2 or X_predict.ndim != 2:
        raise ValueError("X_train and X_predict must be two-dimensional")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train row counts differ")
    if X_predict.shape[1] != X_train.shape[1]:
        raise ValueError("training and prediction feature counts differ")
    if sigma2 <= 0.0 or tau2 <= 0.0:
        raise ValueError("sigma2 and tau2 must be positive")

    n_samples, n_features = X_train.shape
    supports = enumerate_support_array(n_features, dimension)
    support_count = supports.shape[0]
    x_mean = np.mean(X_train, axis=0)
    y_mean = float(np.mean(y_train))
    X_centered = X_train - x_mean
    X_predict_centered = X_predict - x_mean
    y_centered = y_train - y_mean
    gram = X_centered.T @ X_centered
    cross = X_centered.T @ y_centered
    y_sum_squares = float(y_centered @ y_centered)

    log_weights = np.empty(support_count, dtype=float)
    for start in range(0, support_count, batch_size):
        stop = min(start + batch_size, support_count)
        log_weights[start:stop] = _posterior_batch(
            supports[start:stop], gram, cross, y_sum_squares,
            n_samples, sigma2, tau2,
        )[0]

    log_evidence = float(logsumexp(log_weights))
    probabilities = np.exp(log_weights - log_evidence)
    probability_sum = float(np.sum(probabilities))

    inclusion = np.zeros(n_features, dtype=float)
    for active_position in range(dimension):
        np.add.at(inclusion, supports[:, active_position], probabilities)

    map_index = int(np.argmax(log_weights))
    map_support = supports[map_index : map_index + 1]
    _, map_beta, map_cov = _posterior_batch(
        map_support, gram, cross, y_sum_squares, n_samples, sigma2, tau2
    )
    map_columns = tuple(int(value) for value in map_support[0])
    map_predictions = y_mean + X_predict_centered[:, list(map_columns)] @ map_beta[0]

    bma_mean = np.zeros(X_predict.shape[0], dtype=float)
    bma_second = np.zeros(X_predict.shape[0], dtype=float)
    included_mass = 0.0
    base_variance = sigma2 * (1.0 + 1.0 / n_samples)
    for start in range(0, support_count, batch_size):
        stop = min(start + batch_size, support_count)
        support_batch = supports[start:stop]
        probability_batch = probabilities[start:stop]
        keep = probability_batch >= probability_threshold
        if not np.any(keep):
            continue
        support_batch = support_batch[keep]
        probability_batch = probability_batch[keep]
        _, beta_mean, beta_cov = _posterior_batch(
            support_batch, gram, cross, y_sum_squares,
            n_samples, sigma2, tau2,
        )
        active_predictors = np.transpose(
            X_predict_centered[:, support_batch], (1, 0, 2)
        )
        conditional_mean = y_mean + np.einsum(
            "bnd,bd->bn", active_predictors, beta_mean, optimize=True
        )
        conditional_variance = base_variance + np.einsum(
            "bni,bij,bnj->bn",
            active_predictors,
            beta_cov,
            active_predictors,
            optimize=True,
        )
        bma_mean += probability_batch @ conditional_mean
        bma_second += probability_batch @ (
            conditional_variance + conditional_mean**2
        )
        included_mass += float(np.sum(probability_batch))

    if included_mass <= 0.0:
        raise RuntimeError("no posterior mass accumulated for BMA prediction")
    # The threshold omits at most a few parts per billion for the support spaces
    # used here. Renormalization makes the approximation explicit and stable.
    bma_mean /= included_mass
    bma_second /= included_mass
    bma_variance = np.maximum(bma_second - bma_mean**2, 0.0)
    bma_sd = np.sqrt(bma_variance)
    bma_intervals = np.column_stack(
        (bma_mean - 1.959963984540054 * bma_sd,
         bma_mean + 1.959963984540054 * bma_sd)
    )

    order = np.argsort(probabilities)[::-1]
    cumulative = np.cumsum(probabilities[order])
    supports_to_95 = int(np.searchsorted(cumulative, 0.95, side="left") + 1)
    top_count = min(top_n, support_count)
    top_supports: list[dict[str, object]] = []
    for rank, index in enumerate(order[:top_count], start=1):
        top_supports.append(
            {
                "rank": rank,
                "columns": [int(value) for value in supports[index]],
                "posterior_probability": float(probabilities[index]),
                "cumulative_probability": float(cumulative[rank - 1]),
                "log_marginal_likelihood": float(log_weights[index]),
            }
        )
    if support_count > 1:
        log_odds = float(log_weights[order[0]] - log_weights[order[1]])
        leading_odds = math.exp(log_odds) if log_odds < 709.0 else math.inf
    else:
        leading_odds = math.inf

    return ExactSupportResult(
        dimension=dimension,
        support_count=support_count,
        log_evidence=log_evidence,
        map_columns=map_columns,
        map_probability=float(probabilities[map_index]),
        map_beta_mean=map_beta[0],
        map_beta_cov=map_cov[0],
        map_predictions=map_predictions,
        bma_predictions=bma_mean,
        bma_variance=bma_variance,
        bma_intervals=bma_intervals,
        inclusion_probabilities=inclusion,
        top_supports=top_supports,
        supports_to_95=supports_to_95,
        leading_posterior_odds=leading_odds,
        accumulated_probability=included_mass,
        omitted_probability=max(0.0, probability_sum - included_mass),
        y_mean=y_mean,
        x_mean=x_mean,
    )
