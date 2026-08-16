"""Synthetic, realistically-imbalanced fraud transaction dataset.

Why synthetic rather than a downloaded public dataset:
  - No Kaggle auth / large binary CSV needed to reproduce the repo.
  - Full control over injecting a *concept drift* regime, which the spec
    calls out as an edge case to handle and which real public dumps
    (a single static CSV) cannot demonstrate.
  - The generative process below is documented in full below and in the
    README, so nothing about "genuine fraud/normal patterns" is a black
    box -- every feature's relationship to the fraud label is explicit.

Explicit assumption (stated per spec Sec. 13 "Risks"): the fraud rate
used here (1.2%) is chosen for statistical stability of the threshold
sweep on a laptop-sized sample, not because it is claimed to match the
real-world base rate for card-present/card-not-present fraud (which is
usually far lower, ~0.1-0.5%). This affects the *absolute* $ numbers
reported but not the methodology, which is the actual claim of this
project (see README "Limitations").
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "is_foreign",
    "merchant_risk_score",
    "distance_from_home_km",
    "velocity_24h",
    "account_age_days",
    "amount_to_avg_ratio",
]

LABEL_COLUMN = "label"


@dataclass(frozen=True)
class DriftConfig:
    """Describes how the fraud-generating process shifts over time.

    Fraud patterns are not static: fraudsters adapt to whatever signal a
    detector was trained on. We simulate one such regime shift at
    ``drift_day`` -- before it, fraud correlates most strongly with
    "foreign card-present" transactions; after it, fraud shifts toward a
    velocity-burst pattern (many transactions in a short window), which is
    a documented real-world adaptation once foreign-transaction blocking
    becomes common.
    """

    drift_day: int = 90
    pre_is_foreign_weight: float = 1.6
    post_is_foreign_weight: float = 0.5
    pre_velocity_weight: float = 0.6
    post_velocity_weight: float = 2.0


DEFAULT_DRIFT = DriftConfig()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def _fit_intercept_for_rate(logit_no_intercept: np.ndarray, target_rate: float, rng: np.random.Generator) -> float:
    """Binary-search an intercept so that E[sigmoid(logit + b)] ~= target_rate."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = _sigmoid(logit_no_intercept + mid).mean()
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate_transactions(
    n_samples: int = 80_000,
    fraud_rate: float = 0.012,
    n_days: int = 150,
    drift: DriftConfig = DEFAULT_DRIFT,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic, time-ordered fraud transaction dataset.

    Each row is one transaction. Features are drawn from realistic marginal
    distributions; the fraud label is then sampled from a logistic function
    of those features (not the other way around), so the classifier has to
    learn a genuine, noisy, non-trivial signal -- not a deterministic rule.

    Returns a DataFrame with columns: transaction_id, day_index,
    FEATURE_COLUMNS..., label.
    """
    rng = np.random.default_rng(random_state)

    day_index = rng.integers(0, n_days, size=n_samples)

    amount = rng.lognormal(mean=3.6, sigma=1.1, size=n_samples)  # ~ median $37, heavy tail
    hour_of_day = rng.integers(0, 24, size=n_samples)
    is_foreign = rng.binomial(1, 0.045, size=n_samples)
    merchant_risk_score = rng.beta(2, 6, size=n_samples)  # skewed low, most merchants low-risk
    distance_from_home_km = rng.exponential(scale=12.0, size=n_samples)
    velocity_24h = rng.poisson(lam=1.3, size=n_samples)
    account_age_days = rng.exponential(scale=400.0, size=n_samples).clip(1, 4000)
    amount_to_avg_ratio = rng.lognormal(mean=0.0, sigma=0.6, size=n_samples)

    late_night = ((hour_of_day >= 1) & (hour_of_day <= 5)).astype(float)

    pre_mask = day_index < drift.drift_day
    is_foreign_weight = np.where(pre_mask, drift.pre_is_foreign_weight, drift.post_is_foreign_weight)
    velocity_weight = np.where(pre_mask, drift.pre_velocity_weight, drift.post_velocity_weight)

    logit = (
        0.55 * _zscore(amount)
        + is_foreign_weight * is_foreign
        + 1.8 * merchant_risk_score
        + 0.35 * _zscore(distance_from_home_km)
        + velocity_weight * _zscore(velocity_24h)
        - 0.30 * _zscore(account_age_days)
        + 0.70 * _zscore(amount_to_avg_ratio)
        + 0.45 * late_night
        + rng.normal(0, 1.0, size=n_samples)  # irreducible noise -> label is never deterministic
    )

    intercept = _fit_intercept_for_rate(logit, fraud_rate, rng)
    fraud_prob = _sigmoid(logit + intercept)
    label = rng.binomial(1, fraud_prob)

    transaction_id = [str(uuid.UUID(int=rng.integers(0, 2**63) << 64 | rng.integers(0, 2**63))) for _ in range(n_samples)]

    df = pd.DataFrame(
        {
            "transaction_id": transaction_id,
            "day_index": day_index,
            "amount": amount,
            "hour_of_day": hour_of_day,
            "is_foreign": is_foreign,
            "merchant_risk_score": merchant_risk_score,
            "distance_from_home_km": distance_from_home_km,
            "velocity_24h": velocity_24h,
            "account_age_days": account_age_days,
            "amount_to_avg_ratio": amount_to_avg_ratio,
            LABEL_COLUMN: label,
        }
    )
    return df.sort_values("day_index").reset_index(drop=True)
