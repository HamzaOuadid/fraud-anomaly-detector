"""End-to-end pipeline gluing data -> model -> sweep -> comparison -> drift
check -> sensitivity, used by both the CLI and the integration test so
there's exactly one implementation of "how a run works."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_anomaly_detector.cost import CostAssumptions, evaluate_cost
from fraud_anomaly_detector.data import DEFAULT_DRIFT, DriftConfig, FEATURE_COLUMNS, generate_transactions
from fraud_anomaly_detector.model import TrainedModel, score, train_baseline
from fraud_anomaly_detector.threshold import (
    ThresholdResult,
    find_cost_optimal,
    find_default_threshold,
    find_f1_optimal,
    sensitivity_analysis,
    sweep_thresholds,
)

DEFAULT_FP_COST_RANGE = [3, 6, 12, 20, 35, 50, 75, 100]


@dataclass
class DriftCheck:
    pre_drift_optimal_threshold: float
    post_drift_optimal_threshold: float
    stale_threshold_cost_usd: float  # applying pre-drift threshold to post-drift data
    recalibrated_cost_usd: float  # post-drift threshold recomputed on post-drift data
    drift_penalty_usd: float  # stale - recalibrated (cost of NOT recalibrating)


@dataclass
class PipelineResult:
    df: pd.DataFrame
    model: TrainedModel
    holdout_df: pd.DataFrame
    sweep_results: list[ThresholdResult]
    cost_optimal: dict
    f1_optimal: dict
    default_result: dict
    sensitivity_df: pd.DataFrame
    drift_check: DriftCheck
    cost_assumptions: CostAssumptions


def run_pipeline(
    n_samples: int = 80_000,
    fraud_rate: float = 0.012,
    cost_assumptions: CostAssumptions | None = None,
    drift: DriftConfig = DEFAULT_DRIFT,
    fp_cost_range: list[float] | None = None,
    random_state: int = 42,
) -> PipelineResult:
    cost_assumptions = cost_assumptions or CostAssumptions()
    fp_cost_range = fp_cost_range or DEFAULT_FP_COST_RANGE

    df = generate_transactions(
        n_samples=n_samples, fraud_rate=fraud_rate, drift=drift, random_state=random_state
    )
    pre = df[df["day_index"] < drift.drift_day]
    post = df[df["day_index"] >= drift.drift_day]

    train_df, holdout_df = train_test_split(
        pre, test_size=0.25, stratify=pre["label"], random_state=random_state
    )
    model = train_baseline(train_df, FEATURE_COLUMNS, random_state=random_state)

    sweep_results = sweep_thresholds(model, holdout_df, cost_assumptions)
    cost_optimal = find_cost_optimal(sweep_results)
    f1_optimal = find_f1_optimal(sweep_results)
    default_result = find_default_threshold(sweep_results, threshold=0.5)

    y_true = holdout_df["label"].to_numpy()
    y_score = score(model, holdout_df)
    amounts = holdout_df["amount"].to_numpy()
    sensitivity_df = sensitivity_analysis(
        y_true, y_score, amounts, cost_assumptions, fp_cost_range=fp_cost_range
    )

    # Concept-drift check: model trained pre-drift, evaluated on post-drift
    # transactions it has never seen the pattern for.
    post_results = sweep_thresholds(model, post, cost_assumptions)
    post_optimal = find_cost_optimal(post_results)
    y_true_post = post["label"].to_numpy()
    y_score_post = score(model, post)
    amounts_post = post["amount"].to_numpy()
    stale_cost = evaluate_cost(
        y_true_post, y_score_post, amounts_post, cost_optimal["threshold"], cost_assumptions
    ).total_cost_usd
    drift_check = DriftCheck(
        pre_drift_optimal_threshold=cost_optimal["threshold"],
        post_drift_optimal_threshold=post_optimal["threshold"],
        stale_threshold_cost_usd=stale_cost,
        recalibrated_cost_usd=post_optimal["total_cost"],
        drift_penalty_usd=stale_cost - post_optimal["total_cost"],
    )

    return PipelineResult(
        df=df,
        model=model,
        holdout_df=holdout_df,
        sweep_results=sweep_results,
        cost_optimal=cost_optimal,
        f1_optimal=f1_optimal,
        default_result=default_result,
        sensitivity_df=sensitivity_df,
        drift_check=drift_check,
        cost_assumptions=cost_assumptions,
    )
