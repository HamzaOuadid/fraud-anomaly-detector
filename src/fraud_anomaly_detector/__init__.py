"""fraud_anomaly_detector

A fraud/anomaly classifier whose decision threshold is chosen by sweeping
the real $ cost of false positives vs. false negatives -- not by optimizing
F1 or accuracy.

Public API (mirrors the spec's interface contract):
    train_baseline(data, feature_cols) -> TrainedModel
    sweep_thresholds(model, holdout, cost_fn) -> list[ThresholdResult]
    find_cost_optimal(sweep_results) -> dict
"""

from fraud_anomaly_detector.cost import CostAssumptions, CostResult, evaluate_cost
from fraud_anomaly_detector.data import FEATURE_COLUMNS, generate_transactions
from fraud_anomaly_detector.model import TrainedModel, score, train_baseline
from fraud_anomaly_detector.threshold import (
    ThresholdResult,
    find_cost_optimal,
    find_default_threshold,
    find_f1_optimal,
    sensitivity_analysis,
    sweep_thresholds,
)

__all__ = [
    "CostAssumptions",
    "CostResult",
    "evaluate_cost",
    "FEATURE_COLUMNS",
    "generate_transactions",
    "TrainedModel",
    "score",
    "train_baseline",
    "ThresholdResult",
    "find_cost_optimal",
    "find_default_threshold",
    "find_f1_optimal",
    "sensitivity_analysis",
    "sweep_thresholds",
]

__version__ = "0.1.0"
