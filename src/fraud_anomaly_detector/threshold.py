"""Threshold sweep: total $ cost (and, for comparison, F1) across a range
of decision thresholds on ONE trained model's ONE set of predicted
probabilities.

This is the module that satisfies the second user story: the comparison
between the cost-optimal threshold and the F1-optimal / default-0.5
threshold must vary *only* the threshold, holding the trained model (and
therefore the predicted-probability array) fixed. ``sweep_thresholds``
below calls ``score()`` on the model exactly once and reuses that single
``y_score`` array for every threshold in the sweep -- see
``test_threshold.py::test_predict_proba_called_once`` for a regression
test of that property.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_anomaly_detector.cost import CostAssumptions, evaluate_cost
from fraud_anomaly_detector.model import TrainedModel, score


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    fp_cost_usd: float
    fn_cost_usd: float
    total_cost_usd: float


def _safe_div(numerator: float, denominator: float) -> float:
    """Division with the extreme-imbalance edge case handled: 0/0 -> 0.0
    instead of raising or returning NaN, matching sklearn's
    ``zero_division=0`` convention."""
    return float(numerator) / denominator if denominator > 0 else 0.0


def _threshold_result_from_arrays(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    assumptions: CostAssumptions,
) -> ThresholdResult:
    cost_result = evaluate_cost(y_true, y_score, amounts, threshold, assumptions)
    precision = _safe_div(cost_result.tp, cost_result.tp + cost_result.fp)
    recall = _safe_div(cost_result.tp, cost_result.tp + cost_result.fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return ThresholdResult(
        threshold=cost_result.threshold,
        tp=cost_result.tp,
        fp=cost_result.fp,
        tn=cost_result.tn,
        fn=cost_result.fn,
        precision=precision,
        recall=recall,
        f1=f1,
        fp_cost_usd=cost_result.fp_cost_usd,
        fn_cost_usd=cost_result.fn_cost_usd,
        total_cost_usd=cost_result.total_cost_usd,
    )


def sweep_thresholds_from_arrays(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    cost_fn: CostAssumptions,
    thresholds: np.ndarray | None = None,
) -> list[ThresholdResult]:
    """Array-level sweep (used directly by tests and by sensitivity
    analysis, and internally by ``sweep_thresholds``)."""
    if thresholds is None:
        thresholds = np.round(np.linspace(0.0, 1.0, 501), 6)
    return [
        _threshold_result_from_arrays(y_true, y_score, amounts, float(t), cost_fn)
        for t in thresholds
    ]


def sweep_thresholds(
    model: TrainedModel,
    holdout: pd.DataFrame,
    cost_fn: CostAssumptions,
    thresholds: np.ndarray | None = None,
    label_col: str = "label",
    amount_col: str = "amount",
) -> list[ThresholdResult]:
    """``sweep_thresholds(model, holdout, cost_fn) -> list[ThresholdResult]``

    Scores ``holdout`` with ``model`` exactly once, then evaluates the
    stated cost function at every threshold in the sweep against that
    single, fixed set of predicted probabilities.
    """
    y_true = holdout[label_col].to_numpy()
    amounts = holdout[amount_col].to_numpy()
    y_score = score(model, holdout)  # <-- called once; reused for every threshold
    return sweep_thresholds_from_arrays(y_true, y_score, amounts, cost_fn, thresholds)


def find_cost_optimal(sweep_results: list[ThresholdResult]) -> dict:
    """``find_cost_optimal(sweep_results) -> {threshold, total_cost}``"""
    best = min(sweep_results, key=lambda r: r.total_cost_usd)
    return {"threshold": best.threshold, "total_cost": best.total_cost_usd, "result": best}


def find_f1_optimal(sweep_results: list[ThresholdResult]) -> dict:
    """Same idea as ``find_cost_optimal`` but maximizing F1 -- this is the
    "accuracy/F1-optimal" comparison point the spec asks to publish
    against the cost-optimal point, on the same model."""
    best = max(sweep_results, key=lambda r: r.f1)
    return {"threshold": best.threshold, "total_cost": best.total_cost_usd, "f1": best.f1, "result": best}


def find_default_threshold(sweep_results: list[ThresholdResult], threshold: float = 0.5) -> dict:
    """The naive default-0.5 threshold, for the same comparison."""
    closest = min(sweep_results, key=lambda r: abs(r.threshold - threshold))
    return {"threshold": closest.threshold, "total_cost": closest.total_cost_usd, "result": closest}


def sensitivity_analysis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    base_assumptions: CostAssumptions,
    fp_cost_range: list[float],
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Show sensitivity of the cost-optimal threshold to the (debatable)
    FP-cost assumption, per the spec's edge case: "cost assumptions are
    inherently debatable -- show sensitivity across a plausible range."

    Holds fn_flat_fee_usd / fn_loss_fraction fixed and varies fp_cost_usd
    across ``fp_cost_range``, recomputing the cost-optimal threshold for
    each. Returns a tidy DataFrame, one row per fp_cost value.
    """
    rows = []
    for fp_cost in fp_cost_range:
        assumptions = CostAssumptions(
            fp_cost_usd=fp_cost,
            fn_flat_fee_usd=base_assumptions.fn_flat_fee_usd,
            fn_loss_fraction=base_assumptions.fn_loss_fraction,
        )
        sweep = sweep_thresholds_from_arrays(y_true, y_score, amounts, assumptions, thresholds)
        best = find_cost_optimal(sweep)
        rows.append(
            {
                "fp_cost_usd": fp_cost,
                "optimal_threshold": best["threshold"],
                "optimal_total_cost_usd": best["total_cost"],
            }
        )
    return pd.DataFrame(rows)
