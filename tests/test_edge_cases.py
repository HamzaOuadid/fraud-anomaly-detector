"""Edge cases the spec calls out explicitly (Sec. 9):

  1. Extreme class imbalance making some thresholds produce zero positives
     -- must not divide by zero.
  2. Concept drift over time -- a threshold optimized on one period should
     not be assumed optimal on a later, drifted period.
  3. Threshold/cost-assumption sensitivity -- cost assumptions are
     debatable, so the optimal threshold's sensitivity to them must be
     shown across a plausible range, not just one fixed number.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from fraud_anomaly_detector.cost import CostAssumptions, evaluate_cost
from fraud_anomaly_detector.data import DriftConfig, FEATURE_COLUMNS, generate_transactions
from fraud_anomaly_detector.model import score, train_baseline
from fraud_anomaly_detector.threshold import (
    find_cost_optimal,
    sensitivity_analysis,
    sweep_thresholds,
    sweep_thresholds_from_arrays,
)


# ---------------------------------------------------------------------------
# 1. Extreme imbalance / zero-positive thresholds
# ---------------------------------------------------------------------------

def test_zero_predicted_positives_does_not_raise_and_costs_only_fn():
    y_true = np.array([1, 0, 0, 0, 0])
    y_score = np.array([0.4, 0.1, 0.2, 0.05, 0.15])
    amounts = np.array([300.0, 10.0, 20.0, 5.0, 8.0])
    # threshold above every score -> zero predicted positives
    result = evaluate_cost(y_true, y_score, amounts, threshold=0.99, assumptions=CostAssumptions())
    assert result.tp == 0 and result.fp == 0
    assert result.fn == 1
    assert result.fp_cost_usd == 0.0
    assert result.fn_cost_usd > 0.0  # missed the one real fraud


def test_precision_and_recall_are_zero_not_nan_when_no_positives_predicted():
    y_true = np.array([1, 0, 0])
    y_score = np.array([0.2, 0.1, 0.05])
    amounts = np.array([50.0, 10.0, 5.0])
    results = sweep_thresholds_from_arrays(
        y_true, y_score, amounts, CostAssumptions(), thresholds=np.array([0.99])
    )
    r = results[0]
    assert r.tp == 0 and r.fp == 0
    assert r.precision == 0.0  # 0/0 guarded, not NaN
    assert not np.isnan(r.precision)
    assert not np.isnan(r.f1)


def test_extreme_imbalance_sweep_runs_end_to_end_without_error():
    """A dataset with a very low fraud rate (0.2%) should sweep cleanly,
    including thresholds near 1.0 where predicted positives hit zero."""
    df = generate_transactions(n_samples=50_000, fraud_rate=0.002, random_state=99)
    train_df, holdout_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=99)
    model = train_baseline(train_df, FEATURE_COLUMNS)
    results = sweep_thresholds(model, holdout_df, CostAssumptions())
    assert len(results) > 0
    assert all(np.isfinite(r.total_cost_usd) for r in results)
    assert all(not np.isnan(r.precision) and not np.isnan(r.f1) for r in results)


# ---------------------------------------------------------------------------
# 2. Concept drift
# ---------------------------------------------------------------------------

def test_threshold_optimized_pre_drift_is_suboptimal_post_drift():
    drift = DriftConfig(drift_day=90)
    df = generate_transactions(n_samples=100_000, fraud_rate=0.015, drift=drift, random_state=55)
    pre = df[df["day_index"] < drift.drift_day]
    post = df[df["day_index"] >= drift.drift_day]

    train_df, holdout_pre = train_test_split(pre, test_size=0.3, stratify=pre["label"], random_state=55)
    model = train_baseline(train_df, FEATURE_COLUMNS)  # trained ONLY on pre-drift data

    assumptions = CostAssumptions()
    pre_results = sweep_thresholds(model, holdout_pre, assumptions)
    post_results = sweep_thresholds(model, post, assumptions)

    pre_optimal = find_cost_optimal(pre_results)
    post_optimal = find_cost_optimal(post_results)

    # Cost of applying the STALE pre-drift-optimal threshold to post-drift
    # data, vs. the cost of the threshold recomputed specifically for
    # post-drift data.
    y_true_post = post["label"].to_numpy()
    y_score_post = score(model, post)
    amounts_post = post["amount"].to_numpy()
    stale_cost = evaluate_cost(
        y_true_post, y_score_post, amounts_post, pre_optimal["threshold"], assumptions
    ).total_cost_usd
    recalibrated_cost = post_optimal["total_cost"]

    # Recalibrating on the drifted distribution should be at least as good
    # (in $) as blindly reusing the stale threshold -- and in this
    # generator, strictly better, since the signal composition genuinely
    # shifted.
    assert recalibrated_cost <= stale_cost
    assert pre_optimal["threshold"] != post_optimal["threshold"]


# ---------------------------------------------------------------------------
# 3. Sensitivity to (debatable) cost assumptions
# ---------------------------------------------------------------------------

def test_sensitivity_analysis_shows_a_range_not_one_point():
    df = generate_transactions(n_samples=20_000, fraud_rate=0.015, random_state=33)
    train_df, holdout_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=33)
    model = train_baseline(train_df, FEATURE_COLUMNS)
    y_true = holdout_df["label"].to_numpy()
    y_score = score(model, holdout_df)
    amounts = holdout_df["amount"].to_numpy()

    sweep_df = sensitivity_analysis(
        y_true, y_score, amounts, CostAssumptions(), fp_cost_range=[1, 5, 12, 30, 60, 120]
    )
    assert len(sweep_df) == 6
    # A cheap false-positive assumption should push the optimal threshold
    # LOW (flag aggressively); an expensive one should push it HIGH
    # (flag conservatively).
    low_fp_cost_threshold = sweep_df.loc[sweep_df["fp_cost_usd"] == 1, "optimal_threshold"].iloc[0]
    high_fp_cost_threshold = sweep_df.loc[sweep_df["fp_cost_usd"] == 120, "optimal_threshold"].iloc[0]
    assert low_fp_cost_threshold < high_fp_cost_threshold
