"""Threshold sweep behavior: determinism/reproducibility (testing plan),
and the "same model, only the threshold varies" property required by
user story 2."""

from unittest.mock import patch

import numpy as np
from sklearn.model_selection import train_test_split

from fraud_anomaly_detector.cost import CostAssumptions
from fraud_anomaly_detector.data import FEATURE_COLUMNS, generate_transactions
from fraud_anomaly_detector.model import train_baseline
from fraud_anomaly_detector.threshold import (
    find_cost_optimal,
    find_default_threshold,
    find_f1_optimal,
    sweep_thresholds,
)


def _fitted_model_and_holdout():
    df = generate_transactions(n_samples=20_000, fraud_rate=0.015, random_state=21)
    train_df, holdout_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=21)
    model = train_baseline(train_df, FEATURE_COLUMNS)
    return model, holdout_df


def test_sweep_is_deterministic_given_same_model_and_holdout():
    model, holdout_df = _fitted_model_and_holdout()
    assumptions = CostAssumptions()
    thresholds = np.linspace(0, 1, 51)
    results_a = sweep_thresholds(model, holdout_df, assumptions, thresholds=thresholds)
    results_b = sweep_thresholds(model, holdout_df, assumptions, thresholds=thresholds)
    assert results_a == results_b


def test_predict_proba_called_exactly_once_per_sweep():
    """The comparison must vary only the threshold on ONE trained model's
    ONE set of predictions -- not re-score per threshold."""
    model, holdout_df = _fitted_model_and_holdout()
    assumptions = CostAssumptions()
    thresholds = np.linspace(0, 1, 51)
    with patch.object(
        model.estimator, "predict_proba", wraps=model.estimator.predict_proba
    ) as spy:
        sweep_thresholds(model, holdout_df, assumptions, thresholds=thresholds)
        assert spy.call_count == 1


def test_cost_optimal_is_actually_the_minimum():
    model, holdout_df = _fitted_model_and_holdout()
    assumptions = CostAssumptions()
    results = sweep_thresholds(model, holdout_df, assumptions)
    optimal = find_cost_optimal(results)
    brute_force_min = min(r.total_cost_usd for r in results)
    assert optimal["total_cost"] == brute_force_min


def test_f1_optimal_is_actually_the_maximum():
    model, holdout_df = _fitted_model_and_holdout()
    assumptions = CostAssumptions()
    results = sweep_thresholds(model, holdout_df, assumptions)
    f1_optimal = find_f1_optimal(results)
    brute_force_max = max(r.f1 for r in results)
    assert f1_optimal["f1"] == brute_force_max


def test_cost_optimal_threshold_beats_or_ties_f1_and_default_on_dollar_cost():
    """The whole point of the project: minimizing $ cost directly should
    never do WORSE, in $, than optimizing F1 or using the naive default."""
    model, holdout_df = _fitted_model_and_holdout()
    assumptions = CostAssumptions()
    results = sweep_thresholds(model, holdout_df, assumptions)
    cost_optimal = find_cost_optimal(results)
    f1_optimal = find_f1_optimal(results)
    default = find_default_threshold(results, threshold=0.5)
    assert cost_optimal["total_cost"] <= f1_optimal["total_cost"]
    assert cost_optimal["total_cost"] <= default["total_cost"]
