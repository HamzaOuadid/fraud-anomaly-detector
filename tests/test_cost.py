"""Sanity-check the cost function against hand-computed examples at a few
thresholds, per the spec's testing plan, BEFORE trusting the full sweep.

Also covers user story 1's acceptance criterion: the cost assumptions
must be explicit and readable, not a hidden black box.
"""

import numpy as np
import pytest

from fraud_anomaly_detector.cost import CostAssumptions, evaluate_cost

# Toy 5-row set, hand-verifiable.
Y_TRUE = np.array([1, 0, 1, 0, 0])
Y_SCORE = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
AMOUNTS = np.array([100.0, 50.0, 200.0, 10.0, 20.0])

DEFAULT = CostAssumptions()  # fp_cost=12, fn_flat_fee=25, fn_loss_fraction=1.0


def test_cost_assumptions_are_explicit_with_reasoning():
    """Story 1 AC: assumptions are visible and each comes with reasoning."""
    assumptions = CostAssumptions()
    assert assumptions.fp_cost_usd == 12.0
    assert assumptions.fn_flat_fee_usd == 25.0
    assert assumptions.fn_loss_fraction == 1.0
    for key in ("fp_cost_usd", "fn_loss_fraction", "fn_flat_fee_usd"):
        assert key in assumptions.reasoning
        assert len(assumptions.reasoning[key]) > 20  # a real sentence, not a stub


def test_hand_computed_threshold_0_5():
    # predicted = [1, 1, 0, 0, 0] at threshold 0.5
    # tp: idx0 (true=1,pred=1); fp: idx1 (true=0,pred=1)
    # fn: idx2 (true=1,pred=0, amount=200); tn: idx3, idx4
    result = evaluate_cost(Y_TRUE, Y_SCORE, AMOUNTS, threshold=0.5, assumptions=DEFAULT)
    assert (result.tp, result.fp, result.tn, result.fn) == (1, 1, 2, 1)
    assert result.fp_cost_usd == pytest.approx(12.0)  # 1 fp * $12
    assert result.fn_cost_usd == pytest.approx(200.0 * 1.0 + 25.0)  # missed $200 fraud + $25 fee
    assert result.total_cost_usd == pytest.approx(12.0 + 225.0)


def test_hand_computed_threshold_0_predicts_everyone_positive():
    # predicted = [1,1,1,1,1]: tp = idx0,2 (2); fp = idx1,3,4 (3); fn=0; tn=0
    result = evaluate_cost(Y_TRUE, Y_SCORE, AMOUNTS, threshold=0.0, assumptions=DEFAULT)
    assert (result.tp, result.fp, result.tn, result.fn) == (2, 3, 0, 0)
    assert result.fp_cost_usd == pytest.approx(3 * 12.0)
    assert result.fn_cost_usd == pytest.approx(0.0)
    assert result.total_cost_usd == pytest.approx(36.0)


def test_hand_computed_threshold_above_max_score_predicts_everyone_negative():
    # threshold > all scores: predicted = [0,0,0,0,0]
    # fn = idx0 (amount 100), idx2 (amount 200); tp=0; fp=0; tn=3
    result = evaluate_cost(Y_TRUE, Y_SCORE, AMOUNTS, threshold=1.0, assumptions=DEFAULT)
    assert (result.tp, result.fp, result.tn, result.fn) == (0, 0, 3, 2)
    assert result.fp_cost_usd == pytest.approx(0.0)
    expected_fn = (100.0 + 25.0) + (200.0 + 25.0)
    assert result.fn_cost_usd == pytest.approx(expected_fn)
    assert result.total_cost_usd == pytest.approx(expected_fn)


def test_higher_fp_cost_assumption_increases_total_cost_at_low_threshold():
    """Cost function must actually respond to the stated assumptions."""
    cheap = CostAssumptions(fp_cost_usd=1.0)
    expensive = CostAssumptions(fp_cost_usd=100.0)
    cheap_result = evaluate_cost(Y_TRUE, Y_SCORE, AMOUNTS, threshold=0.0, assumptions=cheap)
    expensive_result = evaluate_cost(Y_TRUE, Y_SCORE, AMOUNTS, threshold=0.0, assumptions=expensive)
    assert expensive_result.total_cost_usd > cheap_result.total_cost_usd
