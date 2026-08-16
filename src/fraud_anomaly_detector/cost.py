"""Explicit $ cost function for false positives vs. false negatives.

This module exists so the cost assumptions are a first-class, readable,
importable object -- not a constant buried inside a training script. Every
number here is a *stated assumption*, with reasoning, per the spec's first
user story ("I can see the actual cost assumptions, not a hidden black
box").

Assumptions (defaults; all overridable, and swept over in
``sensitivity_analysis`` in threshold.py so the published results are not
a single fragile point estimate):

False Negative (a real fraud transaction we let through):
  - The merchant/bank eats the transaction amount as a chargeback loss.
    Modeled as ``amount * fn_loss_fraction`` (default fraction = 1.0, i.e.
    the full amount is lost -- conservative but standard for card-not-
    present fraud where goods/funds are gone).
  - Plus a flat ``fn_flat_fee_usd`` (default $25) for chargeback/dispute
    processing: card network fees, investigator time, and the "dispute
    handling" cost that is charged per incident regardless of transaction
    size. This number is a rough blend of published card-network
    chargeback fee ranges ($15-$40).

False Positive (a legitimate transaction we block/flag):
  - A flat ``fp_cost_usd`` (default $12) approximating the blended cost of
    a false decline: support-agent review time (~$5-8) plus an amortized
    estimate of customer-friction/churn risk from a wrongly-declined
    purchase. This is explicitly a simplification -- a real FP cost should
    scale with customer lifetime value, which we do not model here -- and
    is exactly why we run a sensitivity sweep over a plausible FP-cost
    range rather than reporting one fixed number.

True positives and true negatives are assumed to cost $0 marginal
(the transaction is handled normally either way).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REASONING = {
    "fp_cost_usd": (
        "Blended cost of a false decline: ~$5-8 support-agent review time "
        "plus an amortized customer-friction/churn-risk estimate for "
        "wrongly blocking a legitimate purchase. Does not scale with "
        "customer lifetime value -- treated as a stated simplification, "
        "see sensitivity_analysis()."
    ),
    "fn_loss_fraction": (
        "Fraction of the transaction amount the merchant/issuer loses "
        "when a fraud is missed. Default 1.0 (full amount) is standard "
        "for card-not-present fraud where the goods or funds are gone."
    ),
    "fn_flat_fee_usd": (
        "Flat chargeback/dispute-processing fee charged per fraud "
        "incident regardless of transaction size (card network fee + "
        "investigator time), independent of the transaction amount lost."
    ),
}


@dataclass(frozen=True)
class CostAssumptions:
    """The explicit $ cost model. See module docstring for reasoning."""

    fp_cost_usd: float = 12.0
    fn_flat_fee_usd: float = 25.0
    fn_loss_fraction: float = 1.0
    reasoning: dict = field(default_factory=lambda: dict(REASONING))

    def fn_cost_for_amounts(self, amounts: np.ndarray) -> np.ndarray:
        return amounts * self.fn_loss_fraction + self.fn_flat_fee_usd


@dataclass(frozen=True)
class CostResult:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    fp_cost_usd: float
    fn_cost_usd: float
    total_cost_usd: float


def evaluate_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    assumptions: CostAssumptions,
) -> CostResult:
    """Apply the cost function to the confusion matrix at one threshold.

    Handles the extreme-imbalance edge case explicitly: when a threshold
    produces zero predicted positives (or the holdout has zero actual
    positives), the fp/fn arrays are simply empty and the sums are 0 --
    no division ever happens in this function, so there's nothing to
    divide by zero. (Precision/recall, which *do* divide, are computed
    separately in threshold.py with explicit zero-guards.)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    amounts = np.asarray(amounts, dtype=float)

    y_pred = (y_score >= threshold).astype(int)

    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    fn_mask = (y_true == 1) & (y_pred == 0)

    tp, fp, tn, fn = int(tp_mask.sum()), int(fp_mask.sum()), int(tn_mask.sum()), int(fn_mask.sum())

    fp_cost = fp * assumptions.fp_cost_usd
    fn_cost = float(assumptions.fn_cost_for_amounts(amounts[fn_mask]).sum()) if fn > 0 else 0.0
    total_cost = fp_cost + fn_cost

    return CostResult(
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        fp_cost_usd=float(fp_cost),
        fn_cost_usd=fn_cost,
        total_cost_usd=float(total_cost),
    )
