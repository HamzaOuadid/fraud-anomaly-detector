"""Baseline gradient-boosting fraud classifier.

Handles the "extreme class imbalance" edge case at training time via
per-sample class-balanced weights (``sklearn.utils.class_weight.
compute_sample_weight``) rather than resampling: with a ~1% positive
rate, naive training would let the model minimize loss by predicting
"not fraud" almost everywhere. Sample-weighting is the simplest fix that
keeps every real row in the training set (no synthetic oversampling
noise), and it composes cleanly with HistGradientBoostingClassifier,
which does not expose a ``class_weight`` argument the way e.g.
RandomForestClassifier does.

(imbalanced-learn / SMOTE was considered and deliberately not added as a
dependency: since the actual decision boundary comes from an empirically
swept *threshold*, not from the raw 0.5 cutoff, class-weighting the loss
is sufficient here and keeps the dependency surface small. This tradeoff
is called out again in the README.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


@dataclass
class TrainedModel:
    """A fitted classifier plus the exact feature ordering it expects."""

    estimator: HistGradientBoostingClassifier
    feature_cols: list[str]


def train_baseline(
    data: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    random_state: int = 42,
) -> TrainedModel:
    """Train the baseline classifier (`train_baseline(data) -> Model`).

    ``data`` must contain ``feature_cols`` and ``label_col``. Uses
    HistGradientBoostingClassifier -- a standard gradient-boosting
    ensemble -- with class-balanced sample weights to handle the
    realistic ~1% fraud rate.
    """
    X = data[feature_cols].to_numpy()
    y = data[label_col].to_numpy()
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)

    estimator = HistGradientBoostingClassifier(
        max_iter=250,
        max_depth=6,
        learning_rate=0.08,
        random_state=random_state,
    )
    estimator.fit(X, y, sample_weight=sample_weight)
    return TrainedModel(estimator=estimator, feature_cols=list(feature_cols))


def score(model: TrainedModel, data: pd.DataFrame) -> np.ndarray:
    """Return P(fraud) for each row of ``data``, in row order."""
    X = data[model.feature_cols].to_numpy()
    return model.estimator.predict_proba(X)[:, 1]
