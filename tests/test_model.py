"""The baseline classifier must learn a genuine, non-trivial signal from
the synthetic generator (not just memorize noise), and predict_proba
must return well-formed probabilities."""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from fraud_anomaly_detector.data import FEATURE_COLUMNS, generate_transactions
from fraud_anomaly_detector.model import score, train_baseline


def _train_holdout():
    df = generate_transactions(n_samples=30_000, fraud_rate=0.015, random_state=11)
    train_df, holdout_df = train_test_split(
        df, test_size=0.25, stratify=df["label"], random_state=11
    )
    return train_df, holdout_df


def test_model_learns_real_signal_auc_above_chance():
    train_df, holdout_df = _train_holdout()
    model = train_baseline(train_df, FEATURE_COLUMNS)
    y_score = score(model, holdout_df)
    auc = roc_auc_score(holdout_df["label"], y_score)
    # A random/no-signal model gets ~0.5; our generative process has a real
    # (noisy) logistic relationship, so a well-trained model should clear
    # a solid bar well above chance.
    assert auc > 0.75, f"AUC {auc:.3f} too low -- classifier isn't learning the signal"


def test_predict_proba_outputs_are_valid_probabilities():
    train_df, holdout_df = _train_holdout()
    model = train_baseline(train_df, FEATURE_COLUMNS)
    y_score = score(model, holdout_df)
    assert y_score.shape[0] == len(holdout_df)
    assert np.all((y_score >= 0.0) & (y_score <= 1.0))


def test_feature_cols_recorded_on_model():
    train_df, _ = _train_holdout()
    model = train_baseline(train_df, FEATURE_COLUMNS)
    assert model.feature_cols == FEATURE_COLUMNS
