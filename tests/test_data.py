"""Sanity checks on the synthetic data generator: correct imbalance,
determinism, and a genuine concept-drift regime shift (needed for the
concept-drift edge case tested elsewhere)."""

import numpy as np

from fraud_anomaly_detector.data import DriftConfig, FEATURE_COLUMNS, generate_transactions


def test_fraud_rate_close_to_target():
    df = generate_transactions(n_samples=40_000, fraud_rate=0.012, random_state=1)
    rate = df["label"].mean()
    assert 0.008 < rate < 0.018  # within a plausible band of the 1.2% target


def test_deterministic_given_seed():
    df1 = generate_transactions(n_samples=5_000, random_state=7)
    df2 = generate_transactions(n_samples=5_000, random_state=7)
    assert df1.equals(df2)


def test_different_seeds_differ():
    df1 = generate_transactions(n_samples=5_000, random_state=1)
    df2 = generate_transactions(n_samples=5_000, random_state=2)
    assert not df1.equals(df2)


def test_all_feature_columns_present_and_finite():
    df = generate_transactions(n_samples=5_000, random_state=3)
    for col in FEATURE_COLUMNS:
        assert col in df.columns
        assert np.isfinite(df[col].to_numpy()).all()
    assert set(df["label"].unique()) <= {0, 1}


def test_class_imbalance_is_extreme_enough_to_matter():
    df = generate_transactions(n_samples=40_000, fraud_rate=0.012, random_state=4)
    counts = df["label"].value_counts()
    assert counts[1] < counts[0] * 0.05  # fraud is well under 5% of normals


def test_concept_drift_changes_fraud_pattern_composition():
    """Pre- and post-drift fraud rows should differ systematically on the
    features whose weights the DriftConfig swaps (is_foreign vs.
    velocity_24h), so a model trained pre-drift sees a different signal
    post-drift."""
    drift = DriftConfig(drift_day=90)
    df = generate_transactions(n_samples=80_000, fraud_rate=0.02, drift=drift, random_state=5)
    pre_fraud = df[(df["day_index"] < drift.drift_day) & (df["label"] == 1)]
    post_fraud = df[(df["day_index"] >= drift.drift_day) & (df["label"] == 1)]
    assert len(pre_fraud) > 50 and len(post_fraud) > 50

    pre_foreign_rate = pre_fraud["is_foreign"].mean()
    post_foreign_rate = post_fraud["is_foreign"].mean()
    pre_velocity = pre_fraud["velocity_24h"].mean()
    post_velocity = post_fraud["velocity_24h"].mean()

    # Pre-drift fraud should lean more foreign-transaction than post-drift.
    assert pre_foreign_rate > post_foreign_rate
    # Post-drift fraud should lean more velocity-burst than pre-drift.
    assert post_velocity > pre_velocity
