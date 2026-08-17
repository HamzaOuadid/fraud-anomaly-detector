"""SQLite persistence for the `predictions` and `threshold_sweep_results`
tables from the spec's data model."""

import numpy as np

from fraud_anomaly_detector.cost import CostAssumptions
from fraud_anomaly_detector.db import (
    init_db,
    load_predictions,
    load_sweep_results,
    save_predictions,
    save_sweep_results,
)
from fraud_anomaly_detector.threshold import sweep_thresholds_from_arrays


def test_predictions_round_trip(tmp_path):
    conn = init_db(tmp_path / "test.db")
    ids = ["t1", "t2", "t3"]
    y_true = np.array([1, 0, 0])
    y_score = np.array([0.9, 0.2, 0.55])
    save_predictions(conn, ids, y_true, y_score)

    df = load_predictions(conn)
    assert len(df) == 3
    assert set(df.columns) == {"transaction_id", "true_label", "predicted_probability"}
    row = df[df["transaction_id"] == "t1"].iloc[0]
    assert row["true_label"] == 1
    assert row["predicted_probability"] == 0.9


def test_sweep_results_round_trip(tmp_path):
    conn = init_db(tmp_path / "test.db")
    y_true = np.array([1, 0, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    amounts = np.array([100.0, 50.0, 200.0, 10.0, 20.0])
    results = sweep_thresholds_from_arrays(
        y_true, y_score, amounts, CostAssumptions(), thresholds=np.array([0.0, 0.5, 1.0])
    )
    save_sweep_results(conn, results)

    df = load_sweep_results(conn)
    assert len(df) == 3
    assert "total_cost_usd" in df.columns
    assert df["total_cost_usd"].min() >= 0


def test_save_overwrites_previous_run(tmp_path):
    conn = init_db(tmp_path / "test.db")
    save_predictions(conn, ["a"], np.array([1]), np.array([0.5]))
    save_predictions(conn, ["b"], np.array([0]), np.array([0.1]))
    df = load_predictions(conn)
    assert len(df) == 1
    assert df.iloc[0]["transaction_id"] == "b"
