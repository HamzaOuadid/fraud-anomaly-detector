"""SQLite persistence for the two tables in the spec's data model:

    predictions(transaction_id, true_label, predicted_probability)
    threshold_sweep_results(threshold, tp, fp, tn, fn, total_cost_usd)

SQLite (not Postgres) is used deliberately: this is a single-analyst
reproducible-report tool, not a multi-writer service, and it keeps the
whole project runnable with zero external services.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_anomaly_detector.threshold import ThresholdResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    transaction_id TEXT PRIMARY KEY,
    true_label INTEGER NOT NULL,
    predicted_probability REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS threshold_sweep_results (
    threshold REAL NOT NULL,
    tp INTEGER NOT NULL,
    fp INTEGER NOT NULL,
    tn INTEGER NOT NULL,
    fn INTEGER NOT NULL,
    precision REAL NOT NULL,
    recall REAL NOT NULL,
    f1 REAL NOT NULL,
    fp_cost_usd REAL NOT NULL,
    fn_cost_usd REAL NOT NULL,
    total_cost_usd REAL NOT NULL
);
"""


def init_db(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def save_predictions(
    conn: sqlite3.Connection,
    transaction_ids: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> None:
    conn.execute("DELETE FROM predictions")
    rows = list(zip(transaction_ids, [int(v) for v in y_true], [float(v) for v in y_score]))
    conn.executemany(
        "INSERT INTO predictions (transaction_id, true_label, predicted_probability) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()


def save_sweep_results(conn: sqlite3.Connection, sweep_results: list[ThresholdResult]) -> None:
    conn.execute("DELETE FROM threshold_sweep_results")
    rows = [
        (
            r.threshold,
            r.tp,
            r.fp,
            r.tn,
            r.fn,
            r.precision,
            r.recall,
            r.f1,
            r.fp_cost_usd,
            r.fn_cost_usd,
            r.total_cost_usd,
        )
        for r in sweep_results
    ]
    conn.executemany(
        """INSERT INTO threshold_sweep_results
           (threshold, tp, fp, tn, fn, precision, recall, f1, fp_cost_usd, fn_cost_usd, total_cost_usd)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def load_predictions(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM predictions", conn)


def load_sweep_results(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM threshold_sweep_results", conn)
