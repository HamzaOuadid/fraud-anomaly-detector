"""CLI entry point: `fraud-detector run` executes the full pipeline and
writes real, reproducible evidence (SQLite tables, PNG plots, CSV sweep
tables) to an output directory."""

from __future__ import annotations

from pathlib import Path

import typer

from fraud_anomaly_detector.cost import CostAssumptions
from fraud_anomaly_detector.db import init_db, save_predictions, save_sweep_results
from fraud_anomaly_detector.model import score
from fraud_anomaly_detector.pipeline import run_pipeline
from fraud_anomaly_detector.reporting import plot_cost_vs_threshold, plot_sensitivity, summarize

app = typer.Typer(add_completion=False, help="Fraud detector with a cost-justified decision threshold.")


@app.command()
def run(
    n_samples: int = typer.Option(80_000, help="Number of synthetic transactions to generate."),
    fraud_rate: float = typer.Option(0.012, help="Target fraud rate (fraction of transactions)."),
    fp_cost: float = typer.Option(12.0, help="Assumed $ cost of a false positive (false decline)."),
    fn_flat_fee: float = typer.Option(25.0, help="Assumed flat $ chargeback/dispute fee per missed fraud."),
    fn_loss_fraction: float = typer.Option(1.0, help="Fraction of the transaction amount lost per missed fraud."),
    out_dir: str = typer.Option("reports", help="Directory to write the DB, plots, and CSVs to."),
    random_state: int = typer.Option(42, help="Random seed for data generation and model training."),
) -> None:
    """Run the full pipeline: generate data, train the baseline model,
    sweep thresholds against the stated $ cost function, find the
    cost-optimal threshold, compare it to F1-optimal/default, check
    sensitivity to the cost assumptions, and check for concept drift."""
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    cost_assumptions = CostAssumptions(
        fp_cost_usd=fp_cost, fn_flat_fee_usd=fn_flat_fee, fn_loss_fraction=fn_loss_fraction
    )

    typer.echo(f"Generating {n_samples} synthetic transactions (fraud_rate={fraud_rate})...")
    result = run_pipeline(
        n_samples=n_samples,
        fraud_rate=fraud_rate,
        cost_assumptions=cost_assumptions,
        random_state=random_state,
    )

    db_path = out_dir_path / "fraud.db"
    conn = init_db(db_path)
    y_true = result.holdout_df["label"].to_numpy()
    y_score = score(result.model, result.holdout_df)
    save_predictions(conn, result.holdout_df["transaction_id"].tolist(), y_true, y_score)
    save_sweep_results(conn, result.sweep_results)
    conn.close()
    typer.echo(f"Wrote predictions + sweep results to {db_path}")

    sweep_csv = out_dir_path / "threshold_sweep.csv"
    from fraud_anomaly_detector.reporting import sweep_results_to_df

    sweep_results_to_df(result).to_csv(sweep_csv, index=False)
    result.sensitivity_df.to_csv(out_dir_path / "sensitivity.csv", index=False)

    plot_path = plot_cost_vs_threshold(result, out_dir_path / "cost_vs_threshold.png")
    sensitivity_plot_path = plot_sensitivity(result, out_dir_path / "sensitivity.png")
    typer.echo(f"Wrote {plot_path}")
    typer.echo(f"Wrote {sensitivity_plot_path}")
    typer.echo(f"Wrote {sweep_csv}")

    typer.echo("")
    typer.echo(summarize(result))


if __name__ == "__main__":
    app()
