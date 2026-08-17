"""Turns a PipelineResult into real, on-disk evidence: the cost-vs-
threshold tradeoff curve (PNG), the raw sweep table (CSV), and the
FP-cost sensitivity table (CSV)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt
import pandas as pd

from fraud_anomaly_detector.pipeline import PipelineResult


def sweep_results_to_df(pipeline_result: PipelineResult) -> pd.DataFrame:
    return pd.DataFrame([vars(r) for r in pipeline_result.sweep_results])


def plot_cost_vs_threshold(pipeline_result: PipelineResult, out_path: str | Path) -> Path:
    """The real, computed cost-vs-threshold tradeoff curve, with the
    cost-optimal, F1-optimal, and default-0.5 thresholds marked."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sweep_df = sweep_results_to_df(pipeline_result)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(sweep_df["threshold"], sweep_df["total_cost_usd"], color="#1f5fa8", linewidth=2, label="Total $ cost")
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Total $ cost on holdout", color="#1f5fa8")
    ax1.tick_params(axis="y", labelcolor="#1f5fa8")

    ax2 = ax1.twinx()
    ax2.plot(sweep_df["threshold"], sweep_df["f1"], color="#c0392b", linewidth=1.5, linestyle="--", label="F1 score")
    ax2.set_ylabel("F1 score", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    co = pipeline_result.cost_optimal
    f1o = pipeline_result.f1_optimal
    df_default = pipeline_result.default_result
    ax1.axvline(co["threshold"], color="#1f5fa8", linestyle=":", alpha=0.8)
    ax1.axvline(f1o["threshold"], color="#c0392b", linestyle=":", alpha=0.8)
    ax1.axvline(df_default["threshold"], color="#555555", linestyle=":", alpha=0.6)

    ax1.scatter([co["threshold"]], [co["total_cost"]], color="#1f5fa8", zorder=5,
                label=f"Cost-optimal t={co['threshold']:.3f} (${co['total_cost']:,.0f})")
    ax1.scatter([f1o["threshold"]], [f1o["total_cost"]], color="#c0392b", zorder=5,
                label=f"F1-optimal t={f1o['threshold']:.3f} (${f1o['total_cost']:,.0f})")
    ax1.scatter([df_default["threshold"]], [df_default["total_cost"]], color="#555555", zorder=5,
                label=f"Default t=0.5 (${df_default['total_cost']:,.0f})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper center", fontsize=8, framealpha=0.9)
    ax1.set_title("Total $ cost vs. decision threshold (same model, same holdout)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sensitivity(pipeline_result: PipelineResult, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pipeline_result.sensitivity_df

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["fp_cost_usd"], df["optimal_threshold"], marker="o", color="#1f5fa8")
    ax.set_xlabel("Assumed cost of a false positive ($)")
    ax.set_ylabel("Cost-optimal decision threshold")
    ax.set_title("Sensitivity of the optimal threshold to the FP-cost assumption\n(FN cost held fixed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def summarize(pipeline_result: PipelineResult) -> str:
    co = pipeline_result.cost_optimal
    f1o = pipeline_result.f1_optimal
    df_default = pipeline_result.default_result
    dc = pipeline_result.drift_check
    ca = pipeline_result.cost_assumptions

    delta_vs_f1 = f1o["total_cost"] - co["total_cost"]
    delta_vs_default = df_default["total_cost"] - co["total_cost"]

    lines = [
        "=== Cost assumptions ===",
        f"  FP cost:        ${ca.fp_cost_usd:.2f} flat per false decline",
        f"  FN cost:        amount x {ca.fn_loss_fraction:.2f} + ${ca.fn_flat_fee_usd:.2f} flat fee",
        "",
        "=== Threshold comparison (same model, same holdout) ===",
        f"  Cost-optimal:   threshold={co['threshold']:.3f}  total_cost=${co['total_cost']:,.2f}",
        f"  F1-optimal:     threshold={f1o['threshold']:.3f}  total_cost=${f1o['total_cost']:,.2f}  f1={f1o['f1']:.3f}",
        f"  Default (0.5):  threshold={df_default['threshold']:.3f}  total_cost=${df_default['total_cost']:,.2f}",
        "",
        f"  $ saved choosing cost-optimal over F1-optimal:   ${delta_vs_f1:,.2f}",
        f"  $ saved choosing cost-optimal over default 0.5:  ${delta_vs_default:,.2f}",
        "",
        "=== Concept drift check (model trained pre-drift only) ===",
        f"  Pre-drift-optimal threshold:  {dc.pre_drift_optimal_threshold:.3f}",
        f"  Post-drift-optimal threshold: {dc.post_drift_optimal_threshold:.3f}",
        f"  Cost of reusing stale pre-drift threshold on post-drift data: ${dc.stale_threshold_cost_usd:,.2f}",
        f"  Cost of recalibrating threshold on post-drift data:           ${dc.recalibrated_cost_usd:,.2f}",
        f"  $ penalty for NOT recalibrating after drift:                  ${dc.drift_penalty_usd:,.2f}",
    ]
    return "\n".join(lines)
