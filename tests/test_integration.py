"""End-to-end pipeline test: data -> model -> sweep -> comparison -> drift
check -> sensitivity -> reporting artifacts, at a small scale so it runs
fast in CI."""

from fraud_anomaly_detector.cost import CostAssumptions
from fraud_anomaly_detector.pipeline import run_pipeline
from fraud_anomaly_detector.reporting import plot_cost_vs_threshold, plot_sensitivity, summarize


def test_full_pipeline_runs_and_produces_a_dollar_delta(tmp_path):
    result = run_pipeline(
        n_samples=15_000,
        fraud_rate=0.02,
        cost_assumptions=CostAssumptions(),
        fp_cost_range=[3, 12, 50],
        random_state=123,
    )

    assert result.cost_optimal["total_cost"] <= result.f1_optimal["total_cost"]
    assert result.cost_optimal["total_cost"] <= result.default_result["total_cost"]
    assert 0.0 <= result.cost_optimal["threshold"] <= 1.0
    assert len(result.sensitivity_df) == 3

    summary_text = summarize(result)
    assert "Cost-optimal" in summary_text
    assert "$ saved" in summary_text

    plot_path = plot_cost_vs_threshold(result, tmp_path / "cost_vs_threshold.png")
    sensitivity_plot_path = plot_sensitivity(result, tmp_path / "sensitivity.png")
    assert plot_path.exists() and plot_path.stat().st_size > 0
    assert sensitivity_plot_path.exists() and sensitivity_plot_path.stat().st_size > 0


def test_pipeline_is_reproducible_given_same_seed():
    result_a = run_pipeline(n_samples=8_000, fraud_rate=0.02, random_state=7)
    result_b = run_pipeline(n_samples=8_000, fraud_rate=0.02, random_state=7)
    assert result_a.cost_optimal["threshold"] == result_b.cost_optimal["threshold"]
    assert result_a.cost_optimal["total_cost"] == result_b.cost_optimal["total_cost"]
