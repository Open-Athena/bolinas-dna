"""Tests for metrics computation functions."""

import pandas as pd

from bolinas.evals.metrics import METRIC_FUNCTIONS, aggregate_metrics, compute_metrics


def test_metric_functions_auprc():
    """Test AUPRC metric computation."""
    labels = pd.Series([1, 1, 0, 0, 1])
    scores = pd.Series([0.9, 0.8, 0.6, 0.3, 0.7])

    auprc = METRIC_FUNCTIONS["AUPRC"](labels, scores)
    assert 0.0 <= auprc <= 1.0
    assert isinstance(auprc, float)


def test_metric_functions_auroc():
    """Test AUROC metric computation."""
    labels = pd.Series([1, 1, 0, 0, 1])
    scores = pd.Series([0.9, 0.8, 0.6, 0.3, 0.7])

    auroc = METRIC_FUNCTIONS["AUROC"](labels, scores)
    assert 0.0 <= auroc <= 1.0
    assert isinstance(auroc, float)


def test_metric_functions_spearman():
    """Test Spearman correlation computation."""
    labels = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    scores = pd.Series([1.1, 2.2, 2.9, 4.1, 5.0])

    spearman = METRIC_FUNCTIONS["Spearman"](labels, scores)
    assert -1.0 <= spearman <= 1.0
    assert isinstance(spearman, float)


def test_compute_metrics_without_subsets():
    """Test metric computation on dataset without subsets.

    Should compute metrics globally only, for each score type.
    """
    dataset = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1", "chr1"],
            "pos": [100, 200, 300, 400, 500],
            "ref": ["A", "C", "G", "T", "A"],
            "alt": ["T", "G", "A", "C", "G"],
            "label": [1, 1, 0, 0, 1],
        }
    )

    scores = pd.DataFrame(
        {
            "minus_llr": [0.9, 0.8, 0.6, 0.3, 0.7],
            "abs_llr": [0.9, 0.8, 0.6, 0.3, 0.7],
        }
    )

    metrics = compute_metrics(
        dataset=dataset,
        scores=scores,
        metrics=["AUPRC", "AUROC"],
        score_columns=["minus_llr", "abs_llr"],
    )

    # Should have 2 metrics * 2 score types = 4 rows
    assert len(metrics) == 4
    assert set(metrics["metric"]) == {"AUPRC", "AUROC"}
    assert set(metrics["score_type"]) == {"minus_llr", "abs_llr"}
    assert set(metrics["subset"]) == {"global"}
    assert all(metrics["value"].notna())


def test_compute_metrics_with_subsets():
    """Test metric computation on dataset with subsets.

    Should compute metrics globally and for each subset.
    """
    dataset = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1", "chr1", "chr1"],
            "pos": [100, 200, 300, 400, 500, 600],
            "ref": ["A", "C", "G", "T", "A", "C"],
            "alt": ["T", "G", "A", "C", "G", "T"],
            "label": [1, 1, 0, 0, 1, 0],
            "subset": ["5UTR", "5UTR", "3UTR", "3UTR", "5UTR", "3UTR"],
        }
    )

    scores = pd.DataFrame(
        {
            "minus_llr": [0.9, 0.8, 0.6, 0.3, 0.7, 0.4],
            "abs_llr": [0.9, 0.8, 0.6, 0.3, 0.7, 0.4],
        }
    )

    metrics = compute_metrics(
        dataset=dataset,
        scores=scores,
        metrics=["AUPRC"],
        score_columns=["minus_llr"],
    )

    # Should have 1 metric * 1 score type * 3 subsets (global, 5UTR, 3UTR) = 3 rows
    assert len(metrics) == 3
    assert set(metrics["subset"]) == {"global", "5UTR", "3UTR"}
    assert all(metrics["metric"] == "AUPRC")
    assert all(metrics["score_type"] == "minus_llr")
    assert all(metrics["value"].notna())


def test_aggregate_metrics():
    """Test aggregating metrics from multiple files."""
    # Create temporary parquet files
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create metric files
        metrics1 = pd.DataFrame(
            {
                "metric": ["AUPRC", "AUROC"],
                "score_type": ["llr_minus", "llr_minus"],
                "subset": ["global", "global"],
                "value": [0.85, 0.80],
            }
        )
        metrics1.to_parquet(tmpdir / "metrics1.parquet")

        metrics2 = pd.DataFrame(
            {
                "metric": ["AUPRC", "AUROC"],
                "score_type": ["llr_minus", "llr_minus"],
                "subset": ["global", "global"],
                "value": [0.90, 0.85],
            }
        )
        metrics2.to_parquet(tmpdir / "metrics2.parquet")

        # Aggregate
        result = aggregate_metrics(
            metric_files=[
                str(tmpdir / "metrics1.parquet"),
                str(tmpdir / "metrics2.parquet"),
            ],
            dataset_names=["dataset1", "dataset1"],
            model_steps=["10000", "20000"],
        )

        # Should have 4 rows (2 metrics * 2 steps)
        assert len(result) == 4
        assert set(result["step"]) == {10000, 20000}
        assert set(result["dataset"]) == {"dataset1"}
        assert set(result["metric"]) == {"AUPRC", "AUROC"}
        assert all(result["value"].notna())


def test_compute_metrics_default_score_columns():
    """Test that compute_metrics uses default score columns when not specified."""
    dataset = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "pos": [100],
            "ref": ["A"],
            "alt": ["T"],
            "label": [1],
        }
    )

    scores = pd.DataFrame(
        {
            "minus_llr": [0.9],
            "abs_llr": [0.9],
        }
    )

    metrics = compute_metrics(dataset=dataset, scores=scores, metrics=["AUPRC"])

    # Should use default score_columns=['minus_llr', 'abs_llr']
    assert set(metrics["score_type"]) == {"minus_llr", "abs_llr"}
