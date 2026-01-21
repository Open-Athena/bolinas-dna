"""Metric computation utilities for evaluating variant effect predictions."""

from typing import Callable

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

METRIC_FUNCTIONS: dict[str, Callable[[pd.Series, pd.Series], float]] = {
    "AUPRC": lambda label, score: average_precision_score(label, score),
    "AUROC": lambda label, score: roc_auc_score(label, score),
    "Spearman": lambda label, score: spearmanr(label, score)[0],
}


def compute_metrics(
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    metrics: list[str],
    score_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute evaluation metrics for variant predictions.

    Aligns dataset with scores by row index and computes specified metrics both
    globally and on each subset (if dataset contains a 'subset' column). For datasets
    with subsets, computes metrics on the full dataset (subset='global') and
    on each individual subset.

    Args:
        dataset: DataFrame with columns [chrom, pos, ref, alt, label] and optionally [subset].
        scores: DataFrame with columns [minus_llr, abs_llr, ...] aligned by index with dataset.
        metrics: List of metric names to compute (e.g., ['AUPRC', 'AUROC']).
        score_columns: List of score column names to evaluate. If None, uses ['minus_llr', 'abs_llr'].

    Returns:
        DataFrame with columns [metric, score_type, subset, value].
        Each row represents one metric computed on one score type for one subset.
    """
    if score_columns is None:
        score_columns = ["minus_llr", "abs_llr"]

    # Combine dataset and scores by index (assumes same order)
    merged = pd.concat(
        [dataset.reset_index(drop=True), scores.reset_index(drop=True)], axis=1
    )

    # Build list of subsets to evaluate
    subsets_to_evaluate = [("global", merged)]
    if "subset" in merged.columns:
        for subset_name in merged["subset"].unique():
            subset_data = merged[merged["subset"] == subset_name]
            subsets_to_evaluate.append((subset_name, subset_data))

    # Compute metrics for each subset
    results = []
    for subset_name, subset_data in subsets_to_evaluate:
        n_pos = int((subset_data["label"] == 1).sum())
        n_neg = int((subset_data["label"] == 0).sum())
        for metric_name in metrics:
            metric_func = METRIC_FUNCTIONS[metric_name]
            for score_col in score_columns:
                value = metric_func(subset_data["label"], subset_data[score_col])
                results.append(
                    {
                        "metric": metric_name,
                        "score_type": score_col,
                        "subset": subset_name,
                        "value": value,
                        "n_pos": n_pos,
                        "n_neg": n_neg,
                    }
                )

    return pd.DataFrame(results)


def aggregate_metrics(
    metric_files: list[str], dataset_names: list[str], model_steps: list[str]
) -> pd.DataFrame:
    """Aggregate metrics from multiple evaluation runs into a single DataFrame.

    Args:
        metric_files: List of paths to metric parquet files.
        dataset_names: List of dataset names corresponding to each file.
        model_steps: List of model training steps corresponding to each file.

    Returns:
        DataFrame with columns [step, dataset, metric, score_type, subset, value].
        Sorted by step and dataset for easier analysis.
    """
    all_metrics = []

    for file_path, dataset_name, step in zip(metric_files, dataset_names, model_steps):
        df = pd.read_parquet(file_path)
        df["dataset"] = dataset_name
        df["step"] = int(step)
        all_metrics.append(df)

    result = pd.concat(all_metrics, ignore_index=True)
    return result.sort_values(["step", "dataset"]).reset_index(drop=True)
