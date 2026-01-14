"""Metrics computation and aggregation rules."""

import pandas as pd
from datasets import load_dataset

from bolinas.evals import aggregate_metrics, compute_metrics


def get_dataset_config(dataset_name):
    """Get configuration for a specific dataset."""
    for dataset in config["datasets"]:
        if dataset["name"] == dataset_name:
            return dataset
    raise ValueError(f"Dataset {dataset_name} not found in config")


rule compute_metrics:
    """Compute metrics for a dataset and model checkpoint."""
    input:
        scores="results/scores/{dataset}/{model}/{step}.parquet",
    output:
        "results/metrics/{dataset}/{model}/{step}.parquet",
    params:
        dataset_config=lambda wildcards: get_dataset_config(wildcards.dataset),
    run:
        # Load dataset directly from HuggingFace
        hf_dataset = load_dataset(
            params.dataset_config["hf_path"], split=params.dataset_config["split"]
        )
        dataset = hf_dataset.to_pandas()

        # Load scores
        scores = pd.read_parquet(input.scores)

        # Compute metrics (globally and per subset)
        metrics = compute_metrics(
            dataset=dataset,
            scores=scores,
            metrics=params.dataset_config["metrics"],
            score_columns=["minus_llr", "abs_llr", "embed_last_l2", "embed_middle_l2"],
        )

        # Save results
        metrics.to_parquet(output[0], index=False)
