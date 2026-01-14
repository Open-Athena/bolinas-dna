"""Plotting rules for visualizing evaluation results."""

import pandas as pd

from bolinas.evals import plot_metrics_vs_step


rule plot_metrics_vs_step:
    """Plot metrics vs training step for a specific model across all datasets and subsets."""
    input:
        # All metrics for this model across all datasets and steps
        lambda wildcards: [
            f"results/metrics/{dataset}/{wildcards.model}/{step}.parquet"
            for dataset in get_all_datasets()
            for step in get_model_config(wildcards.model)["steps"]
        ],
    output:
        "results/plots/metrics_vs_step/{model}.svg",
    run:
        # Load and aggregate all metrics for this model
        all_metrics = []
        for file_path in input:
            df = pd.read_parquet(file_path)
            # Extract dataset and step from path: results/metrics/{dataset}/{model}/{step}.parquet
            parts = file_path.split("/")
            dataset = parts[2]
            step = int(parts[4].replace(".parquet", ""))
            df["dataset"] = dataset
            df["step"] = step
            all_metrics.append(df)

        metrics_df = pd.concat(all_metrics, ignore_index=True)

        # Create plot for this model
        plot_metrics_vs_step(metrics_df, wildcards.model, output[0])
