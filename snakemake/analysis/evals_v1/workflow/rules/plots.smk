"""Plotting rules for visualizing evaluation results."""

import pandas as pd

from bolinas.evals.plotting import plot_metrics_vs_step, plot_models_comparison


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


rule plot_models_comparison:
    """Plot metrics vs training step comparing all models for a specific score type."""
    input:
        # All metrics for all models across all datasets and steps
        [
            f"results/metrics/{dataset}/{model}/{step}.parquet"
            for dataset in get_all_datasets()
            for model, step in get_all_model_steps()
        ],
    output:
        "results/plots/models_comparison/{score_type}.svg",
    run:
        # Load and aggregate all metrics for all models
        all_metrics = []
        for file_path in input:
            df = pd.read_parquet(file_path)
            # Extract dataset, model, and step from path: results/metrics/{dataset}/{model}/{step}.parquet
            parts = file_path.split("/")
            dataset = parts[2]
            model = parts[3]
            step = int(parts[4].replace(".parquet", ""))
            df["dataset"] = dataset
            df["model"] = model
            df["step"] = step
            all_metrics.append(df)

        metrics_df = pd.concat(all_metrics, ignore_index=True)

        # Create plot comparing all models for this score type
        plot_models_comparison(metrics_df, output[0], score_type=wildcards.score_type)


rule plot_custom_models_comparison:
    """Plot filtered models comparison based on config specifications."""
    input:
        # All metrics for all models across all datasets and steps
        [
            f"results/metrics/{dataset}/{model}/{step}.parquet"
            for dataset in get_all_datasets()
            for model, step in get_all_model_steps()
        ],
    output:
        "results/plots/custom_comparison/{plot_name}.svg",
    params:
        plot_config=lambda wildcards: get_custom_plot_config(wildcards.plot_name),
    run:
        # Load and aggregate all metrics for all models
        all_metrics = []
        for file_path in input:
            df = pd.read_parquet(file_path)
            # Extract dataset, model, and step from path: results/metrics/{dataset}/{model}/{step}.parquet
            parts = file_path.split("/")
            dataset = parts[2]
            model = parts[3]
            step = int(parts[4].replace(".parquet", ""))
            df["dataset"] = dataset
            df["model"] = model
            df["step"] = step
            all_metrics.append(df)

        metrics_df = pd.concat(all_metrics, ignore_index=True)

        # Extract filter parameters from config
        plot_cfg = params.plot_config
        models_filter = plot_cfg.get("models", None)

        # Build dataset_subset_score_map from config
        dataset_subset_score_map = None
        if "dataset_subsets" in plot_cfg and plot_cfg["dataset_subsets"] is not None:
            dataset_subset_score_map = {
                (ds["dataset"], ds["subset"]): ds["score_type"]
                for ds in plot_cfg["dataset_subsets"]
            }

            # Create filtered plot
        plot_models_comparison(
            metrics_df,
            output[0],
            dataset_subset_score_map=dataset_subset_score_map,
            models_filter=models_filter,
        )
