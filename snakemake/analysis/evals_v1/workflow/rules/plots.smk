"""Plotting rules for visualizing evaluation results."""


rule plot_metrics_vs_step:
    input:
        lambda wildcards: [
            f"results/metrics/{dataset}/{wildcards.model}/{step}.parquet"
            for dataset in get_all_datasets()
            for step in get_model_config(wildcards.model)["steps"]
        ],
    output:
        "results/plots/metrics_vs_step/{model}.svg",
    run:
        all_metrics = []
        for file_path in input:
            df = pd.read_parquet(file_path)
            # Path: results/metrics/{dataset}/{model}/{step}.parquet
            parts = file_path.split("/")
            df["dataset"] = parts[2]
            df["step"] = int(parts[4].replace(".parquet", ""))
            all_metrics.append(df)

        metrics_df = pd.concat(all_metrics, ignore_index=True)
        plot_metrics_vs_step(metrics_df, wildcards.model, output[0])


rule plot_models_comparison:
    input:
        [
            f"results/metrics/{dataset}/{model}/{step}.parquet"
            for dataset in get_all_datasets()
            for model, step in get_all_model_steps()
        ],
    output:
        "results/plots/models_comparison/{score_type}.svg",
    run:
        all_metrics = []
        for file_path in input:
            df = pd.read_parquet(file_path)
            # Path: results/metrics/{dataset}/{model}/{step}.parquet
            parts = file_path.split("/")
            df["dataset"] = parts[2]
            df["model"] = parts[3]
            df["step"] = int(parts[4].replace(".parquet", ""))
            all_metrics.append(df)

        metrics_df = pd.concat(all_metrics, ignore_index=True)
        plot_models_comparison(metrics_df, output[0], score_type=wildcards.score_type)


rule plot_custom_models_comparison:
    input:
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
        all_metrics = []
        for file_path in input:
            df = pd.read_parquet(file_path)
            # Path: results/metrics/{dataset}/{model}/{step}.parquet
            parts = file_path.split("/")
            df["dataset"] = parts[2]
            df["model"] = parts[3]
            df["step"] = int(parts[4].replace(".parquet", ""))
            all_metrics.append(df)

        metrics_df = pd.concat(all_metrics, ignore_index=True)

        plot_cfg = params.plot_config
        models_filter = plot_cfg.get("models", None)

        dataset_subset_score_map = None
        if "dataset_subsets" in plot_cfg and plot_cfg["dataset_subsets"] is not None:
            dataset_subset_score_map = {
                (ds["dataset"], ds["subset"]): ds["score_type"]
                for ds in plot_cfg["dataset_subsets"]
            }

        plot_models_comparison(
            metrics_df,
            output[0],
            dataset_subset_score_map=dataset_subset_score_map,
            models_filter=models_filter,
        )
