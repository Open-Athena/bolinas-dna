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
        metrics=lambda wildcards: [
            f"results/metrics/{dataset}/{model_id}/{step}.parquet"
            for dataset in {
                ds["dataset"]
                for ds in get_custom_plot_config(wildcards.plot_name).get(
                    "dataset_subsets", []
                )
            }
            for model_entry in get_custom_plot_config(wildcards.plot_name).get(
                "models", []
            )
            for model_id in [get_custom_plot_model_id(model_entry)]
            for step in get_model_config(model_id)["steps"]
        ],
        baselines=lambda wildcards: get_baseline_input_files(wildcards.plot_name),
    output:
        "results/plots/custom_comparison/{plot_name}.svg",
    params:
        plot_config=lambda wildcards: get_custom_plot_config(wildcards.plot_name),
    run:
        all_metrics = []
        for file_path in input.metrics:
            df = pd.read_parquet(file_path)
            # Path: results/metrics/{dataset}/{model}/{step}.parquet
            parts = file_path.split("/")
            df["dataset"] = parts[2]
            df["model"] = parts[3]
            df["step"] = int(parts[4].replace(".parquet", ""))
            all_metrics.append(df)

        metrics_df = pd.concat(all_metrics, ignore_index=True)

        plot_cfg = params.plot_config

        # Build models_filter and model_labels from config
        model_entries = plot_cfg.get("models", [])
        models_filter = (
            [get_custom_plot_model_id(m) for m in model_entries]
            if model_entries
            else None
        )
        model_labels = (
            {
                get_custom_plot_model_id(m): get_custom_plot_model_title(m)
                for m in model_entries
            }
            if model_entries
            else None
        )
        model_colors = (
            {
                get_custom_plot_model_id(m): m.get("color")
                for m in model_entries
                if m.get("color")
            }
            if model_entries
            else None
        )
        if model_colors is not None and not model_colors:
            model_colors = None

        dataset_subset_score_map = None
        if "dataset_subsets" in plot_cfg and plot_cfg["dataset_subsets"] is not None:
            dataset_subset_score_map = {
                (ds["dataset"], ds["subset"]): ds["score_type"]
                for ds in plot_cfg["dataset_subsets"]
            }

        baseline_data = load_baseline_data(plot_cfg)

        # Build subplot titles map from config
        subplot_titles = None
        if "dataset_subsets" in plot_cfg:
            subplot_titles = {
                (ds["dataset"], ds["subset"]): ds.get("title")
                for ds in plot_cfg["dataset_subsets"]
                if ds.get("title")
            }
            if not subplot_titles:
                subplot_titles = None

        ylim = plot_cfg.get("ylim")
        if ylim is not None:
            ylim = tuple(ylim)

        plot_models_comparison(
            metrics_df,
            output[0],
            dataset_subset_score_map=dataset_subset_score_map,
            models_filter=models_filter,
            baseline_data=baseline_data,
            title=plot_cfg.get("title", plot_cfg["name"]),
            subplot_titles=subplot_titles,
            n_cols=plot_cfg.get("n_cols"),
            model_labels=model_labels,
            model_colors=model_colors,
            ylim=ylim,
            scale=plot_cfg.get("scale", 1.0),
        )
