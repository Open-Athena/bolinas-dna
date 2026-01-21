"""Baseline model evaluation rules."""


rule compute_baseline_scores:
    output:
        "results/baselines/scores/{dataset}/{baseline}.parquet",
    params:
        dataset_hf_path=lambda wildcards: get_dataset_config(wildcards.dataset)[
            "hf_path"
        ],
        dataset_split=lambda wildcards: get_dataset_config(wildcards.dataset)["split"],
        original_dataset_path=lambda wildcards: get_original_dataset_path(
            wildcards.dataset
        ),
    run:
        our_dataset = load_dataset(
            params.dataset_hf_path, split=params.dataset_split
        ).to_pandas()

        original_coords = pd.read_parquet(
            f"hf://datasets/{params.original_dataset_path}/test.parquet",
            columns=COORDINATES,
        )

        baseline_scores = pd.read_parquet(
            f"hf://datasets/{params.original_dataset_path}/predictions/{wildcards.baseline}.parquet"
        )
        assert original_coords.shape[0] == baseline_scores.shape[0]

        # Negate score so higher = more deleterious
        original_coords["score"] = -baseline_scores.iloc[:, 0]

        merged = our_dataset.merge(original_coords, on=COORDINATES, how="left")
        assert (
            merged["score"].isna().sum() == 0
        ), "Some variants not found in baseline predictions"

        merged[["score"]].to_parquet(output[0], index=False)


rule compute_baseline_metrics:
    input:
        scores="results/baselines/scores/{dataset}/{baseline}.parquet",
    output:
        "results/baselines/metrics/{dataset}/{baseline}.parquet",
    params:
        dataset_hf_path=lambda wildcards: get_dataset_config(wildcards.dataset)[
            "hf_path"
        ],
        dataset_split=lambda wildcards: get_dataset_config(wildcards.dataset)["split"],
        metrics=lambda wildcards: get_dataset_config(wildcards.dataset)["metrics"],
    run:
        dataset = load_dataset(
            params.dataset_hf_path, split=params.dataset_split
        ).to_pandas()

        scores = pd.read_parquet(input.scores)
        dataset["score"] = scores["score"].values

        results = []

        for metric_name in params.metrics:
            if metric_name == "AUPRC":
                value = average_precision_score(dataset["label"], dataset["score"])
            elif metric_name == "AUROC":
                value = roc_auc_score(dataset["label"], dataset["score"])
            else:
                raise ValueError(f"Unknown metric: {metric_name}")
            results.append({"metric": metric_name, "subset": "global", "value": value})

        if "subset" in dataset.columns:
            for subset_name in dataset["subset"].unique():
                subset_data = dataset[dataset["subset"] == subset_name]
                for metric_name in params.metrics:
                    if metric_name == "AUPRC":
                        value = average_precision_score(
                            subset_data["label"], subset_data["score"]
                        )
                    elif metric_name == "AUROC":
                        value = roc_auc_score(
                            subset_data["label"], subset_data["score"]
                        )
                    else:
                        raise ValueError(f"Unknown metric: {metric_name}")
                    results.append(
                        {"metric": metric_name, "subset": subset_name, "value": value}
                    )

        pd.DataFrame(results).to_parquet(output[0], index=False)
