"""Metrics computation and aggregation rules."""


rule compute_metrics:
    input:
        scores="results/scores/{dataset}/{model}/{step}.parquet",
    output:
        "results/metrics/{dataset}/{model}/{step}.parquet",
    params:
        dataset_config=lambda wildcards: get_dataset_config(wildcards.dataset),
    run:
        hf_dataset = load_dataset(
            params.dataset_config["hf_path"], split=params.dataset_config["split"]
        )
        dataset = hf_dataset.to_pandas()

        scores = pd.read_parquet(input.scores)

        metrics = compute_metrics(
            dataset=dataset,
            scores=scores,
            metrics=params.dataset_config["metrics"],
            score_columns=config["score_types"],
        )

        metrics.to_parquet(output[0], index=False)
