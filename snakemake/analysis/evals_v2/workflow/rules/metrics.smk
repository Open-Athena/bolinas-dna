"""Compute PairwiseAccuracy + binomial SE per (model, dataset).

One rule, fired per (model, dataset) — no cross-model aggregation, no markdown
rendering. The parquet is the deliverable.
"""


rule compute_metrics:
    input:
        "results/scores/{model}/{dataset}.parquet",
    output:
        "results/metrics/{model}/{dataset}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
    run:
        score_col = get_dataset_config(wildcards.dataset)["score_column"]
        df = pd.read_parquet(input[0])
        for col in REQUIRED_VARIANT_COLUMNS:
            assert col in df.columns, f"scores parquet missing column {col!r}"
        assert score_col in df.columns, (
            f"scores parquet missing score column {score_col!r}"
        )

        metrics = compute_pairwise_metrics(
            dataset=df[list(REQUIRED_VARIANT_COLUMNS)],
            scores=df[[score_col]],
            score_columns=[score_col],
        )
        metrics["model"] = wildcards.model
        metrics["dataset"] = wildcards.dataset
        metrics["split"] = config["split"]
        metrics.to_parquet(output[0], index=False)
        print(
            f"[evals_v2] {wildcards.model} {wildcards.dataset}: "
            f"{len(metrics)} subset rows"
        )
