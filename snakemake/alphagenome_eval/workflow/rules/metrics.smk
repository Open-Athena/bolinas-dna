"""PairwiseAccuracy + binomial SE per (dataset)."""


rule compute_metrics:
    input:
        "results/scores/{dataset}.parquet",
    output:
        "results/metrics/{dataset}.parquet",
    wildcard_constraints:
        dataset="|".join(DATASETS),
    run:
        score_col = config["score_column"]
        df = pd.read_parquet(input[0])
        for col in REQUIRED_VARIANT_COLUMNS:
            assert col in df.columns, f"scores parquet missing column {col!r}"
        assert (
            score_col in df.columns
        ), f"scores parquet missing score column {score_col!r}"

        metrics = compute_pairwise_metrics(
            dataset=df[list(REQUIRED_VARIANT_COLUMNS)],
            scores=df[[score_col]],
            score_columns=[score_col],
        )
        metrics["dataset"] = wildcards.dataset
        metrics["split"] = config["split"]
        metrics.to_parquet(output[0], index=False)
        print(f"[alphagenome_eval] {wildcards.dataset}: {len(metrics)} subset rows")
