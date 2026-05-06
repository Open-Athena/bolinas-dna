"""PairwiseAccuracy + binomial SE per (dataset)."""


rule compute_metrics:
    input:
        "results/scores/{dataset}.parquet",
    output:
        "results/metrics/{dataset}.parquet",
    wildcard_constraints:
        dataset="|".join(DATASETS),
    run:
        df = pd.read_parquet(input[0])
        for col in REQUIRED_VARIANT_COLUMNS:
            assert col in df.columns, f"scores parquet missing column {col!r}"
        assert (
            SCORE_COLUMN in df.columns
        ), f"scores parquet missing score column {SCORE_COLUMN!r}"

        metrics = compute_pairwise_metrics(
            dataset=df[list(REQUIRED_VARIANT_COLUMNS)],
            scores=df[[SCORE_COLUMN]],
            score_columns=[SCORE_COLUMN],
        )
        metrics["dataset"] = wildcards.dataset
        metrics["split"] = SPLIT
        metrics.to_parquet(output[0], index=False)
        print(f"[alphagenome_eval] {wildcards.dataset}: " f"{len(metrics)} subset rows")
