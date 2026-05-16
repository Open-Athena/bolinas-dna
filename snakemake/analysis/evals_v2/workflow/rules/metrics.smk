"""Compute PairwiseAccuracy + binomial SE per (model, dataset).

One rule, fired per (model, dataset) — no cross-model aggregation, no markdown
rendering. The parquet is the deliverable.

We always compute the dataset's primary `score_column` (the LLR-protocol
column: `minus_llr` for mendelian, `abs_llr` for complex/eqtl). When the
scores parquet also contains `next_token_jsd_mean` (newer runs; all current
checkpoints do), we additionally compute pairwise PA on that column so the
dashboard's `JSD` protocol toggle has data without a second pipeline pass.
The output parquet has one block of `[subset × _global_ × _macro_avg_]`
rows per score_type, distinguishable by the `score_type` column.
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

        # Primary score column + JSD when present (no-op for legacy scores
        # parquets without the JSD column).
        score_cols = [score_col]
        if "next_token_jsd_mean" in df.columns:
            score_cols.append("next_token_jsd_mean")
        for c in score_cols:
            assert c in df.columns, f"scores parquet missing score column {c!r}"

        metrics = compute_pairwise_metrics(
            dataset=df[list(REQUIRED_VARIANT_COLUMNS)],
            scores=df[score_cols],
            score_columns=score_cols,
        )
        metrics["model"] = wildcards.model
        metrics["dataset"] = wildcards.dataset
        metrics["split"] = config["split"]
        metrics.to_parquet(output[0], index=False)
        print(
            f"[evals_v2] {wildcards.model} {wildcards.dataset}: "
            f"{len(metrics)} subset rows (score_cols={score_cols})"
        )
