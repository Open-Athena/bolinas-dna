"""Stage 2 (CPU): post-process npz cache → parquet of variant rows + score columns.

Cheap (pure pandas/numpy). Re-running with new scoring rules only needs to
re-run this stage (and stages 3 / 4), not the GPU extract_features stage.
"""


rule compute_scores:
    input:
        cache="results/cache/{model}__win{window}__{dataset}.npz",
    output:
        "results/scores/{model}__win{window}__{dataset}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
        window=r"[0-9]+",
    params:
        hf_path=lambda wc: f"{config['input_hf_prefix']}_{wc.dataset}",
    run:
        ds = load_dataset(params.hf_path, split=config["split"]).to_pandas()
        score_df = score_cache(input.cache)

        # The cache is row-aligned with the input dataset; defensive shape check.
        assert len(score_df) == len(ds), (
            f"cache ({len(score_df)}) and dataset ({len(ds)}) row counts differ"
        )

        out = pd.concat(
            [ds.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1
        )
        # No-NaN invariant per pairwise_accuracy's contract.
        for col in score_df.columns:
            assert not out[col].isna().any(), (
                f"score column {col} has NaN — investigate features cache"
            )
        out.to_parquet(output[0], index=False)
        print(
            f"[zeroshot_vep/scores] {wildcards.model} win={wildcards.window} "
            f"{wildcards.dataset}: {len(out)} rows × {len(score_df.columns)} scores",
            flush=True,
        )
