"""Compute LLR / abs(LLR) / embedding-distance scores per (model, dataset).

GPU-bound: meant to run on a SkyPilot GPU node, not the local CPU box.
"""


rule compute_scores:
    input:
        genome="results/genome.fa.gz",
        checkpoint="results/checkpoints/{model}",
    output:
        "results/scores/{model}/{dataset}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
    params:
        # 255 for BOS-using checkpoints (e.g. exp136), 256 for older runs;
        # the tokenizer baked into each checkpoint handles BOS itself.
        window_size=lambda wc: get_model_config(wc.model)["window_size"],
        hf_path=lambda wc: f"{config['input_hf_prefix']}_{wc.dataset}",
    threads: config["inference"]["num_workers"]
    run:
        ds = load_dataset(params.hf_path, split=config["split"]).to_pandas()
        for col in REQUIRED_VARIANT_COLUMNS:
            assert col in ds.columns, f"dataset missing column {col!r}"

        scores = compute_variant_scores(
            checkpoint_path=input.checkpoint,
            dataset=ds,
            genome_path=input.genome,
            context_size=params.window_size,
            batch_size=config["inference"]["batch_size"],
            num_workers=config["inference"]["num_workers"],
            data_transform_on_the_fly=config["inference"]["data_transform_on_the_fly"],
            torch_compile=config["inference"]["torch_compile"],
        )
        assert len(scores) == len(ds)

        # Preserve all variant columns (chrom, pos, ref, alt, label, subset,
        # match_group, …) alongside the score columns.
        out = pd.concat(
            [ds.reset_index(drop=True), scores.reset_index(drop=True)], axis=1
        )
        out.to_parquet(output[0], index=False)
        print(
            f"[evals_v2] {wildcards.model} {wildcards.dataset} "
            f"({config['split']}): n={len(out)}"
        )
