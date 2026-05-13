"""Compute supervised-VEP feature cache per (model, dataset).

GPU-bound: meant to run on a SkyPilot GPU node, not the local CPU box.

Each output parquet has one row per variant (input row order preserved) and
the columns:

* variant metadata: ``chrom, pos, ref, alt, label, subset, match_group, ...``
* zero-shot scalars: ``llr, minus_llr, abs_llr, embed_last_l2``
* dense feature blocks (parquet list columns, D-dim each):
  ``mean_ref, mean_alt, traitgym_innerprod``
"""


rule compute_features:
    input:
        genome="results/genome.fa.gz",
    output:
        "results/features/{model}/{dataset}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
    params:
        window_size=lambda wc: get_model_config(wc.model)["window_size"],
        hf_repo=lambda wc: get_model_config(wc.model)["hf_repo"],
        hf_path=lambda wc: f"{config['input_hf_prefix']}_{wc.dataset}",
    threads: config["inference"]["num_workers"]
    run:
        ds = load_dataset(params.hf_path, split=config["split"]).to_pandas()
        for col in REQUIRED_VARIANT_COLUMNS:
            assert col in ds.columns, f"dataset missing column {col!r}"

        features = compute_pooled_features(
            checkpoint_path=params.hf_repo,
            dataset=ds,
            genome_path=input.genome,
            context_size=params.window_size,
            batch_size=config["inference"]["batch_size"],
            num_workers=config["inference"]["num_workers"],
            data_transform_on_the_fly=config["inference"]["data_transform_on_the_fly"],
            torch_compile=config["inference"]["torch_compile"],
        )
        assert len(features) == len(ds)

        out = pd.concat(
            [ds.reset_index(drop=True), features.reset_index(drop=True)], axis=1
        )
        out.to_parquet(output[0], index=False)
        n_dense = len(out["mean_ref"].iloc[0])
        print(
            f"[supervised_vep] {wildcards.model} {wildcards.dataset} "
            f"({config['split']}): n={len(out)} D={n_dense}"
        )
