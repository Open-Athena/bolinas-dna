"""Stage 1 (GPU): 4-pass forward inference + on-the-fly scoring → per-variant parquet.

Embeddings are scored in-batch and discarded — we do NOT cache them, because
storing the (N, T, D) fp16 tensors for the largest configs (mendelian win=512
≈ 41 GB per cache, 1.8 TB across all 45 caches) would dominate S3 storage
and upload time. The output is a small parquet with variant metadata + 30
score columns (~MB per config). To try a NEW scoring rule, re-run this rule
(GPU expense) — the rank-based combinations slated for iteration 2 don't
need new forward passes.
"""


# Constrain wildcard `window` to integers so it doesn't accidentally swallow
# parts of `model` or `dataset` (since `__` is the delimiter in the filename).
WINDOW_REGEX = r"[0-9]+"


rule extract_features:
    input:
        genome="results/genome.fa.gz",
        checkpoint="results/checkpoints/{model}",
    output:
        "results/scores/{model}__win{window}__{dataset}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
        window=WINDOW_REGEX,
    params:
        hf_path=lambda wc: f"{config['input_hf_prefix']}_{wc.dataset}",
        batch_size=config["inference"]["batch_size"],
        dtype=config["inference"]["dtype"],
    threads: 4
    resources:
        gpu=1,
    run:
        import torch
        from bolinas.zeroshot_vep.features import extract_features_and_score

        ds = load_dataset(params.hf_path, split=config["split"]).to_pandas()
        for col in REQUIRED_VARIANT_COLUMNS:
            assert col in ds.columns, f"dataset {wildcards.dataset!r} missing {col!r}"

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map[params.dtype]

        scores_df = extract_features_and_score(
            checkpoint_path=input.checkpoint,
            dataset=ds,
            genome_path=input.genome,
            window_size=int(wildcards.window),
            batch_size=int(params.batch_size),
            dtype=torch_dtype,
        )
        assert len(scores_df) == len(ds), (
            f"score rows ({len(scores_df)}) and dataset rows ({len(ds)}) differ"
        )

        # Preserve variant metadata alongside score columns.
        out = pd.concat(
            [ds.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1
        )
        # snakemake's s3-storage local-copy path may not exist yet — mkdir.
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output[0], index=False)

        print(
            f"[zeroshot_vep/features] {wildcards.model} win={wildcards.window} "
            f"{wildcards.dataset}: {len(out)} rows × {scores_df.shape[1]} scores → {output[0]}",
            flush=True,
        )
