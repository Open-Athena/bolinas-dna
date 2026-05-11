"""Stage 1 (GPU): 4-pass forward inference → npz feature cache.

One cache file per (model, window, dataset). The cache contains:
- ``seq_logprob`` (N, 4) — joint seq log-prob under each candidate nucleotide.
- ``pos_logprob`` (N, 4, T-1) — per-position log-prob (cheap; for future scoring).
- ``emb_{ref,alt}_{last,middle}`` (N, T, D) fp16 — per-position embeddings.
- ``ref_idx`` / ``alt_idx`` (N,) — index in {A,C,G,T}.
- ``var_pos``, ``window_size``, ``n_prefix``, ``n_suffix`` — scalars.
- ``row_idx`` — alignment back to the source HF dataset row.
"""


# Constrain wildcard `window` to integers so it doesn't accidentally swallow
# parts of `model` or `dataset` (since `__` is the delimiter in the filename).
WINDOW_REGEX = r"[0-9]+"


rule extract_features:
    input:
        genome="results/genome.fa.gz",
        checkpoint="results/checkpoints/{model}",
    output:
        "results/cache/{model}__win{window}__{dataset}.npz",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
        window=WINDOW_REGEX,
    params:
        hf_path=lambda wc: f"{config['input_hf_prefix']}_{wc.dataset}",
        batch_size=config["inference"]["batch_size"],
        dtype=config["inference"]["dtype"],
        store_pos_logprob=config["inference"]["store_pos_logprob"],
    threads: 4
    resources:
        gpu=1,
    run:
        import torch
        from bolinas.zeroshot_vep.features import extract_features, write_cache

        # Load dataset, train split only (per project convention).
        ds = load_dataset(params.hf_path, split=config["split"]).to_pandas()
        for col in REQUIRED_VARIANT_COLUMNS:
            assert col in ds.columns, f"dataset {wildcards.dataset!r} missing {col!r}"

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map[params.dtype]

        cache = extract_features(
            checkpoint_path=input.checkpoint,
            dataset=ds,
            genome_path=input.genome,
            window_size=int(wildcards.window),
            batch_size=int(params.batch_size),
            dtype=torch_dtype,
            store_pos_logprob=bool(params.store_pos_logprob),
        )
        write_cache(output[0], cache)

        print(
            f"[zeroshot_vep/features] {wildcards.model} win={wildcards.window} "
            f"{wildcards.dataset}: N={len(ds)} cache→{output[0]}",
            flush=True,
        )
