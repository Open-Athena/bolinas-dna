"""LoRA fine-tuning rules (iter-2 of #180).

Per (dataset, fold) → train LoRA adapters on the (n_splits-1) chrom-train
folds and predict on the held-out fold. Concatenate per-fold predictions
into an OOF leaderboard the same way iter-1 does it.

GPU-bound: run on the SkyPilot GPU cluster via ``sky/run_lora.yaml``.

Scope follows the iter-2 plan: complex_traits first (smallest, fastest),
then eqtl, then mendelian.
"""


rule lora_fit_predict_one_fold:
    """Train LoRA on (n_splits-1) chrom-folds, predict on the held-out fold."""
    input:
        genome="results/genome.fa.gz",
    output:
        predictions="results/lora_predictions/{model}/{dataset}/fold_{fold}.parquet",
        stats="results/lora_stats/{model}/{dataset}/fold_{fold}.json",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
        fold=r"\d+",
    params:
        hf_repo=lambda wc: get_model_config(wc.model)["hf_repo"],
        window_size=lambda wc: get_model_config(wc.model)["window_size"],
        hf_path=lambda wc: f"{config['input_hf_prefix']}_{wc.dataset}",
        split=config["split"],
        n_splits=config["cv"]["n_splits"],
        lora_rank=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        lora_target_modules=config["lora"]["target_modules"],
        epochs=config["lora"]["epochs"],
        lr=config["lora"]["lr"],
        batch_size=config["lora"]["batch_size"],
        margin=config["lora"]["margin"],
    threads: 4
    run:
        import json

        from bolinas.supervised.lora_pipeline import fit_predict_one_fold

        preds, stats = fit_predict_one_fold(
            hf_dataset_path=params.hf_path,
            split=params.split,
            backbone_id=params.hf_repo,
            window_size=params.window_size,
            genome_path=input.genome,
            fold=int(wildcards.fold),
            n_splits=params.n_splits,
            lora_rank=params.lora_rank,
            lora_alpha=params.lora_alpha,
            lora_dropout=params.lora_dropout,
            lora_target_modules=tuple(params.lora_target_modules),
            epochs=params.epochs,
            lr=params.lr,
            batch_size=params.batch_size,
            margin=params.margin,
        )
        preds.to_parquet(output.predictions, index=False)
        with open(output.stats, "w") as f:
            json.dump(stats, f, indent=2, default=str)


rule lora_aggregate_oof:
    """Concatenate per-fold predictions into one OOF prediction parquet per (model, dataset)."""
    input:
        folds=lambda wc: [
            f"results/lora_predictions/{wc.model}/{wc.dataset}/fold_{i}.parquet"
            for i in range(config["cv"]["n_splits"])
        ],
    output:
        oof="results/lora_oof/{model}/{dataset}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
    threads: 1
    run:
        parts = [pd.read_parquet(p) for p in input.folds]
        oof = pd.concat(parts, ignore_index=True)
        # Confirm coverage = full train split row count.
        ds = load_dataset(
            f"{config['input_hf_prefix']}_{wildcards.dataset}",
            split=config["split"],
        )
        assert len(oof) == len(ds), (
            f"OOF row count {len(oof)} != dataset row count {len(ds)} for "
            f"{wildcards.model}×{wildcards.dataset}"
        )
        oof.to_parquet(output.oof, index=False)
        print(
            f"[lora_aggregate_oof] {wildcards.model} {wildcards.dataset}: "
            f"n={len(oof)} folds={oof['fold'].nunique()}"
        )


rule lora_compute_metrics:
    """Run OOF predictions through compute_pairwise_metrics for direct comparison to iter-1."""
    input:
        oof="results/lora_oof/{model}/{dataset}.parquet",
    output:
        metrics="results/lora_metrics/{model}/{dataset}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
    threads: 1
    run:
        oof = pd.read_parquet(input.oof)
        metrics = compute_metrics_from_oof(oof, score_column="score")
        metrics["model"] = wildcards.model
        metrics["dataset"] = wildcards.dataset
        metrics["recipe"] = "lora_pairwise_l2"
        metrics["classifier"] = "lora"
        metrics["family"] = "lora"
        metrics.to_parquet(output.metrics, index=False)
