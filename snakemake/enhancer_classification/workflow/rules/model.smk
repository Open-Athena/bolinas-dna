WEIGHTS_FILE = f"results/weights/{config['alphagenome_weights']['filename']}"


rule download_alphagenome_weights:
    output:
        "results/weights/{filename}",
    params:
        repo=config["alphagenome_weights"]["repo"],
    run:
        from pathlib import Path

        out = Path(output[0])
        hf_hub_download(params.repo, wildcards.filename, local_dir=str(out.parent))


rule train_model:
    threads: workflow.cores
    input:
        train="results/dataset/{dataset}/train.parquet",
        val="results/dataset/{dataset}/validation.parquet",
        weights=lambda wc: [] if get_model_config(wc.model).get("random_init") else WEIGHTS_FILE,
    output:
        ckpt="results/model/{model}/{dataset}/best.ckpt",
        metrics="results/model/{model}/{dataset}/metrics.json",
    params:
        freeze_flag=lambda wc: (
            "--freeze-backbone"
            if get_model_config(wc.model)["freeze_backbone"]
            else "--no-freeze-backbone"
        ),
        weights_flag=lambda wc, input: (
            f"--weights-path {input.weights}" if input.weights else ""
        ),
        learning_rate=lambda wc: get_model_config(wc.model)["learning_rate"],
        weight_decay=lambda wc: get_model_config(wc.model)["weight_decay"],
        gradient_clip_val=lambda wc: get_model_config(wc.model)["gradient_clip_val"],
        batch_size=lambda wc: get_model_config(wc.model)["batch_size"],
        max_epochs=lambda wc: get_model_config(wc.model)["max_epochs"],
        overfit_batches=lambda wc: get_model_config(wc.model).get("overfit_batches", 0),
        warmup_steps=lambda wc: get_model_config(wc.model)["warmup_steps"],
        early_stopping_patience=lambda wc: get_model_config(wc.model)["early_stopping_patience"],
    shell:
        """
        uv run python -m bolinas.enhancer_classification.train \
            --train-parquet {input.train} \
            --val-parquet {input.val} \
            {params.weights_flag} \
            --output-ckpt {output.ckpt} \
            --output-metrics {output.metrics} \
            --learning-rate {params.learning_rate} \
            --weight-decay {params.weight_decay} \
            --batch-size {params.batch_size} \
            --max-epochs {params.max_epochs} \
            --overfit-batches {params.overfit_batches} \
            --gradient-clip-val {params.gradient_clip_val} \
            --warmup-steps {params.warmup_steps} \
            --early-stopping-patience {params.early_stopping_patience} \
            {params.freeze_flag} \
            --seed {config[seed]} \
            --num-workers {threads} \
            --wandb-run {wildcards.model}-{wildcards.dataset}
        """
