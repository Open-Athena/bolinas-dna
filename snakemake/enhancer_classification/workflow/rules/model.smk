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
    input:
        train="results/dataset/{dataset}/train.parquet",
        val="results/dataset/{dataset}/validation.parquet",
        weights=f"results/weights/{config['alphagenome_weights']['filename']}",
    output:
        ckpt="results/model/{model}/{dataset}/best.ckpt",
        metrics="results/model/{model}/{dataset}/metrics.json",
    params:
        freeze_flag=lambda wc: (
            "--freeze-backbone"
            if get_model_config(wc.model)["freeze_backbone"]
            else "--no-freeze-backbone"
        ),
        learning_rate=lambda wc: get_model_config(wc.model)["learning_rate"],
        weight_decay=lambda wc: get_model_config(wc.model)["weight_decay"],
        gradient_clip_val=lambda wc: get_model_config(wc.model)["gradient_clip_val"],
        batch_size=lambda wc: get_model_config(wc.model)["batch_size"],
        max_epochs=lambda wc: get_model_config(wc.model)["max_epochs"],
        overfit_batches=lambda wc: get_model_config(wc.model).get("overfit_batches", 0),
    shell:
        """
        uv run python -m bolinas.enhancer_classification.train \
            --train-parquet {input.train} \
            --val-parquet {input.val} \
            --weights-path {input.weights} \
            --output-ckpt {output.ckpt} \
            --output-metrics {output.metrics} \
            --learning-rate {params.learning_rate} \
            --weight-decay {params.weight_decay} \
            --batch-size {params.batch_size} \
            --max-epochs {params.max_epochs} \
            --overfit-batches {params.overfit_batches} \
            --gradient-clip-val {params.gradient_clip_val} \
            {params.freeze_flag} \
            --seed {config[seed]} \
            --wandb-run {wildcards.model}-{wildcards.dataset}
        """
