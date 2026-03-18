rule train_model:
    input:
        train=lambda wc: f"results/dataset/{config['models'][wc.model]['dataset']}/train.parquet",
        val=lambda wc: f"results/dataset/{config['models'][wc.model]['dataset']}/validation.parquet",
    output:
        ckpt="results/model/{model}/best.ckpt",
        metrics="results/model/{model}/metrics.json",
    params:
        weights_path=config["alphagenome_weights_path"],
        freeze_flag=lambda wc: (
            "--freeze-backbone"
            if config["models"][wc.model]["freeze_backbone"]
            else "--no-freeze-backbone"
        ),
        learning_rate=lambda wc: config["models"][wc.model]["learning_rate"],
        batch_size=lambda wc: config["models"][wc.model]["batch_size"],
        max_epochs=lambda wc: config["models"][wc.model]["max_epochs"],
        output_dir=lambda wc: f"results/model/{wc.model}",
    shell:
        """
        uv run python -m bolinas.enhancer_classification.train \
            --train-parquet {input.train} \
            --val-parquet {input.val} \
            --weights-path {params.weights_path} \
            --output-dir {params.output_dir} \
            --learning-rate {params.learning_rate} \
            --batch-size {params.batch_size} \
            --max-epochs {params.max_epochs} \
            {params.freeze_flag} \
            --seed {config[seed]}
        """
