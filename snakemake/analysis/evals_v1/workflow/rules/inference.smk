"""Inference rules for computing LLR scores on evaluation datasets."""


def _checkpoint_input(wildcards):
    """Local download dir for `gcs_path` entries; `[]` (no input) for `base_path` entries."""
    cfg = get_model_config(wildcards.model)
    if "gcs_path" in cfg:
        return f"results/checkpoints/{wildcards.model}/step-{wildcards.step}"
    return []


rule compute_scores:
    input:
        genome="results/genome.fa.gz",
        checkpoint=_checkpoint_input,
    output:
        "results/scores/{dataset}/{model}/{step}.parquet",
    params:
        model_context_size=lambda wildcards: get_model_config(wildcards.model)[
            "context_size"
        ],
        dataset_hf_path=lambda wildcards: get_dataset_config(wildcards.dataset)[
            "hf_path"
        ],
        dataset_split=lambda wildcards: get_dataset_config(wildcards.dataset)["split"],
    threads: config["inference"]["num_workers"]
    run:
        cfg = get_model_config(wildcards.model)
        if "gcs_path" in cfg:
            checkpoint_path = input.checkpoint
        else:
            checkpoint_path = f"{cfg['base_path']}/step-{wildcards.step}"

        hf_dataset = load_dataset(params.dataset_hf_path, split=params.dataset_split)
        dataset = hf_dataset.to_pandas()

        scores = compute_variant_scores(
            checkpoint_path=checkpoint_path,
            dataset=dataset,
            genome_path=input.genome,
            context_size=params.model_context_size,
            batch_size=config["inference"]["batch_size"],
            num_workers=config["inference"]["num_workers"],
            data_transform_on_the_fly=config["inference"]["data_transform_on_the_fly"],
            torch_compile=config["inference"]["torch_compile"],
        )

        scores.to_parquet(output[0], index=False)
