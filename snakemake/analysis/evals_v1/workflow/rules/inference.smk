"""Inference rules for computing LLR scores on evaluation datasets."""

from datasets import load_dataset

from bolinas.evals import compute_variant_scores


def get_dataset_config(dataset_name):
    """Get configuration for a specific dataset."""
    for dataset in config["datasets"]:
        if dataset["name"] == dataset_name:
            return dataset
    raise ValueError(f"Dataset {dataset_name} not found in config")


def get_model_config(model_name):
    """Get configuration for a specific model."""
    for model in config["models"]:
        if model["name"] == model_name:
            return model
    raise ValueError(f"Model {model_name} not found in config")


rule compute_scores:
    """Compute LLR scores for a dataset using a model checkpoint."""
    input:
        genome="results/genome.fa.gz",
    output:
        "results/scores/{dataset}/{model}/{step}.parquet",
    params:
        model_base_path=lambda wildcards: get_model_config(wildcards.model)["base_path"],
        model_context_size=lambda wildcards: get_model_config(wildcards.model)[
            "context_size"
        ],
        dataset_hf_path=lambda wildcards: get_dataset_config(wildcards.dataset)[
            "hf_path"
        ],
        dataset_split=lambda wildcards: get_dataset_config(wildcards.dataset)["split"],
        step=lambda wildcards: wildcards.step,
    threads: config["inference"]["num_workers"]
    run:
        # Get model checkpoint path
        checkpoint_path = f"{params.model_base_path}/step-{params.step}"

        # Load dataset directly from HuggingFace
        hf_dataset = load_dataset(params.dataset_hf_path, split=params.dataset_split)
        dataset = hf_dataset.to_pandas()

        # Compute variant scores (using config for performance settings)
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

        # Save results
        scores.to_parquet(output[0], index=False)
