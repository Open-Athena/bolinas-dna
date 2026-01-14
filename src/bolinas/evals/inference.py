"""Inference utilities for computing variant scores using genomic language models."""

from pathlib import Path

import numpy as np
import pandas as pd
from biofoundation.data import Genome
from biofoundation.inference import run_llr_clm
from biofoundation.model.adapters.hf import HFCausalLM, HFTokenizer
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_llr_scores(
    checkpoint_path: str | Path,
    dataset: pd.DataFrame,
    genome_path: str | Path,
    context_size: int = 512,
    batch_size: int = 512,
    num_workers: int = 4,
    data_transform_on_the_fly: bool = True,
    torch_compile: bool = False,
) -> pd.DataFrame:
    """Compute log-likelihood ratio scores for genomic variants.

    Takes a dataset of genomic variants and computes LLR scores using a causal
    language model. Returns only scores, preserving input row order for alignment.

    Args:
        checkpoint_path: Path to model checkpoint directory.
        dataset: DataFrame with columns [chrom, pos, ref, alt, label] and optionally [subset].
        genome_path: Path to genome reference FASTA file.
        context_size: Context window size for model inference.
        batch_size: Number of sequences per batch during inference.
        num_workers: Number of workers for data loading.
        data_transform_on_the_fly: Whether to transform data on the fly during inference.
        torch_compile: Whether to use torch.compile for faster inference.

    Returns:
        DataFrame with columns [llr, minus_llr, abs_llr] only.
        Rows align with input dataset by index.
        - llr: Raw log-likelihood ratio
        - minus_llr: Negated LLR (higher = more deleterious)
        - abs_llr: Absolute LLR (higher = more impactful)
    """
    checkpoint_path = Path(checkpoint_path)
    genome_path = Path(genome_path)

    # Load genome reference
    genome = Genome(genome_path)

    # Load tokenizer and model
    tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(checkpoint_path))
    model = HFCausalLM(
        AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
    )

    # Convert pandas DataFrame to HuggingFace Dataset
    hf_dataset = Dataset.from_pandas(dataset, preserve_index=False)

    # Run LLR inference
    llr = run_llr_clm(
        model,
        tokenizer,
        hf_dataset,
        genome,
        context_size,
        data_transform_on_the_fly=data_transform_on_the_fly,
        inference_kwargs={
            "per_device_eval_batch_size": batch_size,
            "torch_compile": torch_compile,
            "bf16_full_eval": True,
            "dataloader_num_workers": num_workers,
            "remove_unused_columns": False,
        },
    )

    # Return only scores (assumes rows align with input dataset)
    scores = pd.DataFrame(
        {
            "llr": llr,
            "minus_llr": -llr,
            "abs_llr": np.abs(llr),
        }
    )

    return scores
