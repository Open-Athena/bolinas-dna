"""Inference utilities for computing variant scores using genomic language models."""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bolinas.data.genome import Genome
from bolinas.model.runner import run_variant_score_bundle


def compute_variant_scores(
    checkpoint_path: str | Path,
    dataset: pd.DataFrame,
    genome_path: str | Path,
    context_size: int = 512,
    batch_size: int = 512,
    num_workers: int = 4,
    data_transform_on_the_fly: bool = True,
    torch_compile: bool = False,
    rc_avg: bool = False,
) -> pd.DataFrame:
    """Compute variant scores from a CLM: LLR + embedding distances + next-token JSD.

    Takes a dataset of genomic variants and computes the full score bundle
    using a causal language model. Returns only scores, preserving input
    row order for alignment.

    Args:
        checkpoint_path: Path to model checkpoint directory.
        dataset: DataFrame with columns [chrom, pos, ref, alt, label] and optionally [subset].
        genome_path: Path to genome reference FASTA. May be a local filesystem
            path or an fsspec URI (e.g. ``s3://bucket/genome.fa.gz``); the
            latter requires the ``genome-s3`` dependency group.
        context_size: Context window size for model inference.
        batch_size: Number of sequences per batch during inference.
        num_workers: Number of workers for data loading.
        data_transform_on_the_fly: Whether to transform data on the fly during inference.
        torch_compile: Whether to use torch.compile for faster inference.
        rc_avg: If True, also score the reverse-complemented window for each
            variant and return the element-wise FWD/RC average. Doubles
            inference cost. See ``run_variant_score_bundle`` for the
            semantics on the JSD column.

    Returns:
        DataFrame with columns [llr, minus_llr, abs_llr, next_token_jsd_mean].
        Rows align with input dataset by index.

        - llr: Raw log-likelihood ratio
        - minus_llr: Negated LLR (higher = more deleterious)
        - abs_llr: Absolute LLR (higher = more impactful)
        - next_token_jsd_mean: mean per-position 4-nuc JSD over downstream
          positions (called ``down_jsd_mean`` in issue #175)
    """
    checkpoint_path = Path(checkpoint_path)
    # Don't Path()-cast genome_path: would break s3:// URIs (POSIX path
    # normalization collapses // to /). Genome accepts str | Path and
    # detects the remote scheme itself.
    genome = Genome(genome_path)
    # AutoTokenizer / AutoModelForCausalLM satisfy the duck-typed interface
    # bolinas.model.runner expects — no adapter wrappers needed.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    hf_dataset = Dataset.from_pandas(dataset, preserve_index=False)

    results = run_variant_score_bundle(
        model,
        tokenizer,
        hf_dataset,
        genome,
        context_size,
        rc_avg=rc_avg,
        data_transform_on_the_fly=data_transform_on_the_fly,
        inference_kwargs={
            "per_device_eval_batch_size": batch_size,
            "torch_compile": torch_compile,
            "bf16_full_eval": True,
            "dataloader_num_workers": num_workers,
            "remove_unused_columns": False,
        },
    )

    llr = results[:, 0]
    next_token_jsd_mean = results[:, 1]

    scores = pd.DataFrame(
        {
            "llr": llr,
            "minus_llr": -llr,
            "abs_llr": np.abs(llr),
            "next_token_jsd_mean": next_token_jsd_mean,
        }
    )

    return scores
