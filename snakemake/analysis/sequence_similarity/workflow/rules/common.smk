"""Common imports and helper functions for sequence similarity analysis."""

from pathlib import Path

import pandas as pd
import polars as pl
from datasets import load_dataset


def get_dataset_config(dataset_name: str) -> dict:
    """Get configuration for a specific dataset."""
    for dataset in config["datasets"]:
        if dataset["name"] == dataset_name:
            return dataset
    raise ValueError(f"Dataset {dataset_name} not found in config")


def get_hf_path(dataset_name: str) -> str:
    """Get HuggingFace path for a dataset."""
    return get_dataset_config(dataset_name)["hf_path"]


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(complement.get(base, "N") for base in reversed(seq.upper()))


def canonical_sequence(seq: str) -> str:
    """Return the canonical form of a DNA sequence.

    The canonical form is the lexicographically smaller of the sequence
    and its reverse complement. This ensures that a sequence and its
    reverse complement are treated as identical for clustering purposes.
    """
    seq_upper = seq.upper()
    rc = reverse_complement(seq_upper)
    return seq_upper if seq_upper <= rc else rc


def load_sequences_from_hf(
    hf_path: str,
    split: str,
    seq_column: str = "seq",
    canonicalize: bool = False,
) -> pl.DataFrame:
    """Load sequences from a HuggingFace dataset.

    Args:
        hf_path: HuggingFace dataset path
        split: Dataset split (train or validation)
        seq_column: Name of the sequence column
        canonicalize: If True, convert sequences to canonical form
            (lexicographically smaller of seq and reverse complement)

    Returns:
        Polars DataFrame with columns [id, seq, split]
    """
    ds = load_dataset(hf_path, split=split)

    sequences = ds[seq_column]
    if canonicalize:
        sequences = [canonical_sequence(s) for s in sequences]

    df = pl.DataFrame({
        "seq": sequences,
    })
    # Add unique IDs
    df = df.with_row_index("id")
    df = df.with_columns(
        pl.concat_str([pl.lit(f"{split}_"), pl.col("id").cast(pl.Utf8)]).alias("id"),
        pl.lit(split).alias("split"),
    )
    return df.select(["id", "seq", "split"])
