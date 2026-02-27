"""Materialize sequences from a reference genome into eval harness format."""

from __future__ import annotations

from functools import partial
from typing import Any

from biofoundation.data import Genome
from datasets import Dataset


def _add_eval_harness_fields(
    example: dict[str, Any],
    genome: Genome,
    window_size: int,
) -> dict[str, Any]:
    """Per-example transform: extract context/ref_completion/alt_completion.

    Assumes SNVs (single-nucleotide variants) where len(ref) == len(alt) == 1.

    For a variant at 1-based ``pos`` in a window of ``window_size`` centered on
    the variant:

      context        = genome[window_start : variant_pos]   (left flank)
      ref_completion = ref + genome[variant_pos+1 : window_end] (ref + right flank)
      alt_completion = alt + genome[variant_pos+1 : window_end] (alt + right flank)
    """
    chrom = str(example["chrom"])
    pos = int(example["pos"])
    ref = str(example["ref"])
    alt = str(example["alt"])

    center = pos - 1  # 0-based
    start = center - window_size // 2
    end = center + window_size // 2

    context = genome(chrom, start, center).upper()
    right_flank = genome(chrom, center + 1, end).upper()

    return {
        "context": context,
        "ref_completion": ref.upper() + right_flank,
        "alt_completion": alt.upper() + right_flank,
    }


def materialize_sequences(
    dataset: Dataset,
    genome: Genome,
    window_size: int,
) -> Dataset:
    """Add materialized sequence fields to a variant dataset.

    Adds context/ref_completion/alt_completion columns and renames label -> target.
    Assumes all variants are SNVs.

    Args:
        dataset: HF Dataset with columns [chrom, pos, ref, alt, label].
        genome: Loaded Genome instance.
        window_size: Total window size centered on the variant (must be even).

    Returns:
        Dataset with added columns [context, ref_completion, alt_completion, target].
    """
    dataset = dataset.map(
        partial(_add_eval_harness_fields, genome=genome, window_size=window_size),
    )
    if "label" in dataset.column_names:
        dataset = dataset.rename_column("label", "target")
    return dataset
