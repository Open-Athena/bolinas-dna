"""Common imports and helpers for the dELS_orthologs pipeline."""

from pathlib import Path

import bioframe as bf
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns


def get_search_window(species: str) -> tuple[str, int, int | None]:
    """Return (chrom, start, end) of the search region for `species`.

    Windowed mode (default): applies per-species `flank_bp` to the configured
    biological coordinates. End is the flanked end.

    Whole-chromosome mode (`whole_chrom: true` in config): returns (chrom, 0,
    None). Callers that need the chromosome length should read it from the
    `{species}.chrom.sizes` file; callers that only need `start` for coord
    projection (the normalize rules) can ignore end.
    """
    region = config["search_region"][species]
    chrom = region["chrom"]
    if region.get("whole_chrom"):
        return chrom, 0, None
    flank = region.get("flank_bp", 0)
    start = max(0, region["start"] - flank)
    end = region["end"] + flank
    return chrom, start, end


def soft_masked_fraction_per_record(fasta_path: str) -> dict[str, float]:
    """Per-record fraction of lowercase (soft-masked, RepeatMasker) bases."""
    fracs: dict[str, float] = {}
    cur_id: str | None = None
    cur_chunks: list[str] = []

    def _commit() -> None:
        if cur_id is None:
            return
        s = "".join(cur_chunks)
        fracs[cur_id] = sum(1 for c in s if c.islower()) / max(1, len(s))

    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                _commit()
                cur_id = line[1:].strip().split()[0]
                cur_chunks = []
            else:
                cur_chunks.append(line.strip())
        _commit()
    return fracs
