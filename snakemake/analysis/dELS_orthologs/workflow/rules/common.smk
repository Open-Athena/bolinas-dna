"""Common imports and helpers for the dELS_orthologs pipeline."""

from pathlib import Path

import bioframe as bf
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns


def get_search_window(species: str) -> tuple[str, int, int]:
    """Return the flanked (chrom, start, end) of the search region for `species`.

    Flanks are per-species (asymmetric search): the hg38 query side uses
    `flank_bp: 0` (ZRS proper) while the mm10 target side uses `flank_bp:
    100000` (room for the mouse ortholog to sit anywhere within the wider
    locus).
    """
    region = config["search_region"][species]
    flank = region["flank_bp"]
    start = max(0, region["start"] - flank)
    end = region["end"] + flank
    return region["chrom"], start, end


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
