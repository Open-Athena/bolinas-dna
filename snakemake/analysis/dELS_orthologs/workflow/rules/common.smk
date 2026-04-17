"""Common imports and helpers for the dELS_orthologs pipeline."""

from pathlib import Path

import bioframe as bf
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns


def get_search_window(species: str) -> tuple[str | None, int, int | None]:
    """Return (chrom, start, end) of the search region for `species`.

    Three modes, in order of precedence:

    - Whole-genome (`whole_genome: true`): returns (None, 0, None). Target
      spans every standard chromosome; downstream code should use the
      aligner's per-hit target name as `hit_chrom` rather than a config
      constant.
    - Whole-chromosome (`whole_chrom: true`): returns (chrom, 0, None).
      Target is one full chromosome.
    - Windowed (default): applies per-species `flank_bp` to the configured
      biological coordinates and returns (chrom, start, end).

    Callers that need a chromosome length read it from `{species}.chrom.sizes`
    (produced by the `chrom_sizes` rule).
    """
    region = config["search_region"][species]
    if region.get("whole_genome"):
        return None, 0, None
    chrom = region["chrom"]
    if region.get("whole_chrom"):
        return chrom, 0, None
    flank = region.get("flank_bp", 0)
    start = max(0, region["start"] - flank)
    end = region["end"] + flank
    return chrom, start, end


def is_whole_genome(species: str) -> bool:
    return bool(config["search_region"][species].get("whole_genome"))


# Standard mm10 chromosomes for whole-genome target extraction.
# Excludes chrM, chr*_random, chrUn_*, alt scaffolds.
MM10_STANDARD_CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]


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
