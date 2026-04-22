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


def proportional_lift(
    qs: int, qe: int, ts: int, te: int, rev: bool, cs: int, ce: int
) -> tuple[int, int] | None:
    """Linear proportional lift of a cCRE query interval to target coords
    within an alignment, ignoring gaps.

    Given an alignment mapping query `[qs, qe)` → target `[ts, te)`
    (and a reverse-strand flag), and a cCRE query interval `[cs, ce)`
    contained in `[qs, qe)`, return the target sub-interval corresponding
    to the cCRE's portion of the alignment.

    On reverse-stranded alignments, the target end of the cCRE's span maps
    to the target "lower" coord and vice versa — we swap accordingly.

    Returns None if the cCRE does not overlap the alignment. If the cCRE
    extends past the alignment boundary, the lift is clipped to the
    alignment span.

    This is a first-order approximation — correct for gap-free alignments,
    off by the number of gaps on either side for gappy ones. Exact lift
    requires CIGAR parsing.
    """
    aln_len = qe - qs
    if aln_len <= 0 or ce <= qs or cs >= qe:
        return None
    cs_clip = max(cs, qs)
    ce_clip = min(ce, qe)
    frac_start = (cs_clip - qs) / aln_len
    frac_end = (ce_clip - qs) / aln_len
    t_span = te - ts
    if rev:
        return int(ts + (1.0 - frac_end) * t_span), int(ts + (1.0 - frac_start) * t_span)
    return int(ts + frac_start * t_span), int(ts + frac_end * t_span)


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
