"""Materialize sequences from a reference genome into eval harness format.

Each input variant emits **two rows**: one for ``strand="+"`` (FWD), one for
``strand="-"`` (reverse complement of the same window). Online lm_eval scorers
compute the per-strand LLR and average across strands per variant — the
matched-pair leaderboard improvement documented in #175 conclusion 2.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import datasets

from bolinas.data.dna import complement_base, reverse_complement
from bolinas.data.genome import Genome
from bolinas.data.transforms import in_seq_var_pos


def _add_eval_harness_fields(
    example: dict[str, Any],
    genome: Genome,
    window_size: int,
    strand: Literal["+", "-"],
) -> dict[str, Any]:
    """Per-example transform: extract context/ref_completion/alt_completion for one strand.

    Assumes SNVs (single-nucleotide variants) where ``len(ref) == len(alt) == 1``.

    For a variant at 1-based ``pos`` in a window of ``window_size`` centered on
    the variant:

      ``var_pos = in_seq_var_pos(window_size, strand)``         # variant index in window
      ``context``        = window[:var_pos]                     # left flank
      ``ref_completion`` = ref_in_strand + window[var_pos + 1:] # ref + right flank
      ``alt_completion`` = alt_in_strand + window[var_pos + 1:] # alt + right flank

    where ``window`` is the genomic window (FWD strand) or its reverse complement
    (RC strand), and ``ref_in_strand`` / ``alt_in_strand`` are the ref/alt
    nucleotides as read off the requested strand (complemented for RC).

    Window length math (general for any ``window_size``):

      - FWD: ``var_pos = window_size // 2``. Context length =
        ``window_size // 2``; completion length = ``window_size - window_size // 2``.
      - RC:  ``var_pos = window_size - 1 - window_size // 2``. Context length =
        ``window_size - 1 - window_size // 2``; completion length = ``window_size // 2 + 1``.

      For odd ``window_size`` (e.g. 255) the FWD and RC layouts are symmetric
      — same context/completion lengths on both strands. For even
      ``window_size`` (e.g. 256) the RC context is one bp shorter and the RC
      completion is one bp longer; the harness consumes (context, completion)
      tokens independently per row, so the asymmetry is fine.
    """
    chrom = str(example["chrom"])
    pos = int(example["pos"])
    ref = str(example["ref"]).upper()
    alt = str(example["alt"]).upper()

    center = pos - 1  # 0-based
    start = center - window_size // 2
    end = start + window_size

    window = genome(chrom, start, end).upper()
    if strand == "-":
        window = reverse_complement(window)
        ref_in_strand = complement_base(ref)
        alt_in_strand = complement_base(alt)
    else:
        ref_in_strand = ref
        alt_in_strand = alt

    var_pos = in_seq_var_pos(window_size, strand)
    assert window[var_pos] == ref_in_strand, (
        f"window[{var_pos}]={window[var_pos]!r} != ref_in_strand={ref_in_strand!r} "
        f"(chrom={chrom}, pos={pos}, ref={ref}, alt={alt}, strand={strand}, "
        f"window_size={window_size})"
    )

    right_flank = window[var_pos + 1 :]
    return {
        "context": window[:var_pos],
        "ref_completion": ref_in_strand + right_flank,
        "alt_completion": alt_in_strand + right_flank,
        "strand": strand,
    }


def materialize_sequences(
    dataset: datasets.Dataset,
    genome: Genome,
    window_size: int,
) -> datasets.Dataset:
    """Add materialized sequence fields to a variant dataset.

    Each input variant emits two output rows: one with ``strand="+"`` (FWD),
    one with ``strand="-"`` (RC of the same genomic window). Renames
    ``label`` → ``target`` and adds the columns
    ``[context, ref_completion, alt_completion, strand]``. Assumes SNVs.

    Output rows are sorted by ``(chrom, pos, ref, alt, strand)`` so per-variant
    pairs are adjacent — protects per-variant aggregation downstream against
    a future caller slicing the dataset between strand pairs.

    Args:
        dataset: HF Dataset with columns ``[chrom, pos, ref, alt, label]`` and
            optionally ``[subset, match_group, ...]`` (any extra columns are
            preserved on both output rows).
        genome: Loaded :class:`Genome` instance.
        window_size: Total window size (bp) centered on the variant.

    Returns:
        Dataset with ``2 * len(dataset)`` rows; each input variant has one row
        per strand, plus the new columns above.
    """
    fwd = dataset.map(
        partial(
            _add_eval_harness_fields, genome=genome, window_size=window_size, strand="+"
        ),
    )
    rc = dataset.map(
        partial(
            _add_eval_harness_fields, genome=genome, window_size=window_size, strand="-"
        ),
    )
    out = datasets.concatenate_datasets([fwd, rc])
    if "label" in out.column_names:
        out = out.rename_column("label", "target")
    sort_keys = [
        c for c in ("chrom", "pos", "ref", "alt", "strand") if c in out.column_names
    ]
    return out.sort(sort_keys)
