"""Per-window phyloP scoring via pyBigWig.

Two reasons we use pyBigWig directly instead of the kentUtils
``bigWigAverageOverBed`` route:

1. The bioconda kentUtils binaries (``bedGraphToBigWig``, ``faToTwoBit``)
   don't accept pipes on ``stdin`` — they need a regular file. The
   ``binarize via bigWigToBedGraph | awk | bedGraphToBigWig`` chain we
   tried fails on the ``stdin`` step. Materialising the per-base bedGraph
   to a temp file would cost ~30 GB and another full I/O pass.
2. ``run:`` blocks don't activate the rule's conda env, so calling
   ``bigWigAverageOverBed`` via ``subprocess`` from a ``run:`` block needs
   a separate shell rule and TSV-parsing step. Doing the math directly
   in Python keeps it to one rule and one library function.

Performance: pyBigWig handles ~10K windows/sec/core on 255 bp windows.
24M windows → ~40 min single-threaded; parallelisable across windows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pyBigWig

def score_windows(
    bw_path: str | Path,
    windows: pl.DataFrame,
    threshold: float,
    *,
    chrom_prefix: str = "chr",
) -> pl.DataFrame:
    """Score each window in ``windows`` against the bigWig at ``bw_path``.

    Args:
        bw_path: path to the phyloP bigWig (UCSC ``chr1``-style chrom names).
        windows: polars frame with columns ``chrom``, ``start``, ``end``;
            ``chrom`` may be either ``"1"`` or ``"chr1"``-style — bare names
            are auto-prefixed with ``chrom_prefix`` for the bigWig lookup.
        threshold: phyloP value at/above which a base is "conserved".
        chrom_prefix: prefix to add to bare chrom names (default ``"chr"``).

    Returns:
        ``windows`` with four extra columns:
          - ``conserved_bases`` (Int32): count of bases with value ≥
            ``threshold``. NaN positions are counted as **non-conserved**
            (= 0), so the count is over the whole window length, not just
            the covered (non-NaN) bases.
          - ``proportion_conserved`` (Float32): ``conserved_bases / (end - start)``.
          - ``mean_phylop`` (Float32): mean of finite values in the window
            (NaN if no finite values).
          - ``n_valid_bases`` (Int32): count of bases where the bigWig has
            a finite value (i.e. excludes NaN).
    """
    assert {"chrom", "start", "end"}.issubset(windows.columns), (
        f"windows must have chrom/start/end columns; got {windows.columns}"
    )

    conserved_bases = np.empty(len(windows), dtype=np.int32)
    proportion_conserved = np.empty(len(windows), dtype=np.float32)
    mean_phylop = np.empty(len(windows), dtype=np.float32)
    n_valid_bases = np.empty(len(windows), dtype=np.int32)

    bw = pyBigWig.open(str(bw_path))
    try:
        bw_chroms = set(bw.chroms())
        chroms = windows["chrom"].to_list()
        starts = windows["start"].to_list()
        ends = windows["end"].to_list()
        for i, (chrom, start, end) in enumerate(zip(chroms, starts, ends)):
            start = int(start)
            end = int(end)
            size = end - start
            assert size > 0, f"empty window at index {i}: {chrom}:{start}-{end}"
            bw_chrom = (
                chrom if chrom.startswith(chrom_prefix) else f"{chrom_prefix}{chrom}"
            )
            if bw_chrom not in bw_chroms:
                conserved_bases[i] = 0
                proportion_conserved[i] = 0.0
                mean_phylop[i] = np.nan
                n_valid_bases[i] = 0
                continue
            values = np.asarray(
                bw.values(bw_chrom, start, end, numpy=True), dtype=np.float64
            )
            finite_mask = np.isfinite(values)
            n_finite = int(finite_mask.sum())
            # NaN treated as non-conserved: NaN >= threshold evaluates to
            # False in NumPy regardless of threshold, so the NaN positions
            # are naturally excluded from the conserved-bases count
            # *without* needing to fill them. (Don't fill with 0 — that
            # breaks the case where threshold <= 0.)
            n_conserved = int((values >= threshold).sum())
            conserved_bases[i] = n_conserved
            proportion_conserved[i] = n_conserved / size
            mean_phylop[i] = float(np.nanmean(values)) if n_finite > 0 else np.nan
            n_valid_bases[i] = n_finite
    finally:
        bw.close()

    return windows.with_columns(
        [
            pl.Series("conserved_bases", conserved_bases, dtype=pl.Int32),
            pl.Series("proportion_conserved", proportion_conserved, dtype=pl.Float32),
            pl.Series("mean_phylop", mean_phylop, dtype=pl.Float32),
            pl.Series("n_valid_bases", n_valid_bases, dtype=pl.Int32),
        ]
    )


_COLUMNS: tuple[str, ...] = ("name", "size", "covered", "sum", "mean0", "mean")


def parse_bigwig_average_over_bed(tsv_path: str | Path) -> pl.DataFrame:
    """Read a ``bigWigAverageOverBed`` TSV into a Polars frame.

    Output schema:
      name (str), size (i64), covered (i64), sum (f64), mean0 (f64), mean (f64)

    The ``mean`` column is NaN where ``covered == 0`` (no bigWig signal in
    the interval). All asserts are run after parsing: invariants that
    should hold for every row are checked here so a malformed TSV fails
    fast at the parsing boundary.
    """
    df = pl.read_csv(
        tsv_path,
        separator="\t",
        has_header=False,
        new_columns=list(_COLUMNS),
        schema_overrides={
            "name": pl.Utf8,
            "size": pl.Int64,
            "covered": pl.Int64,
            "sum": pl.Float64,
            "mean0": pl.Float64,
            "mean": pl.Float64,
        },
        null_values=["n/a"],
    )

    assert set(df.columns) == set(_COLUMNS), (
        f"unexpected columns from bigWigAverageOverBed: {df.columns}"
    )
    assert (df["size"] > 0).all(), "non-positive size in bigWigAverageOverBed output"
    assert (df["covered"] >= 0).all(), (
        "negative `covered` in bigWigAverageOverBed output"
    )
    assert (df["covered"] <= df["size"]).all(), (
        "`covered` > `size` in bigWigAverageOverBed output"
    )
    return df
