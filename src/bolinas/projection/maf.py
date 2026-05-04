"""MAF block parsing and column-precise projection.

The Zoonomia 447 single-copy MAF (``447-mammalian-2022v1.fix2.single.maf.gz``,
779 GB, anchored on ``hg38``) is a sequence of alignment blocks.
Each block has one ``s`` row per species present, with the aligned
sub-sequence (gap characters ``-`` interspersed). ``parse_maf_blocks``
yields one block at a time; ``project_window_through_block`` walks the
alignment column-by-column to map a human-anchored query interval
onto every non-anchor species in the block, producing a forward-strand
target coord (with strand recorded separately, matching halLiftover's
output convention).

Two parsing modes:

- **Sequential** (``region=None``): ``bx-python`` over the gzipped MAF
  from byte 0. Works on any MAF (with or without ``.tai``); slow on the
  full 779 GB file (CPU-bound on gzip+parse, ~7 MB/s).
- **Indexed random access** (``region="chr1:0-N"``): subprocess to the
  ``taffy view -r`` CLI, which seeks via the companion ``.tai`` index
  and emits MAF blocks for the requested region. Bx-python parses
  taffy's stdout. Dramatically faster for narrow regions (e.g. chr1
  windows from a whole-genome MAF).

The CLI path is configurable via ``taffy_path``; defaults to ``"taffy"``
on PATH (the Cactus binary tarball ships ``taffy`` alongside
``halLiftover``).
"""

from __future__ import annotations

import gzip
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from bx.align import maf as bx_maf


@dataclass(frozen=True)
class MafRow:
    """One species' row within a MAF block.

    Coordinate system follows the MAF spec: ``start`` and ``size`` give the
    aligning region in the source sequence, and if ``strand == '-'`` they
    refer to positions on the reverse-complemented source. ``aligned_seq``
    is the sequence text (uppercase + gap chars).
    """

    species: str
    chrom: str
    start: int
    size: int
    strand: str
    src_size: int
    aligned_seq: str

    @property
    def end(self) -> int:
        return self.start + self.size


@dataclass(frozen=True)
class ProjectionRecord:
    """One target-species projection of a query window through a single MAF block.

    ``t_start`` / ``t_end`` are forward-strand coords of ``t_chrom``,
    matching halLiftover's output convention. ``t_strand`` records the
    strand the alignment landed on. ``t_aligned_len`` is the count of
    non-gap target-row chars in the projected slice — useful for
    diagnostics; equals ``t_end - t_start``.
    """

    query_name: str
    species: str
    t_chrom: str
    t_start: int
    t_end: int
    t_strand: str
    t_src_size: int
    t_aligned_len: int


def _component_to_row(c) -> MafRow:
    """Convert a ``bx.align`` component to ``MafRow``."""
    species, _, chrom = c.src.partition(".")
    return MafRow(
        species=species,
        chrom=chrom,
        start=int(c.start),
        size=int(c.size),
        strand=c.strand,
        src_size=int(c.src_size),
        aligned_seq=c.text,
    )


def parse_maf_blocks(
    path: str | Path,
    *,
    region: str | None = None,
    taffy_path: str = "taffy",
) -> Iterator[list[MafRow]]:
    """Yield each MAF block as a list of ``MafRow``.

    Args:
        path: ``.maf`` or ``.maf.gz`` file. Handles gzip transparently.
        region: optional ``"chrom:start-end"`` (0-based half-open, matching
            ``taffy view -r``). When given, uses ``taffy`` for indexed
            random access via the companion ``.tai``; ``path.tai`` must
            exist alongside the MAF. When ``None``, sequentially streams
            the whole file with ``bx-python``.
        taffy_path: path to the ``taffy`` CLI; defaults to PATH lookup.
    """
    path = Path(path)
    if region is None:
        fobj = gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)
        try:
            reader = bx_maf.Reader(fobj)
            for alignment in reader:
                yield [_component_to_row(c) for c in alignment.components]
        finally:
            fobj.close()
        return
    # Indexed random access via taffy. Pipe stdout into bx-python.
    proc = subprocess.Popen(
        [taffy_path, "view", "--inputFile", str(path), "--maf", "--region", region],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1 << 20,
    )
    try:
        assert proc.stdout is not None
        reader = bx_maf.Reader(proc.stdout)
        for alignment in reader:
            yield [_component_to_row(c) for c in alignment.components]
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"taffy view exited with code {rc} for region={region!r} on {path}"
            )


def _find_column_range(
    aligned_seq: str, row_start: int, target_start: int, target_end: int
) -> tuple[int, int] | None:
    """Find the alignment column range covering ``[target_start, target_end)`` on a row.

    The row's coordinates run from ``row_start`` along non-gap columns of
    ``aligned_seq``. Returns ``(col_start, col_end)`` such that the non-gap
    bases in ``aligned_seq[col_start:col_end]`` are exactly the row's bases
    overlapping ``[target_start, target_end)``. Returns ``None`` if there
    is no overlap.
    """
    pos = row_start
    col_start: int | None = None
    col_end: int | None = None
    for i, ch in enumerate(aligned_seq):
        if ch == "-":
            continue
        if col_start is None and pos >= target_start:
            col_start = i
        if pos >= target_end:
            col_end = i
            break
        pos += 1
    if col_start is None:
        return None
    if col_end is None:
        col_end = len(aligned_seq)
    return col_start, col_end


def project_window_through_block(
    block: list[MafRow],
    query_name: str,
    anchor_species: str,
    anchor_chrom: str,
    anchor_start: int,
    anchor_end: int,
) -> list[ProjectionRecord]:
    """Project ``[anchor_start, anchor_end)`` through one MAF block.

    Column-precise: walks the anchor row's alignment text to locate the
    overlapping column range, then walks each non-anchor species' row to
    map that column range back to forward-strand target coordinates.

    Returns ``[]`` if the block doesn't contain the anchor's chrom or if
    the query doesn't overlap the anchor row.
    """
    anchor_row: MafRow | None = None
    for row in block:
        if row.species == anchor_species and row.chrom == anchor_chrom:
            anchor_row = row
            break
    if anchor_row is None:
        return []
    if anchor_row.strand != "+":
        # Zoonomia MAF anchors on forward strand of Homo_sapiens; defensive.
        return []

    a_start = max(anchor_start, anchor_row.start)
    a_end = min(anchor_end, anchor_row.end)
    if a_start >= a_end:
        return []

    cols = _find_column_range(anchor_row.aligned_seq, anchor_row.start, a_start, a_end)
    if cols is None:
        return []
    col_start, col_end = cols

    records: list[ProjectionRecord] = []
    for row in block:
        if row.species == anchor_species:
            continue
        prefix = row.aligned_seq[:col_start]
        slice_ = row.aligned_seq[col_start:col_end]
        before = len(prefix) - prefix.count("-")
        within = len(slice_) - slice_.count("-")
        if within == 0:
            continue
        if row.strand == "+":
            t_start = row.start + before
            t_end = t_start + within
        else:
            # MAF '-' rows: row.start is on the reverse-complemented strand.
            # Forward-strand coords for the slice:
            t_end = row.src_size - row.start - before
            t_start = t_end - within
        # Defensive: forward-strand coords must lie within the chrom.
        assert 0 <= t_start <= t_end <= row.src_size, (
            f"projected coords out of bounds: species={row.species} "
            f"chrom={row.chrom} t_start={t_start} t_end={t_end} src_size={row.src_size} "
            f"strand={row.strand} row.start={row.start} row.size={row.size} "
            f"before={before} within={within}"
        )
        records.append(
            ProjectionRecord(
                query_name=query_name,
                species=row.species,
                t_chrom=row.chrom,
                t_start=t_start,
                t_end=t_end,
                t_strand=row.strand,
                t_src_size=row.src_size,
                t_aligned_len=within,
            )
        )
    return records
