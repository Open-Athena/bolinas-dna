import gzip
import textwrap

import pytest

from bolinas.projection.maf import (
    MafRow,
    parse_maf_blocks,
    project_window_through_block,
)


SIMPLE_MAF = textwrap.dedent(
    """\
    ##maf version=1

    a score=100
    s Homo_sapiens.chr1 1000 30 + 250000000 ACGTACGTACGTACGTACGTACGTACGTAC
    s Mus_musculus.chr5 5000 30 + 200000000 GCATGCATGCATGCATGCATGCATGCATGC
    s Bos_taurus.chr3 3000 30 - 100000000 TTAATTAATTAATTAATTAATTAATTAATT

    a score=50
    s Homo_sapiens.chr2 100 8 + 250000000 ACGTACGT
    s Mus_musculus.chr1 200 6 + 200000000 AC--GTAC
    """
)


@pytest.fixture
def maf_path(tmp_path):
    p = tmp_path / "test.maf"
    p.write_text(SIMPLE_MAF)
    return p


@pytest.fixture
def maf_gz_path(tmp_path):
    p = tmp_path / "test.maf.gz"
    with gzip.open(p, "wt") as f:
        f.write(SIMPLE_MAF)
    return p


def test_parse_maf_block_count(maf_path):
    blocks = list(parse_maf_blocks(maf_path))
    assert len(blocks) == 2
    assert len(blocks[0]) == 3  # human + mouse + cow
    assert len(blocks[1]) == 2


def test_parse_maf_handles_gz(maf_gz_path):
    blocks = list(parse_maf_blocks(maf_gz_path))
    assert len(blocks) == 2


def test_maf_row_fields(maf_path):
    blocks = list(parse_maf_blocks(maf_path))
    human, mouse, cow = blocks[0]
    assert human.species == "Homo_sapiens"
    assert human.chrom == "chr1"
    assert human.start == 1000
    assert human.size == 30
    assert human.strand == "+"
    assert human.src_size == 250_000_000
    assert human.aligned_seq == "ACGTACGTACGTACGTACGTACGTACGTAC"

    assert cow.species == "Bos_taurus"
    assert cow.strand == "-"
    assert cow.src_size == 100_000_000


def test_maf_row_end_property():
    row = MafRow(
        species="Mus_musculus",
        chrom="chr1",
        start=100,
        size=50,
        strand="+",
        src_size=200_000,
        aligned_seq="N" * 50,
    )
    assert row.end == 150


def test_project_full_block_plus_and_minus(maf_path):
    blocks = list(parse_maf_blocks(maf_path))
    block = blocks[0]
    records = project_window_through_block(
        block,
        query_name="win_chr1_000001000",
        anchor_species="Homo_sapiens",
        anchor_chrom="chr1",
        anchor_start=1000,
        anchor_end=1030,
    )
    by_species = {r.species: r for r in records}
    assert set(by_species) == {"Mus_musculus", "Bos_taurus"}

    # Mus is + strand; full alignment slice -> [5000, 5030).
    mus = by_species["Mus_musculus"]
    assert mus.t_chrom == "chr5"
    assert mus.t_start == 5000
    assert mus.t_end == 5030
    assert mus.t_strand == "+"
    assert mus.t_src_size == 200_000_000
    assert mus.t_aligned_len == 30

    # Bos is - strand; forward-strand coords for the slice:
    #   t_end = src_size - row.start - before = 100M - 3000 - 0 = 99_997_000
    #   t_start = t_end - within = 99_997_000 - 30 = 99_996_970
    bos = by_species["Bos_taurus"]
    assert bos.t_chrom == "chr3"
    assert bos.t_strand == "-"
    assert bos.t_start == 99_996_970
    assert bos.t_end == 99_997_000


def test_project_partial_window_plus_strand(maf_path):
    blocks = list(parse_maf_blocks(maf_path))
    # Project just the leading 10 bp of the first block.
    records = project_window_through_block(
        blocks[0],
        query_name="q",
        anchor_species="Homo_sapiens",
        anchor_chrom="chr1",
        anchor_start=1000,
        anchor_end=1010,
    )
    mus = next(r for r in records if r.species == "Mus_musculus")
    assert mus.t_start == 5000
    assert mus.t_end == 5010

    # Project the trailing 10 bp.
    records = project_window_through_block(
        blocks[0],
        query_name="q",
        anchor_species="Homo_sapiens",
        anchor_chrom="chr1",
        anchor_start=1020,
        anchor_end=1030,
    )
    mus = next(r for r in records if r.species == "Mus_musculus")
    assert mus.t_start == 5020
    assert mus.t_end == 5030

    # And the trailing 10 bp on the - strand cow:
    bos = next(r for r in records if r.species == "Bos_taurus")
    # For trailing slice: before=20, within=10. t_end = 100M - 3000 - 20 = 99_996_980
    # t_start = 99_996_980 - 10 = 99_996_970
    assert bos.t_end == 99_996_980
    assert bos.t_start == 99_996_970


def test_project_handles_gaps_in_target(maf_path):
    """Block 2 has a target row with internal gaps. Verify the column-precise
    walk maps human [102, 105) to mouse [202, 203) — only one mouse base in
    that human range because cols 2 and 3 are mouse gaps.
    """
    blocks = list(parse_maf_blocks(maf_path))
    records = project_window_through_block(
        blocks[1],
        query_name="q",
        anchor_species="Homo_sapiens",
        anchor_chrom="chr2",
        anchor_start=102,
        anchor_end=105,
    )
    assert len(records) == 1
    mus = records[0]
    assert mus.species == "Mus_musculus"
    assert mus.t_start == 202
    assert mus.t_end == 203
    assert mus.t_aligned_len == 1


def test_project_no_anchor_in_block_returns_empty(maf_path):
    blocks = list(parse_maf_blocks(maf_path))
    records = project_window_through_block(
        blocks[0],
        query_name="q",
        anchor_species="Homo_sapiens",
        anchor_chrom="chrX",  # not in block
        anchor_start=0,
        anchor_end=255,
    )
    assert records == []


def test_project_no_overlap_returns_empty(maf_path):
    blocks = list(parse_maf_blocks(maf_path))
    records = project_window_through_block(
        blocks[0],
        query_name="q",
        anchor_species="Homo_sapiens",
        anchor_chrom="chr1",
        anchor_start=2_000,  # past the block's end at 1030
        anchor_end=2_255,
    )
    assert records == []


def test_project_target_all_gap_in_slice_skipped(tmp_path):
    """When a target row has only gap chars in the projected slice, that
    species is skipped (no record emitted)."""
    maf = textwrap.dedent(
        """\
        ##maf version=1

        a score=10
        s Homo_sapiens.chr1 100 4 + 250000000 ACGT
        s Mus_musculus.chr1 200 1 + 200000000 ---T
        """
    )
    p = tmp_path / "gap.maf"
    p.write_text(maf)
    blocks = list(parse_maf_blocks(p))
    # Project human [100, 103) which corresponds to anchor cols 0..3, mouse
    # has only gaps in cols 0..2 -> within=0, no record.
    records = project_window_through_block(
        blocks[0],
        query_name="q",
        anchor_species="Homo_sapiens",
        anchor_chrom="chr1",
        anchor_start=100,
        anchor_end=103,
    )
    assert records == []


def test_project_minus_strand_row_start_math(tmp_path):
    """Sanity-check the - strand math against a hand-computed example.

    Cow row: start=10 on rev-comp coord, size=4, src_size=1000, strand='-'.
    Forward-strand range: [1000 - 10 - 4, 1000 - 10) = [986, 990).
    Aligned text "GCAT" maps to forward-strand bases at 989, 988, 987, 986
    (in that order — first char of rev-comp = highest forward pos).

    Project human [105, 107) (anchor cols 0..1 of "ACGT") -> col_start=0,
    col_end=2. Cow slice = "GC" -> within=2. before=0.
    t_end = 1000 - 10 - 0 = 990.
    t_start = 990 - 2 = 988.
    Forward-strand range [988, 990) — corresponding to cow forward positions
    988 and 989, which are the rev-comp counterparts of cow cols 0 and 1.
    """
    maf = textwrap.dedent(
        """\
        ##maf version=1

        a score=20
        s Homo_sapiens.chr5 105 4 + 250000000 ACGT
        s Bos_taurus.chr1 10 4 - 1000 GCAT
        """
    )
    p = tmp_path / "rev.maf"
    p.write_text(maf)
    blocks = list(parse_maf_blocks(p))
    records = project_window_through_block(
        blocks[0],
        query_name="q",
        anchor_species="Homo_sapiens",
        anchor_chrom="chr5",
        anchor_start=105,
        anchor_end=107,
    )
    assert len(records) == 1
    bos = records[0]
    assert bos.t_strand == "-"
    assert bos.t_start == 988
    assert bos.t_end == 990
    assert bos.t_src_size == 1000
