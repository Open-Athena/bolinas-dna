import textwrap

from bolinas.projection.hal import (
    HALLIFTOVER_BED_COLUMNS,
    HALLIFTOVER_BED_SCHEMA,
    attach_src_size,
    parse_halliftover_bed,
)


def test_parse_empty_returns_schema_correct_frame(tmp_path):
    p = tmp_path / "empty.bed"
    p.write_text("")
    out = parse_halliftover_bed(p, species="Mus_musculus")
    assert out.is_empty()
    assert set(out.columns) == set(HALLIFTOVER_BED_COLUMNS) | {"species"}


def test_parse_six_column_bed(tmp_path):
    bed = textwrap.dedent(
        """\
        chr5\t1000\t1100\twin_chr1_000001000\t0\t+
        chr5\t1100\t1200\twin_chr1_000001000\t0\t+
        chr7\t5000\t5100\twin_chr1_000001128\t0\t-
        """
    )
    p = tmp_path / "out.bed"
    p.write_text(bed)
    df = parse_halliftover_bed(p, species="Mus_musculus")
    assert df.height == 3
    assert df["species"].unique().to_list() == ["Mus_musculus"]
    assert df["t_chrom"].to_list() == ["chr5", "chr5", "chr7"]
    assert df["t_start"].to_list() == [1000, 1100, 5000]
    assert df["t_end"].to_list() == [1100, 1200, 5100]
    assert df["query_name"].to_list() == [
        "win_chr1_000001000",
        "win_chr1_000001000",
        "win_chr1_000001128",
    ]
    assert df["t_strand"].to_list() == ["+", "+", "-"]


def test_parse_multi_block_per_query(tmp_path):
    """Two BED rows for one query name = projection split across blocks."""
    bed = "chrX\t100\t150\twin\t0\t+\nchrX\t200\t250\twin\t0\t+\n"
    p = tmp_path / "split.bed"
    p.write_text(bed)
    df = parse_halliftover_bed(p, species="Mus_musculus")
    assert df.filter(df["query_name"] == "win").height == 2


def test_attach_src_size(tmp_path):
    bed = "chr5\t1000\t1100\twin1\t0\t+\nchr7\t5000\t5100\twin2\t0\t-\n"
    bed_path = tmp_path / "out.bed"
    bed_path.write_text(bed)
    sizes = "chr5\t200000000\nchr7\t150000000\nchr_unused\t99\n"
    sizes_path = tmp_path / "chrom.sizes"
    sizes_path.write_text(sizes)

    df = parse_halliftover_bed(bed_path, species="Mus_musculus")
    df = attach_src_size(df, sizes_path)

    assert df.height == 2
    assert df.filter(df["t_chrom"] == "chr5")["t_src_size"].item() == 200_000_000
    assert df.filter(df["t_chrom"] == "chr7")["t_src_size"].item() == 150_000_000


def test_attach_src_size_drops_unmapped_chrom(tmp_path):
    """A chrom not in the chrom.sizes table is dropped (inner join). This
    surfaces as a row count drop in the comparison script."""
    bed = "chr_orphan\t0\t100\twin\t0\t+\n"
    bed_path = tmp_path / "out.bed"
    bed_path.write_text(bed)
    sizes_path = tmp_path / "chrom.sizes"
    sizes_path.write_text("chr5\t200000000\n")

    df = parse_halliftover_bed(bed_path, species="Mus_musculus")
    df = attach_src_size(df, sizes_path)
    assert df.is_empty()


def test_schema_types_match_constants():
    assert set(HALLIFTOVER_BED_SCHEMA) == set(HALLIFTOVER_BED_COLUMNS)
