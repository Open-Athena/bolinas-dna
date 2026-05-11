"""Tests for the eQTL Catalogue per-tissue parser.

The labeling contract is critical: every variant that's tested but never in
a credible set should end up with ``pip=0`` after merge, then label=False
after the downstream cascade. These tests pin down the 0-fill semantics
and integrate with ``label_variants_by_pip`` to verify the end-to-end
labeling story matches the plan's Stage A → C tables.
"""

from __future__ import annotations

import gzip
import tempfile
from pathlib import Path

import polars as pl
import pytest

from bolinas.evals.catalogue_parser import (
    _parse_variant_id,
    extract_tested_variants,
    merge_cs_and_sumstats,
    parse_credible_sets,
)
from bolinas.evals.labeling import label_variants_by_pip


def _write_tsv_gz(rows: list[list[str]], header: list[str]) -> str:
    """Write a TSV.gz to a tempfile; return its path."""
    fd = tempfile.NamedTemporaryFile(suffix=".tsv.gz", delete=False)
    fd.close()
    with gzip.open(fd.name, "wt") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    return fd.name


# Header schemas — must match real Catalogue files (verified on QTD000116).
_CS_HEADER = [
    "molecular_trait_id", "gene_id", "cs_id", "variant", "rsid", "cs_size",
    "pip", "pvalue", "beta", "se", "z", "cs_min_r2", "region",
]
_SUMSTATS_HEADER = [
    "molecular_trait_id", "chromosome", "position", "ref", "alt", "variant",
    "ma_samples", "maf", "pvalue", "beta", "se", "type", "ac", "an", "r2",
    "molecular_trait_object_id", "gene_id", "median_tpm", "rsid",
]


def _cs_row(
    variant: str, gene: str, pip: float, cs_index: int = 1
) -> list[str]:
    """Build one credible-set row."""
    return [
        gene, gene, f"{gene}_L{cs_index}", variant, ".", "1",
        f"{pip}", "1e-8", "0.5", "0.1", "5.0", "0.6", "chr1:1-2000000",
    ]


def _sumstats_row(
    variant: str, gene: str, maf: float
) -> list[str]:
    """Build one sumstats row. Parse chrom/pos/ref/alt from variant for the
    explicit columns Catalogue ships."""
    chrom, pos, ref, alt = variant.removeprefix("chr").split("_")
    return [
        gene, chrom, pos, ref, alt, variant,
        "18", f"{maf}", "1e-3", "0.3", "0.2", "SNP", "18", "1162", "NA",
        gene, gene, "3.227", ".",
    ]


# ---------- _parse_variant_id ----------


def test_parse_variant_id_snp():
    df = pl.DataFrame({"variant": ["chr1_100002416_C_T"]})
    chrom, pos, ref, alt = _parse_variant_id(pl.col("variant"))
    out = df.with_columns(chrom, pos, ref, alt).to_dicts()[0]
    assert out["chrom"] == "1"
    assert out["pos"] == 100002416
    assert out["ref"] == "C"
    assert out["alt"] == "T"


def test_parse_variant_id_indel():
    """Indel ref/alt are multi-char (e.g. CTTAT). Splitting on _ still yields
    exactly 4 parts because chromosome never contains _."""
    df = pl.DataFrame({"variant": ["chr7_66367367_C_CTTAT"]})
    chrom, pos, ref, alt = _parse_variant_id(pl.col("variant"))
    out = df.with_columns(chrom, pos, ref, alt).to_dicts()[0]
    assert out["chrom"] == "7"
    assert out["pos"] == 66367367
    assert out["ref"] == "C"
    assert out["alt"] == "CTTAT"


def test_parse_variant_id_sex_chrom():
    df = pl.DataFrame({"variant": ["chrX_1000_A_G"]})
    chrom, pos, ref, alt = _parse_variant_id(pl.col("variant"))
    out = df.with_columns(chrom, pos, ref, alt).to_dicts()[0]
    assert out["chrom"] == "X"
    assert out["pos"] == 1000


# Default gene_biotype frame for tests that don't care about the biotype agg.
def _gene_biotype_df(pairs: list[tuple[str, str]] | None = None) -> pl.DataFrame:
    """Build a `(gene_id, biotype_class)` frame. Defaults to all-`pc` for the
    test fixture gene names. Tests covering biotype-class aggregation pass
    custom pairs."""
    pairs = pairs or [("ENSG00000000001", "pc"), ("GENE_A", "pc"), ("GENE_B", "pc"),
                      ("GENE_C", "pc"), ("G", "pc"), ("GENE_NC", "nc")]
    return pl.DataFrame({"gene_id": [p[0] for p in pairs],
                         "biotype_class": [p[1] for p in pairs]})


# ---------- parse_credible_sets ----------


def test_parse_credible_sets_single_variant():
    rows = [_cs_row("chr1_100_A_T", "ENSG00000000001", 0.95)]
    path = _write_tsv_gz(rows, _CS_HEADER)
    out = parse_credible_sets(path)
    assert out.height == 1
    r = out.to_dicts()[0]
    assert (r["chrom"], r["pos"], r["ref"], r["alt"]) == ("1", 100, "A", "T")
    assert r["gene_id"] == "ENSG00000000001"
    assert r["pip"] == pytest.approx(0.95)


def test_parse_credible_sets_max_per_variant_gene_pair():
    """Per-(variant, gene_id) max PIP. Same variant in two different genes
    gets two rows; same variant + same gene with multiple CS signals
    (L1, L2, ...) gets one row with max PIP."""
    rows = [
        _cs_row("chr1_100_A_T", "GENE_A", 0.3, cs_index=1),
        _cs_row("chr1_100_A_T", "GENE_B", 0.95, cs_index=1),
        _cs_row("chr1_100_A_T", "GENE_A", 0.5, cs_index=2),  # same variant+gene, L2
    ]
    path = _write_tsv_gz(rows, _CS_HEADER)
    out = parse_credible_sets(path).sort("gene_id")
    assert out.height == 2  # GENE_A and GENE_B, NOT collapsed to 1 row
    by_gene = {r["gene_id"]: r["pip"] for r in out.to_dicts()}
    assert by_gene["GENE_A"] == pytest.approx(0.5)  # max of L1=0.3 and L2=0.5
    assert by_gene["GENE_B"] == pytest.approx(0.95)


def test_parse_credible_sets_distinct_variants():
    rows = [
        _cs_row("chr1_100_A_T", "GENE_A", 0.95),
        _cs_row("chr1_200_C_G", "GENE_B", 0.005),
        _cs_row("chr2_300_G_C", "GENE_C", 0.5),
    ]
    path = _write_tsv_gz(rows, _CS_HEADER)
    out = parse_credible_sets(path).sort(["chrom", "pos"])
    assert out.height == 3
    assert set(out.columns) == {"chrom", "pos", "ref", "alt", "gene_id", "pip"}
    pips = out["pip"].to_list()
    assert pips == pytest.approx([0.95, 0.005, 0.5])


# ---------- extract_tested_variants ----------


def test_extract_tested_variants_keeps_gene_id():
    """Each (variant, gene_id) pair is one output row. MAF is constant
    across gene-rows for one variant in one tissue, so first() suffices."""
    rows = [
        _sumstats_row("chr1_100_A_T", "GENE_A", 0.015),
        _sumstats_row("chr1_100_A_T", "GENE_B", 0.015),  # same variant, diff gene
        _sumstats_row("chr1_200_C_G", "GENE_A", 0.05),
    ]
    path = _write_tsv_gz(rows, _SUMSTATS_HEADER)
    out = extract_tested_variants(path).collect().sort(["chrom", "pos", "gene_id"])
    assert out.height == 3
    assert out["maf"].to_list() == pytest.approx([0.015, 0.015, 0.05])
    assert out["gene_id"].to_list() == ["GENE_A", "GENE_B", "GENE_A"]


def test_extract_tested_variants_columns():
    """Output schema is exactly (chrom, pos, ref, alt, gene_id, maf)."""
    rows = [_sumstats_row("chr1_100_A_T", "GENE_A", 0.015)]
    path = _write_tsv_gz(rows, _SUMSTATS_HEADER)
    out = extract_tested_variants(path).collect()
    assert set(out.columns) == {"chrom", "pos", "ref", "alt", "gene_id", "maf"}


# ---------- merge_cs_and_sumstats: 0-fill semantics ----------


def _do_merge(
    cs_rows: list[list[str]],
    sumstats_rows: list[list[str]],
    tissue: str = "T",
    biotype_pairs: list[tuple[str, str]] | None = None,
    pip_pos: float = 0.9,
) -> pl.DataFrame:
    """Test helper: write CS + sumstats TSVs, call the parsers + merge."""
    cs = parse_credible_sets(_write_tsv_gz(cs_rows, _CS_HEADER))
    sumstats = extract_tested_variants(
        _write_tsv_gz(sumstats_rows, _SUMSTATS_HEADER)
    )
    return merge_cs_and_sumstats(
        cs,
        sumstats,
        tissue,
        gene_biotype_df=_gene_biotype_df(biotype_pairs),
        pip_pos_threshold=pip_pos,
    ).collect()


def test_merge_zero_fills_variants_outside_cs():
    """Variants present in sumstats but not in any CS get pip=0 (not null).
    This is the load-bearing sentinel that makes downstream label_variants_by_pip
    produce a negative label for tested-but-no-signal variants."""
    out = _do_merge(
        cs_rows=[_cs_row("chr1_100_A_T", "GENE_A", 0.95)],
        sumstats_rows=[
            _sumstats_row("chr1_100_A_T", "GENE_A", 0.015),  # in CS
            _sumstats_row("chr1_200_C_G", "GENE_A", 0.05),   # tested, not in CS
        ],
    ).sort(["chrom", "pos"])
    # After per-variant aggregation, each variant gets max(pip) across its genes.
    pip_by_variant = {r["pos"]: r["pip"] for r in out.to_dicts()}
    assert pip_by_variant[100] == pytest.approx(0.95)
    assert pip_by_variant[200] == pytest.approx(0.0)
    # Critically — pip=0.0 is a real value, not null
    assert out.filter(pl.col("pip").is_null()).height == 0


def test_merge_adds_tissue_column():
    out = _do_merge(
        cs_rows=[_cs_row("chr1_100_A_T", "G", 0.95)],
        sumstats_rows=[_sumstats_row("chr1_100_A_T", "G", 0.015)],
        tissue="muscle_skeletal",
    )
    assert (out["tissue"] == "muscle_skeletal").all()


def test_merge_preserves_maf():
    """MAF comes from sumstats (CS doesn't have MAF)."""
    out = _do_merge(
        cs_rows=[_cs_row("chr1_100_A_T", "G", 0.95)],
        sumstats_rows=[_sumstats_row("chr1_100_A_T", "G", 0.0154905)],
    )
    assert out["maf"][0] == pytest.approx(0.0154905)


def test_merge_collects_positive_genes():
    """A variant tested against 3 genes, two of which cross pip_pos: the
    `positive_genes` list contains both, sorted; the negative gene is
    excluded."""
    out = _do_merge(
        cs_rows=[
            _cs_row("chr1_100_A_T", "GENE_HIGH_1", 0.95),
            _cs_row("chr1_100_A_T", "GENE_HIGH_2", 0.92),
            _cs_row("chr1_100_A_T", "GENE_LOW", 0.005),
        ],
        sumstats_rows=[
            _sumstats_row("chr1_100_A_T", "GENE_HIGH_1", 0.015),
            _sumstats_row("chr1_100_A_T", "GENE_HIGH_2", 0.015),
            _sumstats_row("chr1_100_A_T", "GENE_LOW", 0.015),
            _sumstats_row("chr1_100_A_T", "GENE_NOT_IN_CS", 0.015),  # 0-filled
        ],
        biotype_pairs=[("GENE_HIGH_1", "pc"), ("GENE_HIGH_2", "pc"),
                       ("GENE_LOW", "nc"), ("GENE_NOT_IN_CS", "nc")],
    )
    assert out.height == 1
    row = out.to_dicts()[0]
    assert row["pip"] == pytest.approx(0.95)
    assert sorted(row["positive_genes"]) == ["GENE_HIGH_1", "GENE_HIGH_2"]


def test_merge_collects_biotype_classes_only_for_positives():
    """A variant with one pc-gene positive (pip > pip_pos) and one nc-gene
    positive should produce `positive_biotype_classes = ['nc', 'pc']`.
    Genes below pip_pos contribute neither to `positive_genes` nor to
    `positive_biotype_classes`, regardless of their biotype."""
    out = _do_merge(
        cs_rows=[
            _cs_row("chr1_100_A_T", "GENE_PC", 0.95),
            _cs_row("chr1_100_A_T", "GENE_NC", 0.93),
            _cs_row("chr1_100_A_T", "GENE_NC_LOW", 0.005),  # ignored: low pip
        ],
        sumstats_rows=[
            _sumstats_row("chr1_100_A_T", "GENE_PC", 0.015),
            _sumstats_row("chr1_100_A_T", "GENE_NC", 0.015),
            _sumstats_row("chr1_100_A_T", "GENE_NC_LOW", 0.015),
        ],
        biotype_pairs=[("GENE_PC", "pc"), ("GENE_NC", "nc"),
                       ("GENE_NC_LOW", "nc")],
    )
    row = out.to_dicts()[0]
    assert sorted(row["positive_biotype_classes"]) == ["nc", "pc"]


def test_merge_empty_positive_lists_for_pure_negative():
    """A variant whose only gene has pip < pip_pos gets empty
    positive_genes / positive_biotype_classes lists (not null)."""
    out = _do_merge(
        cs_rows=[_cs_row("chr1_100_A_T", "GENE_A", 0.005)],
        sumstats_rows=[_sumstats_row("chr1_100_A_T", "GENE_A", 0.015)],
    )
    row = out.to_dicts()[0]
    assert row["positive_genes"] == []
    assert row["positive_biotype_classes"] == []
    # And the variant's max(pip) reflects the low value
    assert row["pip"] == pytest.approx(0.005)


def test_merge_missing_gene_in_biotype_defaults_to_nc():
    """A gene present in CS but not in the gene_biotype table should get
    biotype_class='nc' (sane default for unannotated genes / lncRNAs etc.)."""
    out = _do_merge(
        cs_rows=[_cs_row("chr1_100_A_T", "ORPHAN_GENE", 0.95)],
        sumstats_rows=[_sumstats_row("chr1_100_A_T", "ORPHAN_GENE", 0.015)],
        # gene_biotype_df doesn't include ORPHAN_GENE → defaults to nc.
        biotype_pairs=[("OTHER_GENE", "pc")],
    )
    row = out.to_dicts()[0]
    assert row["positive_biotype_classes"] == ["nc"]


# ---------- Integration: end-to-end labeling via label_variants_by_pip ----------


def _make_tissue_frame(
    cs_rows: list[tuple[str, str, float]],
    sumstats_rows: list[tuple[str, str, float]],
    tissue: str,
) -> pl.DataFrame:
    """Helper: build one tissue's merged (chrom, pos, ref, alt, pip, maf,
    positive_genes, positive_biotype_classes, tissue) frame from inline
    data. All genes default to `pc` biotype."""
    cs_path = _write_tsv_gz(
        [_cs_row(v, g, pip) for v, g, pip in cs_rows], _CS_HEADER
    )
    sumstats_path = _write_tsv_gz(
        [_sumstats_row(v, g, maf) for v, g, maf in sumstats_rows],
        _SUMSTATS_HEADER,
    )
    cs = parse_credible_sets(cs_path)
    sumstats = extract_tested_variants(sumstats_path)
    return merge_cs_and_sumstats(
        cs,
        sumstats,
        tissue,
        gene_biotype_df=_gene_biotype_df(),
        pip_pos_threshold=0.9,
    ).collect()


def test_integration_sumstats_only_variant_is_negative():
    """Plan §4 example 1: variant tested in nominal sumstats but never in any
    CS → all PIPs after merge are 0 → max(pip) = 0 < 0.01 → label=False."""
    frame = _make_tissue_frame(
        cs_rows=[],  # no credible sets
        sumstats_rows=[("chr1_100_A_T", "G", 0.015)],
        tissue="t1",
    )
    out = label_variants_by_pip(
        frame,
        pip_pos_threshold=0.9,
        pip_neg_threshold=0.01,
        use_null_pip_guard=False,
    )
    rows = out.to_dicts()
    assert len(rows) == 1
    assert rows[0]["label"] is False


def test_integration_cs_positive_plus_sumstats_fill_is_positive():
    """Plan §4 example 2: variant in CS in tissue A with pip=0.95, tested
    (no CS) in tissue B with 0-fill → max = 0.95 → positive."""
    t1 = _make_tissue_frame(
        cs_rows=[("chr1_100_A_T", "G", 0.95)],
        sumstats_rows=[("chr1_100_A_T", "G", 0.015)],
        tissue="t1",
    )
    t2 = _make_tissue_frame(
        cs_rows=[],
        sumstats_rows=[("chr1_100_A_T", "G", 0.015)],
        tissue="t2",
    )
    frame = pl.concat([t1, t2])
    out = label_variants_by_pip(
        frame,
        pip_pos_threshold=0.9,
        pip_neg_threshold=0.01,
        use_null_pip_guard=False,
    )
    rows = out.to_dicts()
    assert len(rows) == 1
    assert rows[0]["label"] is True


def test_integration_cs_mid_plus_sumstats_fill_is_excluded():
    """Plan §4 example 3: variant in CS in tissue A with pip=0.5 (mid),
    tested (no CS) in tissue B with 0-fill → max = 0.5 ∈ [0.01, 0.9] →
    label=None → row filtered out. Regression test against the eqtl
    pre-filter bug (mid-PIP must NOT be silently dropped)."""
    t1 = _make_tissue_frame(
        cs_rows=[("chr1_100_A_T", "G", 0.5)],
        sumstats_rows=[("chr1_100_A_T", "G", 0.015)],
        tissue="t1",
    )
    t2 = _make_tissue_frame(
        cs_rows=[],
        sumstats_rows=[("chr1_100_A_T", "G", 0.015)],
        tissue="t2",
    )
    frame = pl.concat([t1, t2])
    out = label_variants_by_pip(
        frame,
        pip_pos_threshold=0.9,
        pip_neg_threshold=0.01,
        use_null_pip_guard=False,
    )
    assert out.height == 0


def test_integration_cs_positive_plus_cs_negative_is_positive():
    """Plan §4 example 4: variant in CS in tissue A with pip=0.95, also in
    CS in tissue B with pip=0.005 → max picks the strong signal → positive."""
    t1 = _make_tissue_frame(
        cs_rows=[("chr1_100_A_T", "G", 0.95)],
        sumstats_rows=[("chr1_100_A_T", "G", 0.015)],
        tissue="t1",
    )
    t2 = _make_tissue_frame(
        cs_rows=[("chr1_100_A_T", "G", 0.005)],
        sumstats_rows=[("chr1_100_A_T", "G", 0.015)],
        tissue="t2",
    )
    frame = pl.concat([t1, t2])
    out = label_variants_by_pip(
        frame,
        pip_pos_threshold=0.9,
        pip_neg_threshold=0.01,
        use_null_pip_guard=False,
    )
    rows = out.to_dicts()
    assert len(rows) == 1
    assert rows[0]["label"] is True


def test_integration_all_cs_negative_is_negative():
    """Variant in CS in two tissues but both PIPs < 0.01 → max < 0.01 →
    negative."""
    t1 = _make_tissue_frame(
        cs_rows=[("chr1_100_A_T", "G", 0.001)],
        sumstats_rows=[("chr1_100_A_T", "G", 0.015)],
        tissue="t1",
    )
    t2 = _make_tissue_frame(
        cs_rows=[("chr1_100_A_T", "G", 0.005)],
        sumstats_rows=[("chr1_100_A_T", "G", 0.015)],
        tissue="t2",
    )
    frame = pl.concat([t1, t2])
    out = label_variants_by_pip(
        frame,
        pip_pos_threshold=0.9,
        pip_neg_threshold=0.01,
        use_null_pip_guard=False,
    )
    rows = out.to_dicts()
    assert len(rows) == 1
    assert rows[0]["label"] is False


def test_integration_cs_mid_plus_cs_negative_is_excluded():
    """A variant that's CS-negative in one tissue and CS-mid in another
    must be excluded (max-then-filter), matching the eqtl pre-filter bug
    regression already covered in test_labeling.py — repeated here to
    confirm the Catalogue parser feeds into it correctly."""
    t1 = _make_tissue_frame(
        cs_rows=[("chr1_100_A_T", "G", 0.001)],
        sumstats_rows=[("chr1_100_A_T", "G", 0.015)],
        tissue="t1",
    )
    t2 = _make_tissue_frame(
        cs_rows=[("chr1_100_A_T", "G", 0.5)],
        sumstats_rows=[("chr1_100_A_T", "G", 0.015)],
        tissue="t2",
    )
    frame = pl.concat([t1, t2])
    out = label_variants_by_pip(
        frame,
        pip_pos_threshold=0.9,
        pip_neg_threshold=0.01,
        use_null_pip_guard=False,
    )
    assert out.height == 0
