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


# ---------- parse_credible_sets ----------


def test_parse_credible_sets_single_variant():
    rows = [_cs_row("chr1_100_A_T", "ENSG00000000001", 0.95)]
    path = _write_tsv_gz(rows, _CS_HEADER)
    out = parse_credible_sets(path)
    assert out.height == 1
    r = out.to_dicts()[0]
    assert (r["chrom"], r["pos"], r["ref"], r["alt"]) == ("1", 100, "A", "T")
    assert r["pip"] == pytest.approx(0.95)


def test_parse_credible_sets_max_across_credible_sets():
    """A variant in multiple credible sets (different genes / signals) gets
    its max PIP across all memberships."""
    rows = [
        _cs_row("chr1_100_A_T", "GENE_A", 0.3, cs_index=1),
        _cs_row("chr1_100_A_T", "GENE_B", 0.95, cs_index=1),
        _cs_row("chr1_100_A_T", "GENE_A", 0.5, cs_index=2),
    ]
    path = _write_tsv_gz(rows, _CS_HEADER)
    out = parse_credible_sets(path)
    assert out.height == 1
    assert out.to_dicts()[0]["pip"] == pytest.approx(0.95)


def test_parse_credible_sets_distinct_variants():
    rows = [
        _cs_row("chr1_100_A_T", "GENE_A", 0.95),
        _cs_row("chr1_200_C_G", "GENE_B", 0.005),
        _cs_row("chr2_300_G_C", "GENE_C", 0.5),
    ]
    path = _write_tsv_gz(rows, _CS_HEADER)
    out = parse_credible_sets(path).sort(["chrom", "pos"])
    assert out.height == 3
    pips = out["pip"].to_list()
    assert pips == pytest.approx([0.95, 0.005, 0.5])


# ---------- extract_tested_variants ----------


def test_extract_tested_variants_dedupes_per_variant():
    """Sumstats has one row per (variant, gene); dedup collapses to
    one row per variant. MAF is identical across gene-rows for one variant
    in one tissue so first() is sufficient."""
    rows = [
        _sumstats_row("chr1_100_A_T", "GENE_A", 0.015),
        _sumstats_row("chr1_100_A_T", "GENE_B", 0.015),  # same variant, diff gene
        _sumstats_row("chr1_200_C_G", "GENE_A", 0.05),
    ]
    path = _write_tsv_gz(rows, _SUMSTATS_HEADER)
    out = extract_tested_variants(path).collect().sort(["chrom", "pos"])
    assert out.height == 2
    assert out["maf"].to_list() == pytest.approx([0.015, 0.05])


def test_extract_tested_variants_columns():
    """Output schema is exactly (chrom, pos, ref, alt, maf)."""
    rows = [_sumstats_row("chr1_100_A_T", "GENE_A", 0.015)]
    path = _write_tsv_gz(rows, _SUMSTATS_HEADER)
    out = extract_tested_variants(path).collect()
    assert set(out.columns) == {"chrom", "pos", "ref", "alt", "maf"}


# ---------- merge_cs_and_sumstats: 0-fill semantics ----------


def test_merge_zero_fills_variants_outside_cs():
    """Variants present in sumstats but not in any CS get pip=0 (not null).
    This is the load-bearing sentinel that makes downstream label_variants_by_pip
    produce a negative label for tested-but-no-signal variants."""
    cs = parse_credible_sets(
        _write_tsv_gz(
            [_cs_row("chr1_100_A_T", "GENE_A", 0.95)],
            _CS_HEADER,
        )
    )
    sumstats = extract_tested_variants(
        _write_tsv_gz(
            [
                _sumstats_row("chr1_100_A_T", "GENE_A", 0.015),  # in CS
                _sumstats_row("chr1_200_C_G", "GENE_A", 0.05),   # tested, not in CS
            ],
            _SUMSTATS_HEADER,
        )
    )
    out = (
        merge_cs_and_sumstats(cs, sumstats, "adipose_subcutaneous")
        .collect()
        .sort(["chrom", "pos"])
    )
    pip_by_variant = {r["pos"]: r["pip"] for r in out.to_dicts()}
    assert pip_by_variant[100] == pytest.approx(0.95)
    assert pip_by_variant[200] == pytest.approx(0.0)
    # And critically — pip=0.0 is a real value, not null
    assert out.filter(pl.col("pip").is_null()).height == 0


def test_merge_adds_tissue_column():
    cs = parse_credible_sets(
        _write_tsv_gz([_cs_row("chr1_100_A_T", "G", 0.95)], _CS_HEADER)
    )
    sumstats = extract_tested_variants(
        _write_tsv_gz([_sumstats_row("chr1_100_A_T", "G", 0.015)], _SUMSTATS_HEADER)
    )
    out = merge_cs_and_sumstats(cs, sumstats, "muscle_skeletal").collect()
    assert (out["tissue"] == "muscle_skeletal").all()


def test_merge_preserves_maf():
    """MAF comes from sumstats (CS doesn't have MAF). Should appear in
    output verbatim."""
    cs = parse_credible_sets(
        _write_tsv_gz([_cs_row("chr1_100_A_T", "G", 0.95)], _CS_HEADER)
    )
    sumstats = extract_tested_variants(
        _write_tsv_gz([_sumstats_row("chr1_100_A_T", "G", 0.0154905)], _SUMSTATS_HEADER)
    )
    out = merge_cs_and_sumstats(cs, sumstats, "T").collect()
    assert out["maf"][0] == pytest.approx(0.0154905)


# ---------- Integration: end-to-end labeling via label_variants_by_pip ----------


def _make_tissue_frame(
    cs_rows: list[tuple[str, str, float]],
    sumstats_rows: list[tuple[str, str, float]],
    tissue: str,
) -> pl.DataFrame:
    """Helper: build one tissue's merged (chrom, pos, ref, alt, pip, maf, tissue)
    frame from inline data."""
    cs_path = _write_tsv_gz(
        [_cs_row(v, g, pip) for v, g, pip in cs_rows], _CS_HEADER
    )
    sumstats_path = _write_tsv_gz(
        [_sumstats_row(v, g, maf) for v, g, maf in sumstats_rows],
        _SUMSTATS_HEADER,
    )
    cs = parse_credible_sets(cs_path)
    sumstats = extract_tested_variants(sumstats_path)
    return merge_cs_and_sumstats(cs, sumstats, tissue).collect()


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
