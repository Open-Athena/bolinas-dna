"""Parsers for eQTL Catalogue r7 sumstats + credible-set files (GTEx).

Used by `eqtl.smk` per-tissue ingest. The Catalogue ships two TSV files per
fine-mapped dataset (verified empirically against ``QTD000116`` adipose
subcutaneous; April 2023 r6 release notes):

- ``*.credible_sets.tsv.gz`` — ~4 MB compressed. One row per (credible set,
  variant) pair. Columns: ``molecular_trait_id, gene_id, cs_id, variant,
  rsid, cs_size, pip, pvalue, beta, se, z, cs_min_r2, region``.
- ``*.all.tsv.gz`` — ~3.5 GB compressed (~140M rows). Full nominal
  sumstats; one row per (variant, gene) tested. Columns: ``molecular_trait_id,
  chromosome, position, ref, alt, variant, ma_samples, maf, pvalue, beta,
  se, type, ac, an, r2, molecular_trait_object_id, gene_id, median_tpm,
  rsid``.

Both files use a ``variant`` string of the form ``chr{chrom}_{pos}_{ref}_{alt}``
(no ``_b38`` suffix, unlike the Finucane single-file release). Sumstats
also exposes ``chromosome, position, ref, alt`` as separate columns, so we
don't need to parse ``variant`` there; CS only has the combined string and
needs parsing.

**The 0-fill in** ``merge_cs_and_sumstats`` is load-bearing. Variants present
in the nominal sumstats but absent from any credible set get ``pip=0`` (a
sentinel, not null). Without this, ``pl.max()`` in the downstream
``label_variants_by_pip`` cross-tissue aggregation would skip those rows
and a tested-but-no-signal variant would end up with ``max(pip) = null``
falling through to ``label=None`` → excluded. We want those variants
labeled negative — that's the whole reason we switched from the
pre-filtered Finucane source to Catalogue's full sumstats.

Per-variant MAF is constant across (variant, gene) rows in one tissue's
nominal sumstats (MAF is a property of the variant + cohort, not the
variant × gene pair). Empirically verified on ``QTD000116``: both
``ENSG00000177757`` and ``ENSG00000187583`` rows for ``chr1_13550_G_A``
show identical ``maf=0.0154905``. So MAF aggregation per variant uses
``first()``.
"""

from __future__ import annotations

import polars as pl


def _parse_variant_id(variant_col: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    """Parse a Catalogue ``variant`` string into ``(chrom, pos, ref, alt)`` exprs.

    Format: ``chr{chrom}_{pos}_{ref}_{alt}``. Indel ref/alt may have multiple
    characters (e.g. ``chr7_66367367_C_CTTAT``). Splitting on ``_`` always
    yields exactly 4 parts after stripping the ``chr`` prefix, because
    chromosomes never contain ``_``.
    """
    parts = variant_col.str.strip_prefix("chr").str.split("_")
    return (
        parts.list.get(0).alias("chrom"),
        parts.list.get(1).cast(pl.Int64, strict=False).alias("pos"),
        parts.list.get(2).alias("ref"),
        parts.list.get(3).alias("alt"),
    )


def parse_credible_sets(path: str) -> pl.DataFrame:
    """Read a Catalogue ``*.credible_sets.tsv.gz`` and return per-variant max PIP.

    A variant may appear in multiple credible sets within one tissue (across
    genes, or as multiple independent signals for the same gene — ``cs_id``
    suffix ``_L1``, ``_L2``, etc.; verified up to 12× for some multi-mapped
    indels in ``QTD000116``). We take ``pl.col("pip").max()`` per variant
    so the per-tissue PIP reflects the strongest signal across all credible
    sets the variant participated in.

    Args:
        path: filesystem path to ``*.credible_sets.tsv.gz``.

    Returns:
        ``pl.DataFrame`` with columns ``(chrom, pos, ref, alt, pip)``. One
        row per unique variant in the file. PIP is always in (0, 1].
    """
    # Schema overrides pin numeric types — important for tissues whose CS
    # file is empty (rare but happens for low-power datasets) so an empty
    # file produces a Float64-typed `pip` rather than Null, keeping
    # downstream concat / fill_null typed correctly.
    df = pl.read_csv(
        path,
        separator="\t",
        schema_overrides={"pip": pl.Float64, "pvalue": pl.Float64},
    )
    chrom, pos, ref, alt = _parse_variant_id(pl.col("variant"))
    return (
        df.with_columns(chrom, pos, ref, alt)
        .group_by(["chrom", "pos", "ref", "alt"])
        .agg(pl.col("pip").max())
    )


def extract_tested_variants(path: str) -> pl.LazyFrame:
    """Stream-read a Catalogue ``*.all.tsv.gz`` nominal sumstats; return
    per-variant MAF.

    The full file is ~3.5 GB compressed (~140M rows). Per-variant dedup
    collapses (variant, gene) rows to (variant) (~10× row reduction).
    Stays lazy so the caller can chain ``sink_parquet`` without
    materializing the full frame.

    Args:
        path: filesystem path to ``*.all.tsv.gz``.

    Returns:
        ``pl.LazyFrame`` with columns ``(chrom, pos, ref, alt, maf)``.
        ``chromosome`` is renamed to ``chrom``, ``position`` to ``pos``.
    """
    return (
        pl.scan_csv(
            path,
            separator="\t",
            schema_overrides={"chromosome": pl.String},
        )
        .rename({"chromosome": "chrom", "position": "pos"})
        .group_by(["chrom", "pos", "ref", "alt"])
        .agg(pl.col("maf").first())
    )


def merge_cs_and_sumstats(
    cs_df: pl.DataFrame,
    sumstats_lf: pl.LazyFrame,
    tissue_label: str,
) -> pl.LazyFrame:
    """Combine per-tissue CS + sumstats into a per-variant labeled frame.

    Left-join the CS PIPs onto the sumstats variant inventory on
    ``(chrom, pos, ref, alt)``. Variants present in sumstats but not in any
    credible set get ``pip=0`` (sentinel for "tested but no signal" — see
    module docstring for why this isn't null).

    Args:
        cs_df: output of :func:`parse_credible_sets`.
        sumstats_lf: output of :func:`extract_tested_variants` (lazy).
        tissue_label: e.g. ``"adipose_subcutaneous"``. Added as a column
            so downstream cross-tissue concat can carry per-tissue
            provenance.

    Returns:
        ``pl.LazyFrame`` with columns ``(chrom, pos, ref, alt, pip, maf,
        tissue)``. One row per tested variant in the tissue. PIP is 0 for
        variants outside any CS, in (0, 1] for CS members.
    """
    return sumstats_lf.join(
        cs_df.lazy(), on=["chrom", "pos", "ref", "alt"], how="left"
    ).with_columns(
        pl.col("pip").fill_null(0.0),
        tissue=pl.lit(tissue_label),
    )
