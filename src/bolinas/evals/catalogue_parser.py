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
    """Read a Catalogue ``*.credible_sets.tsv.gz`` and return per-(variant,
    gene) max PIP.

    A (variant, gene) pair may appear in multiple credible sets (as
    multiple independent signals — ``cs_id`` suffix ``_L1``, ``_L2``, etc.).
    We take ``pl.col("pip").max()`` per (variant, gene) so the per-tissue
    PIP reflects the strongest signal across all CS participations.

    Args:
        path: filesystem path to ``*.credible_sets.tsv.gz``.

    Returns:
        ``pl.DataFrame`` with columns ``(chrom, pos, ref, alt, gene_id, pip)``.
        One row per unique ``(variant, gene_id)`` pair in the file. PIP in (0, 1].
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
        .group_by(["chrom", "pos", "ref", "alt", "gene_id"])
        .agg(pl.col("pip").max())
    )


def extract_tested_variants(path: str) -> pl.LazyFrame:
    """Stream-read a Catalogue ``*.all.tsv.gz`` nominal sumstats; return
    per-(variant, gene) rows.

    The full file is ~3.5 GB compressed (~140M rows). Each row is already
    unique per (variant, gene_id) — Catalogue's sumstats has one row per
    tested gene-variant pair, no duplicates. So we just project the
    columns we need; no group_by hash agg here (which is what previously
    OOM'd the 128 GB cluster with 8-way parallel parses — each hash table
    held ~140M entries × ~50 bytes ≈ 7 GB per worker).

    Stays lazy so the caller can chain ``sink_parquet`` without
    materializing.

    Args:
        path: filesystem path to ``*.all.tsv.gz``.

    Returns:
        ``pl.LazyFrame`` with columns ``(chrom, pos, ref, alt, gene_id, maf)``.
        ``chromosome`` is renamed to ``chrom``, ``position`` to ``pos``.
    """
    return (
        pl.scan_csv(
            path,
            separator="\t",
            schema_overrides={"chromosome": pl.String},
        )
        .rename({"chromosome": "chrom", "position": "pos"})
        .select(["chrom", "pos", "ref", "alt", "gene_id", "maf"])
    )


def merge_cs_and_sumstats(
    cs_df: pl.DataFrame,
    sumstats_lf: pl.LazyFrame,
    tissue_label: str,
    *,
    gene_biotype_df: pl.DataFrame,
    pip_pos_threshold: float,
) -> pl.LazyFrame:
    """Combine per-tissue CS + sumstats into a per-variant labeled frame
    with eGene metadata.

    Pipeline:

    1. Left-join the CS PIPs onto the sumstats inventory on
       ``(chrom, pos, ref, alt, gene_id)``. Variants × gene pairs present
       in sumstats but absent from any CS get ``pip=0`` (sentinel for
       "tested but no signal"; load-bearing — see module docstring).
    2. Left-join ``gene_biotype_df`` on ``gene_id`` to attach the gene's
       biotype class (``pc`` for protein-coding, ``nc`` otherwise). Genes
       missing from the biotype table get ``nc`` by default.
    3. Aggregate per-variant within the tissue. For each variant, collect:
       - ``max(pip)`` across the genes it was tested against,
       - ``first(maf)`` (MAF is a variant-property, constant across genes),
       - ``positive_genes``: list of gene_ids where ``pip > pip_pos_threshold``,
       - ``positive_biotype_classes``: list of unique biotype classes among
         those positive genes.

    Args:
        cs_df: output of :func:`parse_credible_sets`. Per-(variant, gene_id).
        sumstats_lf: output of :func:`extract_tested_variants`. Per-(variant,
            gene_id), lazy.
        tissue_label: e.g. ``"adipose_subcutaneous"``. Added as a column so
            downstream cross-tissue concat can carry per-tissue provenance.
        gene_biotype_df: ``(gene_id, biotype_class)`` with ``biotype_class
            ∈ {"pc", "nc"}``. Used to annotate the positive genes for the
            ``positive_biotype_classes`` aggregate.
        pip_pos_threshold: PIP cutoff for what counts as a "positive" gene
            for the ``positive_genes`` / ``positive_biotype_classes``
            aggregates. Same value as the cross-tissue ``pip_pos_threshold``
            used downstream by ``label_variants_by_pip``.

    Returns:
        ``pl.LazyFrame`` with columns ``(chrom, pos, ref, alt, pip, maf,
        positive_genes, positive_biotype_classes, tissue)``. One row per
        tested variant in the tissue. ``pip`` is ``max(pip)`` across genes
        (so 0 if the variant isn't in any CS for any gene). The two list
        columns are empty for the bulk of variants and only populated for
        those crossing ``pip_pos_threshold`` in some gene.
    """
    return (
        sumstats_lf.join(
            cs_df.lazy(), on=["chrom", "pos", "ref", "alt", "gene_id"], how="left"
        )
        .with_columns(pl.col("pip").fill_null(0.0))
        .join(gene_biotype_df.lazy(), on="gene_id", how="left")
        .with_columns(pl.col("biotype_class").fill_null("nc"))
        .group_by(["chrom", "pos", "ref", "alt"])
        .agg(
            pl.col("pip").max(),
            pl.col("maf").first(),
            pl.col("gene_id")
            .filter(pl.col("pip") > pip_pos_threshold)
            .unique()
            .sort()
            .alias("positive_genes"),
            pl.col("biotype_class")
            .filter(pl.col("pip") > pip_pos_threshold)
            .unique()
            .sort()
            .alias("positive_biotype_classes"),
        )
        .with_columns(tissue=pl.lit(tissue_label))
    )
