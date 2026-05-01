"""TSS/exon nearest-feature annotations for trait-pipeline curation.

Ported from TraitGym (commit e59d612e9; src/traitgym/intervals.py).

GTF parsing reuses ``bolinas.data.utils.load_annotation`` (functionally identical
to the TraitGym version), so callers should pass an annotation DataFrame loaded
via that helper.
"""

import polars as pl
import polars_bio as pb

from bolinas.evals.variants import CHROMS, COORDINATES, NON_EXONIC


def get_tss(ann: pl.DataFrame) -> pl.DataFrame:
    """Extract transcription start sites from an annotation DataFrame.

    Returns ``[chrom, start, end, gene_id]`` 1bp intervals from protein-coding
    transcripts, deduplicated and sorted.
    """
    return (
        ann.filter(pl.col("feature") == "transcript")
        .with_columns(
            pl.col("attribute").str.extract(r'gene_id "([^;]*)";').alias("gene_id"),
            pl.col("attribute")
            .str.extract(r'transcript_biotype "([^;]*)";')
            .alias("transcript_biotype"),
        )
        .filter(pl.col("transcript_biotype") == "protein_coding")
        .with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("start"))
            .otherwise(pl.col("end") - 1)
            .alias("tss_start"),
        )
        .with_columns((pl.col("tss_start") + 1).alias("tss_end"))
        .select(
            pl.col("chrom"),
            pl.col("tss_start").alias("start"),
            pl.col("tss_end").alias("end"),
            pl.col("gene_id"),
        )
        .unique()
        .sort(["chrom", "start", "end"])
    )


def get_exon(ann: pl.DataFrame) -> pl.DataFrame:
    """Extract protein-coding exons from an annotation DataFrame.

    Returns ``[chrom, start, end, gene_id]`` for protein-coding-transcript exons,
    restricted to canonical chromosomes, deduplicated and sorted.
    """
    return (
        ann.filter(pl.col("feature") == "exon")
        .with_columns(
            pl.col("attribute").str.extract(r'gene_id "([^;]*)";').alias("gene_id"),
            pl.col("attribute")
            .str.extract(r'transcript_biotype "([^;]*)";')
            .alias("transcript_biotype"),
        )
        .filter(
            pl.col("transcript_biotype") == "protein_coding",
            pl.col("chrom").is_in(CHROMS),
        )
        .select(["chrom", "start", "end", "gene_id"])
        .unique()
        .sort(["chrom", "start", "end"])
    )


def add_exon(
    V: pl.DataFrame,
    exon: pl.DataFrame,
    exon_proximal_dist: int = 100,
) -> pl.DataFrame:
    """Attach ``exon_dist``, ``exon_closest_gene_id``, and ``consequence_final``."""
    V_intervals = V.with_columns(
        (pl.col("pos") - 1).alias("start"),
        pl.col("pos").alias("end"),
    )
    result = pb.nearest(
        V_intervals,
        exon,
        cols1=("chrom", "start", "end"),
        cols2=("chrom", "start", "end"),
        suffixes=("", "_exon"),
        output_type="polars.DataFrame",
    )
    return result.select(
        pl.exclude(
            "start",
            "end",
            "chrom_exon",
            "start_exon",
            "end_exon",
            "distance",
            "gene_id_exon",
        ),
        pl.col("distance").alias("exon_dist"),
        pl.col("gene_id_exon").alias("exon_closest_gene_id"),
        pl.when(
            pl.col("consequence") == "intron_variant",
            pl.col("distance") <= exon_proximal_dist,
        )
        .then(pl.lit("exon_proximal"))
        .otherwise(pl.col("consequence_cre"))
        .alias("consequence_final"),
    )


def add_tss(
    V: pl.DataFrame,
    tss: pl.DataFrame,
    tss_proximal_dist: int = 1000,
) -> pl.DataFrame:
    """Attach ``tss_dist``, ``tss_closest_gene_id`` and update ``consequence_final``."""
    V_intervals = V.with_columns(
        (pl.col("pos") - 1).alias("start"),
        pl.col("pos").alias("end"),
    )
    result = pb.nearest(
        V_intervals,
        tss,
        cols1=("chrom", "start", "end"),
        cols2=("chrom", "start", "end"),
        suffixes=("", "_tss"),
        output_type="polars.DataFrame",
    )
    return result.select(
        pl.exclude(
            "start",
            "end",
            "chrom_tss",
            "start_tss",
            "end_tss",
            "distance",
            "gene_id_tss",
            "consequence_final",
        ),
        pl.col("distance").alias("tss_dist"),
        pl.col("gene_id_tss").alias("tss_closest_gene_id"),
        pl.when(
            pl.col("consequence").is_in(NON_EXONIC),
            pl.col("distance") <= tss_proximal_dist,
        )
        .then(pl.lit("tss_proximal"))
        .otherwise(pl.col("consequence_final"))
        .alias("consequence_final"),
    )


def build_dataset(
    V: pl.DataFrame,
    exon: pl.DataFrame,
    tss: pl.DataFrame,
    exclude_consequences: list[str],
    exon_proximal_dist: int,
    tss_proximal_dist: int,
    consequence_groups: dict[str, list[str]],
) -> pl.DataFrame:
    """Build a dataset with final consequence annotations and groups.

    Filters out ``exclude_consequences``, attaches exon/TSS distance + final
    consequence, restricts to consequences observed in positives, and adds the
    ``consequence_group`` column. Output is sorted by COORDINATES.
    """
    V = (
        V.filter(~pl.col("consequence").is_in(exclude_consequences))
        .pipe(add_exon, exon, exon_proximal_dist)
        .pipe(add_tss, tss, tss_proximal_dist)
    )
    consequence_final_pos = V.filter("label")["consequence_final"].unique()
    V = V.filter(pl.col("consequence_final").is_in(consequence_final_pos))
    consequence_to_group = {
        c: group
        for group, consequences in consequence_groups.items()
        for c in consequences
    }
    V = V.with_columns(
        pl.col("consequence_final")
        .replace(consequence_to_group)
        .alias("consequence_group")
    ).sort(COORDINATES)
    assert V["consequence_group"].null_count() == 0
    return V
