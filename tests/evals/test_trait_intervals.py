"""Tests for TraitGym-style TSS/exon nearest-feature annotations."""

from pathlib import Path

import polars as pl
import pytest

from bolinas.data.utils import load_annotation
from bolinas.evals.trait_intervals import (
    add_exon,
    add_tss,
    build_dataset,
    get_exon,
    get_tss,
)


SYNTHETIC_GTF = """\
1\thavana\ttranscript\t1001\t2000\t.\t+\t.\tgene_id "ENSG_PLUS"; transcript_id "T1"; transcript_biotype "protein_coding";
1\thavana\texon\t1001\t1100\t.\t+\t.\tgene_id "ENSG_PLUS"; transcript_id "T1"; transcript_biotype "protein_coding";
1\thavana\texon\t1500\t1600\t.\t+\t.\tgene_id "ENSG_PLUS"; transcript_id "T1"; transcript_biotype "protein_coding";
1\thavana\texon\t1900\t2000\t.\t+\t.\tgene_id "ENSG_PLUS"; transcript_id "T1"; transcript_biotype "protein_coding";
1\thavana\ttranscript\t3000\t4000\t.\t-\t.\tgene_id "ENSG_MINUS"; transcript_id "T2"; transcript_biotype "protein_coding";
1\thavana\texon\t3000\t3200\t.\t-\t.\tgene_id "ENSG_MINUS"; transcript_id "T2"; transcript_biotype "protein_coding";
1\thavana\texon\t3800\t4000\t.\t-\t.\tgene_id "ENSG_MINUS"; transcript_id "T2"; transcript_biotype "protein_coding";
1\thavana\ttranscript\t5000\t6000\t.\t+\t.\tgene_id "ENSG_LNCRNA"; transcript_id "T3"; transcript_biotype "lncRNA";
1\thavana\texon\t5000\t6000\t.\t+\t.\tgene_id "ENSG_LNCRNA"; transcript_id "T3"; transcript_biotype "lncRNA";
MT\thavana\ttranscript\t10\t100\t.\t+\t.\tgene_id "ENSG_MT"; transcript_id "T4"; transcript_biotype "protein_coding";
MT\thavana\texon\t10\t100\t.\t+\t.\tgene_id "ENSG_MT"; transcript_id "T4"; transcript_biotype "protein_coding";
"""


@pytest.fixture
def annotation(tmp_path: Path) -> pl.DataFrame:
    gtf = tmp_path / "synthetic.gtf"
    gtf.write_text(SYNTHETIC_GTF)
    return load_annotation(str(gtf))


class TestGetTss:
    def test_output_schema(self, annotation: pl.DataFrame) -> None:
        tss = get_tss(annotation)
        assert tss.columns == ["chrom", "start", "end", "gene_id"]

    def test_tss_is_1bp(self, annotation: pl.DataFrame) -> None:
        tss = get_tss(annotation)
        assert (tss["end"] - tss["start"] == 1).all()

    def test_plus_strand_tss(self, annotation: pl.DataFrame) -> None:
        tss = get_tss(annotation)
        # 1-based GTF transcript starting at 1001 → 0-based start = 1000
        plus_tss = tss.filter(pl.col("gene_id") == "ENSG_PLUS")
        assert plus_tss["start"][0] == 1000
        assert plus_tss["end"][0] == 1001

    def test_minus_strand_tss(self, annotation: pl.DataFrame) -> None:
        tss = get_tss(annotation)
        # 1-based GTF transcript ending at 4000 → 0-based end = 4000;
        # minus-strand TSS = end - 1 = 3999
        minus_tss = tss.filter(pl.col("gene_id") == "ENSG_MINUS")
        assert minus_tss["start"][0] == 3999
        assert minus_tss["end"][0] == 4000

    def test_skips_non_protein_coding(self, annotation: pl.DataFrame) -> None:
        tss = get_tss(annotation)
        gene_ids = set(tss["gene_id"].to_list())
        assert "ENSG_LNCRNA" not in gene_ids


class TestGetExon:
    def test_output_schema(self, annotation: pl.DataFrame) -> None:
        exon = get_exon(annotation)
        assert exon.columns == ["chrom", "start", "end", "gene_id"]

    def test_filters_protein_coding(self, annotation: pl.DataFrame) -> None:
        exon = get_exon(annotation)
        gene_ids = set(exon["gene_id"].to_list())
        assert "ENSG_LNCRNA" not in gene_ids

    def test_filters_canonical_chroms(self, annotation: pl.DataFrame) -> None:
        exon = get_exon(annotation)
        # MT not in canonical list
        assert "MT" not in exon["chrom"].unique().to_list()

    def test_deduplicates(self, annotation: pl.DataFrame) -> None:
        exon = get_exon(annotation)
        assert exon.shape[0] == exon.unique().shape[0]


class TestAddExon:
    @pytest.fixture
    def variants(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "chrom": ["1", "1", "1"],
                "pos": [1150, 1700, 50000],
                "ref": ["A", "A", "A"],
                "alt": ["T", "T", "T"],
                # 1150: 50bp downstream of exon 1001-1100 (intron)
                # 1700: 100bp downstream of exon 1500-1600 (intron)
                # 50000: far away (intergenic)
                "consequence": [
                    "intron_variant",
                    "intron_variant",
                    "intergenic_variant",
                ],
                "consequence_cre": [
                    "intron_variant",
                    "intron_variant",
                    "intergenic_variant",
                ],
            }
        )

    def test_output_columns(
        self, variants: pl.DataFrame, annotation: pl.DataFrame
    ) -> None:
        result = add_exon(variants, get_exon(annotation))
        assert "exon_dist" in result.columns
        assert "exon_closest_gene_id" in result.columns
        assert "consequence_final" in result.columns

    def test_exon_proximal_within_threshold(
        self, variants: pl.DataFrame, annotation: pl.DataFrame
    ) -> None:
        result = add_exon(variants, get_exon(annotation), exon_proximal_dist=100)
        # 1150 is 50bp away from exon 1001-1100 → exon_proximal
        assert (
            result.filter(pl.col("pos") == 1150)["consequence_final"][0]
            == "exon_proximal"
        )

    def test_exon_proximal_outside_threshold(
        self, variants: pl.DataFrame, annotation: pl.DataFrame
    ) -> None:
        result = add_exon(variants, get_exon(annotation), exon_proximal_dist=10)
        # 1150 is 50bp away → not within 10bp → falls back to consequence_cre
        assert (
            result.filter(pl.col("pos") == 1150)["consequence_final"][0]
            == "intron_variant"
        )

    def test_non_intron_not_modified(
        self, variants: pl.DataFrame, annotation: pl.DataFrame
    ) -> None:
        result = add_exon(variants, get_exon(annotation), exon_proximal_dist=10**6)
        assert (
            result.filter(pl.col("pos") == 50000)["consequence_final"][0]
            == "intergenic_variant"
        )


class TestAddTss:
    @pytest.fixture
    def variants_with_exon(self, annotation: pl.DataFrame) -> pl.DataFrame:
        V = pl.DataFrame(
            {
                "chrom": ["1", "1", "1"],
                "pos": [1001, 50000, 4000],
                "ref": ["A", "A", "A"],
                "alt": ["T", "T", "T"],
                # 1001: at the plus-strand TSS (intron context)
                # 50000: far (intergenic)
                # 4000: at minus-strand TSS (upstream_gene)
                "consequence": [
                    "intron_variant",
                    "intergenic_variant",
                    "upstream_gene_variant",
                ],
                "consequence_cre": [
                    "intron_variant",
                    "intergenic_variant",
                    "upstream_gene_variant",
                ],
            }
        )
        return add_exon(V, get_exon(annotation), exon_proximal_dist=10)

    def test_output_columns(
        self, variants_with_exon: pl.DataFrame, annotation: pl.DataFrame
    ) -> None:
        result = add_tss(variants_with_exon, get_tss(annotation))
        assert "tss_dist" in result.columns
        assert "tss_closest_gene_id" in result.columns
        assert "consequence_final" in result.columns

    def test_tss_proximal_at_tss(
        self, variants_with_exon: pl.DataFrame, annotation: pl.DataFrame
    ) -> None:
        result = add_tss(variants_with_exon, get_tss(annotation), tss_proximal_dist=1000)
        # upstream_gene_variant at 4000 sits at the minus-strand TSS → tss_proximal
        assert (
            result.filter(pl.col("pos") == 4000)["consequence_final"][0]
            == "tss_proximal"
        )

    def test_intergenic_far_unchanged(
        self, variants_with_exon: pl.DataFrame, annotation: pl.DataFrame
    ) -> None:
        result = add_tss(variants_with_exon, get_tss(annotation), tss_proximal_dist=100)
        assert (
            result.filter(pl.col("pos") == 50000)["consequence_final"][0]
            == "intergenic_variant"
        )


class TestBuildDataset:
    def test_end_to_end(self, annotation: pl.DataFrame) -> None:
        V = pl.DataFrame(
            {
                "chrom": ["1", "1", "1", "1"],
                "pos": [1050, 4000, 50000, 1150],
                "ref": ["A", "A", "A", "A"],
                "alt": ["T", "T", "T", "T"],
                "label": [True, True, False, True],
                # 1050 sits inside exon 1001-1100 (missense_variant)
                # 4000 is at minus-strand TSS (upstream_gene → tss_proximal)
                # 50000 is intergenic (negative, will be filtered — not seen in pos)
                # 1150 is intron 50bp from exon (intron → exon_proximal at 100bp)
                "consequence": [
                    "missense_variant",
                    "upstream_gene_variant",
                    "intergenic_variant",
                    "intron_variant",
                ],
                "consequence_cre": [
                    "missense_variant",
                    "upstream_gene_variant",
                    "intergenic_variant",
                    "intron_variant",
                ],
            }
        )
        consequence_groups = {
            "Missense": ["missense_variant"],
            "Promoter": ["tss_proximal"],
            "Splicing": ["exon_proximal"],
            "Distal": ["intergenic_variant", "intron_variant", "upstream_gene_variant"],
        }
        out = build_dataset(
            V,
            exon=get_exon(annotation),
            tss=get_tss(annotation),
            exclude_consequences=[],
            exon_proximal_dist=100,
            # tight tss_proximal_dist so 1150 (~150bp from TSS) doesn't become tss_proximal
            tss_proximal_dist=10,
            consequence_groups=consequence_groups,
        )

        # 50000 is intergenic — its consequence_final is intergenic_variant, which
        # never appears in label=True positives, so it gets filtered out by build_dataset.
        positions = set(out["pos"].to_list())
        assert 50000 not in positions
        # The three positives/negatives whose final consequences match should remain.
        assert {1050, 4000, 1150} <= positions

        # consequence_group must be populated
        assert out["consequence_group"].null_count() == 0

        # 1050 → missense_variant; 4000 → tss_proximal (exact TSS match); 1150 → exon_proximal
        cg = dict(zip(out["pos"].to_list(), out["consequence_group"].to_list()))
        assert cg[1050] == "Missense"
        assert cg[4000] == "Promoter"
        assert cg[1150] == "Splicing"

    def test_excludes_consequences(self, annotation: pl.DataFrame) -> None:
        V = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [1050, 1500],
                "ref": ["A", "A"],
                "alt": ["T", "T"],
                "label": [True, True],
                "consequence": ["missense_variant", "stop_gained"],
                "consequence_cre": ["missense_variant", "stop_gained"],
            }
        )
        out = build_dataset(
            V,
            exon=get_exon(annotation),
            tss=get_tss(annotation),
            exclude_consequences=["stop_gained"],
            exon_proximal_dist=100,
            tss_proximal_dist=1000,
            consequence_groups={"Missense": ["missense_variant"]},
        )
        assert 1500 not in out["pos"].to_list()
