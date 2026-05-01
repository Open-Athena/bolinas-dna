"""Tests for variant constants and coordinate utilities."""

import polars as pl
import pytest

from bolinas.evals.variants import (
    CHROMS,
    COMPLEMENT,
    COORDINATES,
    NON_EXONIC,
    NUCLEOTIDES,
    check_ref_alt,
    filter_chroms,
    filter_snp,
    lift_hg19_to_hg38,
    reverse_complement,
)


class TestConstants:
    def test_coordinates(self) -> None:
        assert COORDINATES == ["chrom", "pos", "ref", "alt"]

    def test_nucleotides(self) -> None:
        assert NUCLEOTIDES == ["A", "C", "G", "T"]

    def test_chroms(self) -> None:
        expected = sorted([str(i) for i in range(1, 23)] + ["X", "Y"])
        assert CHROMS == expected
        assert len(CHROMS) == 24

    def test_complement(self) -> None:
        assert COMPLEMENT == {"A": "T", "T": "A", "C": "G", "G": "C"}

    def test_non_exonic(self) -> None:
        assert "intergenic_variant" in NON_EXONIC
        assert "intron_variant" in NON_EXONIC
        assert "missense_variant" not in NON_EXONIC


class TestReverseComplement:
    def test_simple(self) -> None:
        assert reverse_complement("AC") == "GT"
        assert reverse_complement("ATCG") == "CGAT"

    def test_palindrome(self) -> None:
        assert reverse_complement("GATC") == "GATC"


class TestFilterSnp:
    def test_keeps_valid_snps(self) -> None:
        df = pl.DataFrame(
            [
                ["1", 100, "A", "T"],
                ["1", 200, "C", "G"],
            ],
            schema=COORDINATES,
            orient="row",
        )
        result = filter_snp(df)
        assert result.shape[0] == 2

    def test_filters_invalid_ref(self) -> None:
        df = pl.DataFrame(
            [
                ["1", 100, "A", "T"],
                ["1", 200, "N", "G"],
            ],
            schema=COORDINATES,
            orient="row",
        )
        result = filter_snp(df)
        assert result.shape[0] == 1
        assert result["ref"][0] == "A"

    def test_filters_invalid_alt(self) -> None:
        df = pl.DataFrame(
            [
                ["1", 100, "A", "T"],
                ["1", 200, "C", "-"],
            ],
            schema=COORDINATES,
            orient="row",
        )
        result = filter_snp(df)
        assert result.shape[0] == 1
        assert result["alt"][0] == "T"

    def test_filters_indels(self) -> None:
        df = pl.DataFrame(
            [
                ["1", 100, "A", "T"],
                ["1", 200, "AT", "A"],
                ["1", 300, "G", "GC"],
            ],
            schema=COORDINATES,
            orient="row",
        )
        result = filter_snp(df)
        assert result.shape[0] == 1


class TestFilterChroms:
    def test_keeps_valid_chroms(self) -> None:
        df = pl.DataFrame(
            [
                ["1", 100, "A", "T"],
                ["22", 200, "C", "G"],
                ["X", 300, "G", "A"],
                ["Y", 400, "T", "C"],
            ],
            schema=COORDINATES,
            orient="row",
        )
        result = filter_chroms(df)
        assert result.shape[0] == 4

    def test_filters_invalid_chroms(self) -> None:
        df = pl.DataFrame(
            [
                ["1", 100, "A", "T"],
                ["MT", 200, "C", "G"],
                ["chr1", 300, "G", "A"],
                ["23", 400, "T", "C"],
            ],
            schema=COORDINATES,
            orient="row",
        )
        result = filter_chroms(df)
        assert result.shape[0] == 1
        assert result["chrom"][0] == "1"

    def test_empty_dataframe(self) -> None:
        df = pl.DataFrame(
            schema={
                "chrom": pl.String,
                "pos": pl.Int64,
                "ref": pl.String,
                "alt": pl.String,
            }
        )
        result = filter_chroms(df)
        assert result.shape[0] == 0


@pytest.mark.slow
class TestLiftHg19ToHg38:
    def test_known_lift(self) -> None:
        # rs429358 (APOE), well-known coordinate change between hg19 and hg38.
        # hg19: chr19:45411941, hg38: chr19:44908684 (same strand)
        df = pl.DataFrame(
            {
                "chrom": ["19"],
                "pos": [45411941],
                "ref": ["T"],
                "alt": ["C"],
            }
        )
        result = lift_hg19_to_hg38(df)
        assert result["chrom"][0] == "19"
        assert result["pos"][0] == 44908684
        assert result["ref"][0] == "T"
        assert result["alt"][0] == "C"

    def test_unmappable_marked(self) -> None:
        # An obviously bogus position lifts to pos=-1.
        df = pl.DataFrame(
            {"chrom": ["1"], "pos": [10**12], "ref": ["A"], "alt": ["T"]}
        )
        result = lift_hg19_to_hg38(df)
        assert result["pos"][0] == -1


class TestCheckRefAlt:
    class FakeGenome:
        def __init__(self, table: dict[tuple[str, int], str]) -> None:
            self._table = table

        def get_nuc(self, chrom: str, pos: int) -> str:
            return self._table[(chrom, pos)]

    def test_keeps_matching_ref(self) -> None:
        genome = self.FakeGenome({("1", 100): "A", ("1", 200): "G"})
        df = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [100, 200],
                "ref": ["A", "G"],
                "alt": ["T", "C"],
            }
        )
        result = check_ref_alt(df, genome)
        assert result["ref"].to_list() == ["A", "G"]
        assert result["alt"].to_list() == ["T", "C"]

    def test_swaps_when_alt_matches(self) -> None:
        genome = self.FakeGenome({("1", 100): "T"})
        df = pl.DataFrame(
            {"chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["T"]}
        )
        result = check_ref_alt(df, genome)
        assert result["ref"][0] == "T"
        assert result["alt"][0] == "A"

    def test_drops_when_neither_matches(self) -> None:
        genome = self.FakeGenome({("1", 100): "G", ("1", 200): "A"})
        df = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [100, 200],
                "ref": ["A", "A"],
                "alt": ["T", "C"],
            }
        )
        result = check_ref_alt(df, genome)
        assert result.shape[0] == 1
        assert result["pos"][0] == 200
