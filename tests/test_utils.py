import numpy as np
import pandas as pd
import polars as pl
from Bio.Seq import Seq

from bolinas.data.utils import (
    add_rc,
    get_array_split_pairs,
    get_cds,
    get_mrna_exons,
    get_promoters,
    load_annotation,
    load_fasta,
    read_bed_to_pandas,
    write_pandas_to_bed,
)


def test_get_array_split_pairs_even_division():
    """Test get_array_split_pairs with evenly divisible length.

    Input: L=10, n_shards=2
    Output: [(0, 5), (5, 10)] (each shard has size 5)
    """
    result = get_array_split_pairs(L=10, n_shards=2)
    expected = [(0, 5), (5, 10)]
    assert result == expected


def test_get_array_split_pairs_uneven_division():
    """Test get_array_split_pairs with remainder.

    Input: L=10, n_shards=3
    Output: [(0, 4), (4, 7), (7, 10)] (first shard gets extra element)

    Note: With L=10 and n_shards=3, base_size=3, remainder=1.
    First shard gets size 4, remaining shards get size 3.
    """
    result = get_array_split_pairs(L=10, n_shards=3)
    expected = [(0, 4), (4, 7), (7, 10)]
    assert result == expected


def test_get_array_split_pairs_matches_numpy():
    """Test that get_array_split_pairs matches np.array_split behavior.

    Input: L=23, n_shards=5
    Output: Slices matching np.array_split result
    """
    L = 23
    n_shards = 5
    result = get_array_split_pairs(L, n_shards)

    # Create a dummy array and split it with numpy
    arr = np.arange(L)
    np_splits = np.array_split(arr, n_shards)

    # Verify each slice matches numpy's split
    for i, (start, end) in enumerate(result):
        np.testing.assert_array_equal(arr[start:end], np_splits[i])


def test_get_array_split_pairs_single_shard():
    """Test get_array_split_pairs with n_shards=1.

    Input: L=100, n_shards=1
    Output: [(0, 100)] (single shard contains all elements)
    """
    result = get_array_split_pairs(L=100, n_shards=1)
    expected = [(0, 100)]
    assert result == expected


def test_get_array_split_pairs_many_shards():
    """Test get_array_split_pairs with n_shards equal to length.

    Input: L=5, n_shards=5
    Output: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)] (each shard has size 1)
    """
    result = get_array_split_pairs(L=5, n_shards=5)
    expected = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    assert result == expected


def test_get_array_split_pairs_more_shards_than_length():
    """Test get_array_split_pairs with n_shards > L.

    Input: L=3, n_shards=5
    Output: [(0, 1), (1, 2), (2, 3), (3, 3), (3, 3)] (last shards are empty)

    Note: When n_shards > L, some shards will be empty (start == end).
    This matches np.array_split behavior.
    """
    result = get_array_split_pairs(L=3, n_shards=5)
    expected = [(0, 1), (1, 2), (2, 3), (3, 3), (3, 3)]
    assert result == expected


def test_get_array_split_pairs_empty_array():
    """Test get_array_split_pairs with empty array (L=0).

    Input: L=0, n_shards=3
    Output: [(0, 0), (0, 0), (0, 0)] (all shards are empty)
    """
    result = get_array_split_pairs(L=0, n_shards=3)
    expected = [(0, 0), (0, 0), (0, 0)]
    assert result == expected


def test_get_array_split_pairs_covers_full_range():
    """Test that slices cover the full range without gaps or overlaps.

    Input: L=17, n_shards=4
    Output: Slices that cover [0, 17) exactly once
    """
    L = 17
    n_shards = 4
    result = get_array_split_pairs(L, n_shards)

    # First slice should start at 0
    assert result[0][0] == 0
    # Last slice should end at L
    assert result[-1][1] == L

    # Each slice end should equal next slice start (no gaps or overlaps)
    for i in range(len(result) - 1):
        assert result[i][1] == result[i + 1][0]


def test_get_array_split_pairs_correct_number_of_shards():
    """Test that the number of returned slices equals n_shards.

    Input: L=100, n_shards=7
    Output: List with exactly 7 tuples
    """
    result = get_array_split_pairs(L=100, n_shards=7)
    assert len(result) == 7


def test_get_array_split_pairs_large_values():
    """Test get_array_split_pairs with large values.

    Input: L=1000000, n_shards=13
    Output: 13 slices covering the full range
    """
    L = 1000000
    n_shards = 13
    result = get_array_split_pairs(L, n_shards)

    assert len(result) == n_shards
    assert result[0][0] == 0
    assert result[-1][1] == L

    # Verify total size
    total_size = sum(end - start for start, end in result)
    assert total_size == L


def test_get_array_split_pairs_remainder_distribution():
    """Test that remainder is distributed to first shards.

    Input: L=23, n_shards=5
    Output: First 3 shards get size 5, last 2 get size 4

    Note: base_size=4, remainder=3, so first 3 shards get size 5.
    """
    result = get_array_split_pairs(L=23, n_shards=5)
    sizes = [end - start for start, end in result]

    # First 3 shards should have size 5 (base_size + 1)
    # Last 2 shards should have size 4 (base_size)
    assert sizes == [5, 5, 5, 4, 4]


def test_get_array_split_pairs_all_equal_when_divisible():
    """Test that all shards have equal size when L is divisible by n_shards.

    Input: L=20, n_shards=4
    Output: All shards have size 5
    """
    result = get_array_split_pairs(L=20, n_shards=4)
    sizes = [end - start for start, end in result]

    # All shards should have equal size when evenly divisible
    assert all(size == 5 for size in sizes)


def test_get_array_split_pairs_consistency_with_numpy_various_inputs():
    """Test consistency with np.array_split across various inputs.

    Tests multiple combinations of L and n_shards to ensure consistent behavior.
    """
    test_cases = [
        (10, 3),
        (100, 7),
        (7, 10),
        (50, 50),
        (1, 1),
        (0, 5),
        (15, 4),
    ]

    for L, n_shards in test_cases:
        result = get_array_split_pairs(L, n_shards)
        arr = np.arange(L)
        np_splits = np.array_split(arr, n_shards)

        # Verify each slice matches numpy's split
        for i, (start, end) in enumerate(result):
            np.testing.assert_array_equal(
                arr[start:end],
                np_splits[i],
                err_msg=f"Mismatch for L={L}, n_shards={n_shards}, shard {i}",
            )


# load_annotation tests
def test_load_annotation_basic(tmp_path):
    """Test load_annotation with basic GTF file.

    Input: GTF file with 2 features on '+' and '-' strands
    Output: DataFrame with start coordinates converted to 0-based (BED)
    """
    gtf_file = tmp_path / "test.gtf"
    gtf_content = """chr1\ttest\tgene\t1000\t2000\t.\t+\t.\tgene_id "gene1"
chr1\ttest\texon\t1500\t1800\t.\t-\t.\tgene_id "gene1"
chr2\ttest\tCDS\t500\t600\t.\t+\t.\tgene_id "gene2"
"""
    gtf_file.write_text(gtf_content)

    result = load_annotation(str(gtf_file))

    # Check start coordinates are converted from 1-based to 0-based
    assert result["start"].to_list() == [999, 1499, 499]
    assert result["end"].to_list() == [2000, 1800, 600]
    assert result["strand"].to_list() == ["+", "-", "+"]
    assert len(result) == 3


def test_load_annotation_filters_invalid_strands(tmp_path):
    """Test that load_annotation filters out features without valid strand.

    Input: GTF with '.' strand (invalid)
    Output: Empty DataFrame (invalid strand filtered out)
    """
    gtf_file = tmp_path / "test.gtf"
    gtf_content = """chr1\ttest\tgene\t1000\t2000\t.\t.\t.\tgene_id "gene1"
"""
    gtf_file.write_text(gtf_content)

    result = load_annotation(str(gtf_file))

    # Invalid strand should be filtered out
    assert len(result) == 0


def test_load_annotation_handles_comments(tmp_path):
    """Test that load_annotation skips comment lines.

    Input: GTF file with comment lines starting with '#'
    Output: DataFrame without comment lines
    """
    gtf_file = tmp_path / "test.gtf"
    gtf_content = """# This is a comment
#Another comment
chr1\ttest\tgene\t1000\t2000\t.\t+\t.\tgene_id "gene1"
"""
    gtf_file.write_text(gtf_content)

    result = load_annotation(str(gtf_file))

    assert len(result) == 1
    assert result["chrom"].to_list() == ["chr1"]


# get_mrna_exons tests
def test_get_mrna_exons_with_transcript_biotype():
    """Test get_mrna_exons with transcript_biotype annotation.

    Input: Annotation with mRNA exons (using transcript_biotype)
    Output: DataFrame with only mRNA exons
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [150, 250, 350],
            "strand": ["+", "+", "+"],
            "feature": ["exon", "exon", "CDS"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'transcript_id "trans2"; transcript_biotype "lncRNA"',
                'transcript_id "trans3"; transcript_biotype "mRNA"',
            ],
            "source": ["test", "test", "test"],
            "score": [".", ".", "."],
            "frame": [".", ".", "."],
        }
    )

    result = get_mrna_exons(ann)

    # Only the first exon should be retained (mRNA exon)
    assert len(result) == 1
    assert result["transcript_id"].to_list() == ["trans1"]
    assert result["start"].to_list() == [100]


def test_get_mrna_exons_with_gbkey():
    """Test get_mrna_exons with gbkey annotation.

    Input: Annotation with mRNA exons (using gbkey)
    Output: DataFrame with only mRNA exons
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [150, 250],
            "strand": ["+", "-"],
            "feature": ["exon", "exon"],
            "attribute": [
                'transcript_id "trans1"; gbkey "mRNA"',
                'transcript_id "trans2"; gbkey "tRNA"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_mrna_exons(ann)

    # Only the first exon should be retained (mRNA via gbkey)
    assert len(result) == 1
    assert result["transcript_id"].to_list() == ["trans1"]
    assert result["strand"].to_list() == ["+"]


def test_get_mrna_exons_filters_non_exons():
    """Test that get_mrna_exons filters out non-exon features.

    Input: Annotation with CDS and gene features (no exons)
    Output: Empty DataFrame
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [150, 250],
            "strand": ["+", "+"],
            "feature": ["CDS", "gene"],
            "attribute": [
                'transcript_id "trans1"; transcript_biotype "mRNA"',
                'gene_id "gene1"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_mrna_exons(ann)

    assert len(result) == 0


# get_promoters tests
def test_get_promoters_positive_strand():
    """Test get_promoters for genes on positive strand.

    Input: Exons for gene on '+' strand, n_upstream=100, n_downstream=50
    Output: Promoter region [TSS-100, TSS+50]
    """
    exons = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1000, 1500],
            "end": [1200, 1700],
            "strand": ["+", "+"],
            "transcript_id": ["trans1", "trans1"],
        }
    )

    result = get_promoters(exons, n_upstream=100, n_downstream=50)

    # For '+' strand, TSS is at min(start) = 1000
    # Promoter is [1000-100, 1000+50] = [900, 1050]
    assert len(result) == 1
    assert result["start"].to_list() == [900]
    assert result["end"].to_list() == [1050]


def test_get_promoters_negative_strand():
    """Test get_promoters for genes on negative strand.

    Input: Exons for gene on '-' strand, n_upstream=100, n_downstream=50
    Output: Promoter region [TSS-50, TSS+100]
    """
    exons = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1000, 1500],
            "end": [1200, 1700],
            "strand": ["-", "-"],
            "transcript_id": ["trans1", "trans1"],
        }
    )

    result = get_promoters(exons, n_upstream=100, n_downstream=50)

    # For '-' strand, TSS is at max(end) = 1700
    # Promoter is [1700-50, 1700+100] = [1650, 1800]
    assert len(result) == 1
    assert result["start"].to_list() == [1650]
    assert result["end"].to_list() == [1800]


def test_get_promoters_multiple_transcripts():
    """Test get_promoters with multiple transcripts.

    Input: Exons for 2 different transcripts
    Output: 2 unique promoter regions
    """
    exons = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "start": [1000, 1500, 2000, 2500],
            "end": [1200, 1700, 2200, 2700],
            "strand": ["+", "+", "-", "-"],
            "transcript_id": ["trans1", "trans1", "trans2", "trans2"],
        }
    )

    result = get_promoters(exons, n_upstream=100, n_downstream=50)

    # Should have 2 promoter regions
    assert len(result) == 2
    assert result["chrom"].to_list() == ["chr1", "chr2"]


def test_get_promoters_deduplicates():
    """Test that get_promoters removes duplicate regions.

    Input: Two transcripts with identical promoter regions
    Output: Single unique promoter region
    """
    exons = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1000, 1000],
            "end": [1200, 1200],
            "strand": ["+", "+"],
            "transcript_id": ["trans1", "trans2"],
        }
    )

    result = get_promoters(exons, n_upstream=100, n_downstream=50)

    # Should deduplicate to 1 region
    assert len(result) == 1


# get_cds tests
def test_get_cds_basic():
    """Test get_cds with basic CDS features.

    Input: Annotation with CDS features
    Output: DataFrame with unique CDS regions [chrom, start, end]
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [100, 200, 300],
            "end": [150, 250, 350],
            "strand": ["+", "+", "-"],
            "feature": ["CDS", "CDS", "CDS"],
            "attribute": [
                'transcript_id "trans1"',
                'transcript_id "trans2"',
                'transcript_id "trans3"',
            ],
            "source": ["test", "test", "test"],
            "score": [".", ".", "."],
            "frame": [".", ".", "."],
        }
    )

    result = get_cds(ann)

    assert len(result) == 3
    assert result["chrom"].to_list() == ["chr1", "chr1", "chr2"]
    assert result["start"].to_list() == [100, 200, 300]
    assert result["end"].to_list() == [150, 250, 350]


def test_get_cds_filters_non_cds():
    """Test that get_cds filters out non-CDS features.

    Input: Annotation with exon, gene, and CDS features
    Output: DataFrame with only CDS features
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [150, 250, 350],
            "strand": ["+", "+", "+"],
            "feature": ["exon", "CDS", "gene"],
            "attribute": [
                'transcript_id "trans1"',
                'transcript_id "trans2"',
                'gene_id "gene1"',
            ],
            "source": ["test", "test", "test"],
            "score": [".", ".", "."],
            "frame": [".", ".", "."],
        }
    )

    result = get_cds(ann)

    # Only the CDS feature should be retained
    assert len(result) == 1
    assert result["start"].to_list() == [200]
    assert result["end"].to_list() == [250]


def test_get_cds_deduplicates():
    """Test that get_cds removes duplicate CDS regions.

    Input: Annotation with duplicate CDS regions
    Output: DataFrame with unique CDS regions
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 100, 200],
            "end": [150, 150, 250],
            "strand": ["+", "+", "-"],
            "feature": ["CDS", "CDS", "CDS"],
            "attribute": [
                'transcript_id "trans1"',
                'transcript_id "trans2"',
                'transcript_id "trans3"',
            ],
            "source": ["test", "test", "test"],
            "score": [".", ".", "."],
            "frame": [".", ".", "."],
        }
    )

    result = get_cds(ann)

    # Should deduplicate to 2 unique regions
    assert len(result) == 2
    assert result["start"].to_list() == [100, 200]


def test_get_cds_empty_annotation():
    """Test get_cds with annotation containing no CDS features.

    Input: Annotation with only exons
    Output: Empty DataFrame
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [150, 250],
            "strand": ["+", "+"],
            "feature": ["exon", "exon"],
            "attribute": [
                'transcript_id "trans1"',
                'transcript_id "trans2"',
            ],
            "source": ["test", "test"],
            "score": [".", "."],
            "frame": [".", "."],
        }
    )

    result = get_cds(ann)

    assert len(result) == 0


def test_get_cds_sorted():
    """Test that get_cds returns sorted output.

    Input: Annotation with unsorted CDS regions
    Output: Sorted DataFrame by chrom, start, end
    """
    ann = pl.DataFrame(
        {
            "chrom": ["chr2", "chr1", "chr1"],
            "start": [300, 200, 100],
            "end": [350, 250, 150],
            "strand": ["+", "+", "+"],
            "feature": ["CDS", "CDS", "CDS"],
            "attribute": [
                'transcript_id "trans1"',
                'transcript_id "trans2"',
                'transcript_id "trans3"',
            ],
            "source": ["test", "test", "test"],
            "score": [".", ".", "."],
            "frame": [".", ".", "."],
        }
    )

    result = get_cds(ann)

    # Should be sorted by chrom, start, end
    assert result["chrom"].to_list() == ["chr1", "chr1", "chr2"]
    assert result["start"].to_list() == [100, 200, 300]


# read_bed_to_pandas and write_pandas_to_bed tests
def test_read_bed_to_pandas(tmp_path):
    """Test read_bed_to_pandas with basic BED file.

    Input: BED file with 3 intervals
    Output: DataFrame with columns [chrom, start, end]
    """
    bed_file = tmp_path / "test.bed"
    bed_content = """chr1\t100\t200
chr2\t300\t400
chr3\t500\t600
"""
    bed_file.write_text(bed_content)

    result = read_bed_to_pandas(str(bed_file))

    assert len(result) == 3
    assert list(result.columns) == ["chrom", "start", "end"]
    assert result["chrom"].tolist() == ["chr1", "chr2", "chr3"]
    assert result["start"].tolist() == [100, 300, 500]
    assert result["end"].tolist() == [200, 400, 600]


def test_write_pandas_to_bed(tmp_path):
    """Test write_pandas_to_bed writes correct BED format.

    Input: DataFrame with intervals
    Output: BED file with tab-separated values, no header, no index
    """
    bed_file = tmp_path / "output.bed"
    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [100, 300],
            "end": [200, 400],
        }
    )

    write_pandas_to_bed(df, str(bed_file))

    # Read back and verify
    content = bed_file.read_text()
    lines = content.strip().split("\n")
    assert len(lines) == 2
    assert lines[0] == "chr1\t100\t200"
    assert lines[1] == "chr2\t300\t400"


def test_read_write_bed_roundtrip(tmp_path):
    """Test that reading and writing BED files preserves data.

    Input: DataFrame -> write to BED -> read back
    Output: Original DataFrame matches read DataFrame
    """
    bed_file = tmp_path / "roundtrip.bed"
    original = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr3"],
            "start": [100, 300, 500],
            "end": [200, 400, 600],
        }
    )

    write_pandas_to_bed(original, str(bed_file))
    result = read_bed_to_pandas(str(bed_file))

    pd.testing.assert_frame_equal(result, original)


# load_fasta tests
def test_load_fasta_basic(tmp_path):
    """Test load_fasta with basic FASTA file.

    Input: FASTA file with 2 sequences
    Output: Series with sequence IDs as index
    """
    fasta_file = tmp_path / "test.fasta"
    fasta_content = """>seq1
ATGCATGC
>seq2
GCTAGCTA
"""
    fasta_file.write_text(fasta_content)

    result = load_fasta(str(fasta_file))

    assert len(result) == 2
    assert result.name == "seq"
    assert result.index.tolist() == ["seq1", "seq2"]
    assert result["seq1"] == "ATGCATGC"
    assert result["seq2"] == "GCTAGCTA"


def test_load_fasta_multiline_sequences(tmp_path):
    """Test load_fasta with multi-line sequences.

    Input: FASTA file with sequences split across multiple lines
    Output: Series with concatenated sequences
    """
    fasta_file = tmp_path / "test.fasta"
    fasta_content = """>seq1
ATGC
ATGC
>seq2
GCTA
GCTA
"""
    fasta_file.write_text(fasta_content)

    result = load_fasta(str(fasta_file))

    assert result["seq1"] == "ATGCATGC"
    assert result["seq2"] == "GCTAGCTA"


def test_load_fasta_empty(tmp_path):
    """Test load_fasta with empty FASTA file.

    Input: Empty FASTA file
    Output: Empty Series
    """
    fasta_file = tmp_path / "empty.fasta"
    fasta_file.write_text("")

    result = load_fasta(str(fasta_file))

    assert len(result) == 0
    assert result.name == "seq"


# add_rc tests
def test_add_rc_basic():
    """Test add_rc with basic DNA sequences.

    Input: DataFrame with 1 sequence
    Output: DataFrame with 2 rows (forward and reverse complement)
    """
    df = pd.DataFrame(
        {
            "id": ["seq1"],
            "seq": ["ATGC"],
        }
    )

    result = add_rc(df)

    assert len(result) == 2
    assert result["id"].tolist() == ["seq1_+", "seq1_-"]
    assert result["seq"].tolist() == ["ATGC", "GCAT"]


def test_add_rc_multiple_sequences():
    """Test add_rc with multiple sequences.

    Input: DataFrame with 2 sequences
    Output: DataFrame with 4 rows (2 forward + 2 reverse complement)
    """
    df = pd.DataFrame(
        {
            "id": ["seq1", "seq2"],
            "seq": ["ATGC", "GGCC"],
        }
    )

    result = add_rc(df)

    assert len(result) == 4
    assert result["id"].tolist() == ["seq1_+", "seq2_+", "seq1_-", "seq2_-"]
    assert result["seq"].tolist() == ["ATGC", "GGCC", "GCAT", "GGCC"]


def test_add_rc_preserves_other_columns():
    """Test that add_rc preserves other columns in DataFrame.

    Input: DataFrame with additional columns
    Output: All columns preserved in both forward and RC
    """
    df = pd.DataFrame(
        {
            "id": ["seq1"],
            "seq": ["ATGC"],
            "quality": [30],
        }
    )

    result = add_rc(df)

    assert len(result) == 2
    assert "quality" in result.columns
    assert result["quality"].tolist() == [30, 30]


def test_add_rc_complex_sequence():
    """Test add_rc with a longer, more complex sequence.

    Input: Longer DNA sequence
    Output: Correct reverse complement
    """
    df = pd.DataFrame(
        {
            "id": ["seq1"],
            "seq": ["ATGCTAGCTAGCTA"],
        }
    )

    result = add_rc(df)

    # Expected reverse complement
    expected_rc = str(Seq("ATGCTAGCTAGCTA").reverse_complement())
    assert result[result["id"] == "seq1_-"]["seq"].iloc[0] == expected_rc


def test_add_rc_original_unchanged():
    """Test that add_rc does not modify the original DataFrame.

    Input: Original DataFrame
    Output: Original DataFrame unchanged after add_rc
    """
    df = pd.DataFrame(
        {
            "id": ["seq1"],
            "seq": ["ATGC"],
        }
    )

    original_len = len(df)
    original_id = df["id"].tolist()

    add_rc(df)

    # Original should be unchanged
    assert len(df) == original_len
    assert df["id"].tolist() == original_id
