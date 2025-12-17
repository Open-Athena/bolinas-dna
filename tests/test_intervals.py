import pandas as pd
import pytest

from bolinas.data.intervals import GenomicSet


# GenomicSet tests
def test_genomic_set_initialization_non_overlapping():
    """Test GenomicSet initialization with non-overlapping intervals.

    Input: chr1:0-50, chr1:100-150, chr2:0-200
    Output: chr1:0-50, chr1:100-150, chr2:0-200 (sorted, non-overlapping maintained)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 100, 0],
            "end": [50, 150, 200],
        }
    )
    gs = GenomicSet(data)

    # Should maintain the same intervals since they're non-overlapping, sorted
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr2"],
                "start": [0, 100, 0],
                "end": [50, 150, 200],
            }
        )
    )
    assert gs == expected


def test_genomic_set_merges_overlapping_intervals():
    """Test that GenomicSet merges overlapping intervals on initialization.

    Input: chr1:0-50, chr1:40-60, chr1:90-150
    Output: chr1:0-60, chr1:90-150 (overlapping intervals merged)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 40, 90],
            "end": [50, 60, 150],
        }
    )
    gs = GenomicSet(data)

    # Overlapping intervals [0,50] and [40,60] should merge to [0,60]
    # [90,150] should remain separate
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [0, 90],
                "end": [60, 150],
            }
        )
    )
    assert gs == expected


def test_genomic_set_merges_adjacent_intervals():
    """Test that GenomicSet merges adjacent intervals on initialization.

    Input: chr1:0-100, chr1:100-200 (adjacent: end of first equals start of second)
    Output: chr1:0-200 (adjacent intervals merged into single interval)

    Note: Adjacent intervals (where end of one equals start of the next) are merged
    because they form a continuous region.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [100, 200],
        }
    )
    gs = GenomicSet(data)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [200],
            }
        )
    )
    assert gs == expected


def test_genomic_set_union():
    """Test union operation (|) between two GenomicSets.

    Input set 1: chr1:0-50, chr1:100-150 (two separate intervals with gap 50-100)
    Input set 2: chr1:40-80, chr2:0-200
    Output: chr1:0-80, chr1:100-150, chr2:0-200 (overlaps merged)

    Note: chr1:0-50 and chr1:40-80 merge to chr1:0-80. chr1:100-150 remains separate.
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [50, 150],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [40, 0],
            "end": [80, 200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 | gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr2"],
                "start": [0, 100, 0],
                "end": [80, 150, 200],
            }
        )
    )
    assert result == expected


def test_genomic_set_intersection():
    """Test intersection operation (&) between two GenomicSets.

    Input set 1: chr1:0-50, chr1:100-150
    Input set 2: chr1:25-75, chr1:120-200
    Output: chr1:25-50, chr1:120-150 (only overlapping regions)

    Note: chr1:0-50 and chr1:100-150 are separate intervals (not merged due to gap).
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [50, 150],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [25, 120],
            "end": [75, 200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 & gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [25, 120],
                "end": [50, 150],
            }
        )
    )
    assert result == expected


def test_genomic_set_subtraction():
    """Test subtraction operation (-) between two GenomicSets.

    Input set 1: chr1:0-200 (single merged interval)
    Input set 2: chr1:40-60
    Output: chr1:0-40, chr1:60-200 (subtracted region removed, creating two intervals)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [200],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [40],
            "end": [60],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 - gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [0, 60],
                "end": [40, 200],
            }
        )
    )
    assert result == expected


def test_genomic_set_union_no_overlap():
    """Test union with completely non-overlapping intervals.

    Input set 1: chr1:0-50
    Input set 2: chr2:100-200
    Output: chr1:0-50, chr2:100-200 (all intervals included, no merging)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [50],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr2"],
            "start": [100],
            "end": [200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 | gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr2"],
                "start": [0, 100],
                "end": [50, 200],
            }
        )
    )
    assert result == expected


def test_genomic_set_intersection_no_overlap():
    """Test intersection with no overlapping intervals.

    Input set 1: chr1:0-50
    Input set 2: chr1:100-200
    Output: (empty set - no overlap)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [50],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 & gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    assert result == expected


def test_genomic_set_subtraction_no_overlap():
    """Test subtraction when there's no overlap.

    Input set 1: chr1:0-50
    Input set 2: chr1:100-200
    Output: chr1:0-50 (unchanged, no overlap to subtract)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [50],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 - gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [50],
            }
        )
    )
    assert result == expected


def test_genomic_set_different_chromosomes():
    """Test operations with intervals on different chromosomes.

    Input set 1: chr1:0-100, chr2:0-100
    Input set 2: chr1:50-150, chr3:50-150

    Union output: chr1:0-150, chr2:0-100, chr3:50-150 (all chromosomes)
    Intersection output: chr1:50-100 (only chr1 overlaps)
    Subtraction output: chr1:0-50, chr2:0-100 (chr1 overlap removed, chr2 unchanged)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 0],
            "end": [100, 100],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr3"],
            "start": [50, 50],
            "end": [150, 150],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)

    union_result = gs1 | gs2
    expected_union = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr2", "chr3"],
                "start": [0, 0, 50],
                "end": [150, 100, 150],
            }
        )
    )
    assert union_result == expected_union

    intersection_result = gs1 & gs2
    expected_intersection = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [50],
                "end": [100],
            }
        )
    )
    assert intersection_result == expected_intersection

    subtraction_result = gs1 - gs2
    expected_subtraction = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr2"],
                "start": [0, 0],
                "end": [50, 100],
            }
        )
    )
    assert subtraction_result == expected_subtraction


def test_genomic_set_to_pandas():
    """Test conversion to pandas DataFrame.

    Input: chr1:0-50, chr2:100-200
    Output: Same intervals as pandas DataFrame
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    gs = GenomicSet(data)
    df = gs.to_pandas()

    expected = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    pd.testing.assert_frame_equal(df, expected)
    # Also verify equality works
    assert gs == GenomicSet(expected)


def test_genomic_set_empty():
    """Test GenomicSet with empty DataFrame.

    Input: (empty set)
    Output: (empty set)
    """
    data = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    gs = GenomicSet(data)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    assert gs == expected


def test_genomic_set_repr():
    """Test string representation of GenomicSet.

    Input: chr1:0-100
    Output: String representation containing "GenomicSet"
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    repr_str = repr(gs)

    assert "GenomicSet" in repr_str


def test_genomic_set_adjacent_intervals():
    """Test that adjacent intervals (touching but not overlapping) are handled correctly.

    Input: chr1:0-50, chr1:50-100 (adjacent/touching intervals)
    Output: chr1:0-100 (adjacent intervals merged)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 50],
            "end": [50, 100],
        }
    )
    gs = GenomicSet(data)

    # Adjacent intervals should be merged by bioframe
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [100],
            }
        )
    )
    assert gs == expected


def test_genomic_set_single_interval():
    """Test GenomicSet with a single interval.

    Input: chr1:100-200
    Output: chr1:100-200 (unchanged)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
        }
    )
    gs = GenomicSet(data)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [100],
                "end": [200],
            }
        )
    )
    assert gs == expected


def test_genomic_set_self_union():
    """Test union of a GenomicSet with itself.

    Input: chr1:0-100 | chr1:0-100
    Output: chr1:0-100 (union with itself equals original)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    result = gs | gs

    assert result == gs


def test_genomic_set_self_intersection():
    """Test intersection of a GenomicSet with itself.

    Input: chr1:0-100 & chr1:0-100
    Output: chr1:0-100 (intersection with itself equals original)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    result = gs & gs

    assert result == gs


def test_genomic_set_self_subtraction():
    """Test subtraction of a GenomicSet from itself.

    Input: chr1:0-100 - chr1:0-100
    Output: (empty set - subtracting itself removes everything)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    result = gs - gs

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    assert result == expected


def test_genomic_set_equality():
    """Test equality comparison between GenomicSets.

    Input set 1: chr1:0-50, chr2:100-200
    Input set 2: chr1:0-50, chr2:100-200 (same as set 1)
    Input set 3: chr1:0-51, chr2:100-200 (different end)

    Tests:
    - Set 1 == Set 2: True (identical intervals)
    - Set 1 != Set 3: True (different intervals)
    - Set 1 != non-GenomicSet: True (different type)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    data3 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [51, 200],  # Different end value
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    gs3 = GenomicSet(data3)

    assert gs1 == gs2
    assert gs2 == gs1
    assert gs1 != gs3
    assert gs3 != gs1
    assert gs1 != "not a GenomicSet"
    assert gs1 != None  # noqa: E711
    assert gs1 != 123


def test_genomic_set_equality_empty():
    """Test equality with empty GenomicSets.

    Input: (empty set) == (empty set)
    Output: True (empty sets are equal)
    """
    empty1 = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    empty2 = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )

    assert empty1 == empty2
    assert empty2 == empty1


# Invalid input tests
def test_genomic_set_zero_length_interval():
    """Test that GenomicSet accepts zero-length intervals (start == end).

    Input: chr1:50-50 (start equals end, zero-length interval)
    Output: chr1:50-50 (bioframe allows zero-length intervals)

    Note: While conceptually zero-length intervals may seem invalid,
    bioframe's validation only checks that start <= end (not start < end).
    Zero-length intervals are technically valid in bedFrame format.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [50],  # start == end, zero-length interval
        }
    )

    # This should work - bioframe allows start == end
    gs = GenomicSet(data)
    result_df = gs.to_pandas()
    assert len(result_df) == 1
    assert result_df.iloc[0]["chrom"] == "chr1"
    assert result_df.iloc[0]["start"] == 50
    assert result_df.iloc[0]["end"] == 50


def test_genomic_set_invalid_start_greater_than_end():
    """Test that GenomicSet rejects intervals where start > end.

    Input: chr1:100-50 (start > end, invalid)
    Expected: ValueError (invalid bedFrame - starts exceed ends)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [50],  # start > end, invalid
        }
    )

    with pytest.raises(ValueError, match="starts exceed ends"):
        GenomicSet(data)


def test_genomic_set_negative_start():
    """Test that GenomicSet accepts negative start (bioframe allows it).

    Input: chr1:-10-50 (negative start)
    Output: chr1:-10-50 (bioframe doesn't validate non-negative starts)

    Note: While bioframe validates start < end, it doesn't enforce start >= 0.
    Negative starts are technically valid in bioframe's bedFrame format,
    even though the constraint "0 <= start < end" may be conceptually desired.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [-10],  # negative start, but valid for bioframe
            "end": [50],
        }
    )

    # This should work - bioframe doesn't reject negative starts
    gs = GenomicSet(data)
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [-10],
                "end": [50],
            }
        )
    )
    assert gs == expected


def test_genomic_set_invalid_chrom_dtype():
    """Test that GenomicSet rejects invalid chrom dtypes.

    Input: chrom column with int dtype (should be object/string/categorical)
    Expected: TypeError (invalid bedFrame - invalid column dtypes)
    """
    data = pd.DataFrame(
        {
            "chrom": [1, 2],  # int dtype instead of string
            "start": [0, 100],
            "end": [50, 200],
        }
    )

    with pytest.raises(TypeError, match="Invalid bedFrame"):
        GenomicSet(data)


def test_genomic_set_invalid_start_dtype():
    """Test that GenomicSet rejects invalid start dtype.

    Input: start column with float dtype (should be int/Int64Dtype)
    Expected: TypeError (invalid bedFrame - invalid column dtypes)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0.5, 100.0],  # float dtype instead of int
            "end": [50, 200],
        }
    )

    with pytest.raises(TypeError, match="Invalid bedFrame"):
        GenomicSet(data)


def test_genomic_set_invalid_end_dtype():
    """Test that GenomicSet rejects invalid end dtype.

    Input: end column with float dtype (should be int/Int64Dtype)
    Expected: TypeError (invalid bedFrame - invalid column dtypes)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50.5, 200.0],  # float dtype instead of int
        }
    )

    with pytest.raises(TypeError, match="Invalid bedFrame"):
        GenomicSet(data)


def test_genomic_set_invalid_string_start():
    """Test that GenomicSet rejects string values in start column.

    Input: start column with string values (should be int/Int64Dtype)
    Expected: TypeError (invalid bedFrame - invalid column dtypes)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": ["0"],  # string instead of int
            "end": [50],
        }
    )

    with pytest.raises(TypeError, match="Invalid bedFrame"):
        GenomicSet(data)


def test_genomic_set_valid_categorical_chrom():
    """Test that GenomicSet accepts categorical chrom dtype (valid).

    Input: chr1:0-50 with categorical chrom
    Output: chr1:0-50 (categorical is valid dtype for chrom)

    Note: After processing, categorical may be converted to object dtype,
    but the content should be equivalent.
    """
    data = pd.DataFrame(
        {
            "chrom": pd.Categorical(["chr1"]),
            "start": [0],
            "end": [50],
        }
    )

    gs = GenomicSet(data)
    # After merge and processing, check that we have the same intervals
    result_df = gs.to_pandas()
    assert len(result_df) == 1
    assert result_df.iloc[0]["chrom"] == "chr1"
    assert result_df.iloc[0]["start"] == 0
    assert result_df.iloc[0]["end"] == 50


# expand_min_size and add_random_shift tests
def test_genomic_set_expand_min_size_smaller_intervals():
    """Test expand_min_size with intervals smaller than min_size.

    Input: chr1:10-40 (size=30), min_size=50
    Output: chr1:0-50 (expanded equally on both sides to reach min_size=50)

    Note: Interval is expanded equally on both sides to reach min_size.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [10],
            "end": [40],  # size = 30
        }
    )
    gs = GenomicSet(data)
    result = gs.expand_min_size(min_size=50)

    # Should be expanded: pad = ceil((50-30)/2) = ceil(10) = 10
    # So start = 10-10 = 0, end = 40+10 = 50
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [50],
            }
        )
    )
    assert result == expected


def test_genomic_set_expand_min_size_larger_intervals():
    """Test expand_min_size with intervals already larger than min_size.

    Input: chr1:0-100 (size=100), min_size=50
    Output: chr1:0-100 (unchanged, already larger than min_size)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],  # size = 100, already larger than min_size
        }
    )
    gs = GenomicSet(data)
    result = gs.expand_min_size(min_size=50)

    # Should be unchanged (pad = max(ceil((50-100)/2), 0) = max(ceil(-25), 0) = 0)
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [100],
            }
        )
    )
    assert result == expected


def test_genomic_set_expand_min_size_causes_overlaps():
    """Test expand_min_size causing overlaps that get merged.

    Input: chr1:10-30, chr1:35-50 (separate, gap 30-35), min_size=30
    Output: chr1:5-58 (expanded intervals overlap and merge)

    Note: After expansion, intervals overlap and should be merged.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [10, 35],
            "end": [30, 50],  # sizes: 20, 15; gap between them
        }
    )
    gs = GenomicSet(data)
    result = gs.expand_min_size(min_size=30)

    # First: size=20, pad = ceil((30-20)/2) = 5, becomes 5-35
    # Second: size=15, pad = ceil((30-15)/2) = ceil(7.5) = 8, becomes 27-58
    # Now they overlap (35 > 27), so should merge to chr1:5-58
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [5],
                "end": [58],
            }
        )
    )
    assert result == expected


def test_genomic_set_add_random_shift_different_seeds():
    """Test add_random_shift with different seeds produces different results.

    Input: chr1:50-100, seed=42 vs seed=123
    Output: Different shifted positions for each seed
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [100],
        }
    )
    gs = GenomicSet(data)

    result1 = gs.add_random_shift(max_shift=10, seed=42)
    result2 = gs.add_random_shift(max_shift=10, seed=123)

    # Results should be different (different random shifts)
    assert result1 != result2

    # Both should have the same interval size (shift preserves size)
    df1 = result1.to_pandas()
    df2 = result2.to_pandas()
    assert len(df1) == 1
    assert len(df2) == 1
    assert (df1.iloc[0]["end"] - df1.iloc[0]["start"]) == (
        df2.iloc[0]["end"] - df2.iloc[0]["start"]
    )
    assert (df1.iloc[0]["end"] - df1.iloc[0]["start"]) == 50  # Original size preserved


def test_genomic_set_add_random_shift_same_seed():
    """Test add_random_shift with same seed produces same results.

    Input: chr1:50-100, seed=42 (twice)
    Output: Same shifted positions for both calls
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [100],
        }
    )
    gs = GenomicSet(data)

    result1 = gs.add_random_shift(max_shift=10, seed=42)
    result2 = gs.add_random_shift(max_shift=10, seed=42)

    # Results should be identical (same seed)
    assert result1 == result2


def test_genomic_set_add_random_shift_causes_overlaps():
    """Test add_random_shift causing overlaps that get merged.

    Input: chr1:50-60, chr1:61-70 (adjacent, separate), max_shift=5, seed=42
    Output: Overlapping intervals after shift get merged

    Note: Random shifts may cause intervals to overlap, which should be merged.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [50, 61],
            "end": [60, 70],  # Adjacent intervals
        }
    )
    gs = GenomicSet(data)
    result = gs.add_random_shift(max_shift=5, seed=42)

    # With max_shift=5, intervals could shift and overlap
    # The result should be a valid GenomicSet (merged if overlapping)
    result_df = result.to_pandas()
    assert len(result_df) >= 1  # May be 1 or 2 depending on shifts
    assert all(result_df["chrom"] == "chr1")


def test_genomic_set_add_random_shift_negative_start():
    """Test add_random_shift with negative start values (should be allowed).

    Input: chr1:5-15, max_shift=10, seed that causes negative shift
    Output: chr1 with potentially negative start (allowed)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [5],
            "end": [15],
        }
    )
    gs = GenomicSet(data)
    result = gs.add_random_shift(max_shift=10, seed=999)

    # Should work even if start becomes negative
    result_df = result.to_pandas()
    assert len(result_df) == 1
    assert result_df.iloc[0]["chrom"] == "chr1"
    # Size should be preserved
    assert (result_df.iloc[0]["end"] - result_df.iloc[0]["start"]) == 10


def test_genomic_set_expand_min_size_returns_new_instance():
    """Test that expand_min_size returns a new GenomicSet instance.

    Input: chr1:10-40, expand_min_size(50)
    Output: New GenomicSet, original unchanged
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [10],
            "end": [40],
        }
    )
    gs = GenomicSet(data)
    original_df = gs.to_pandas()

    result = gs.expand_min_size(min_size=50)

    # Original should be unchanged
    assert gs.to_pandas().equals(original_df)
    # Result should be different
    assert result != gs
    # Result should be a new GenomicSet
    assert isinstance(result, GenomicSet)


def test_genomic_set_add_random_shift_returns_new_instance():
    """Test that add_random_shift returns a new GenomicSet instance.

    Input: chr1:50-100, add_random_shift(10, 42)
    Output: New GenomicSet, original unchanged
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    original_df = gs.to_pandas()

    result = gs.add_random_shift(max_shift=10, seed=42)

    # Original should be unchanged
    assert gs.to_pandas().equals(original_df)
    # Result should be different (unless shift happened to be 0)
    # Result should be a new GenomicSet
    assert isinstance(result, GenomicSet)


# n_intervals and total_size tests
def test_genomic_set_n_intervals_empty():
    """Test n_intervals with empty set.

    Input: (empty set)
    Output: 0
    """
    data = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    gs = GenomicSet(data)
    assert gs.n_intervals() == 0


def test_genomic_set_n_intervals_single():
    """Test n_intervals with single interval.

    Input: chr1:0-100
    Output: 1
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    assert gs.n_intervals() == 1


def test_genomic_set_n_intervals_multiple():
    """Test n_intervals with multiple intervals.

    Input: chr1:0-50, chr1:100-150, chr2:0-200
    Output: 3
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 100, 0],
            "end": [50, 150, 200],
        }
    )
    gs = GenomicSet(data)
    assert gs.n_intervals() == 3


def test_genomic_set_n_intervals_after_merge():
    """Test n_intervals after operations that merge intervals.

    Input: chr1:0-50, chr1:40-60 (overlapping, merge to 1 interval)
    Output: 1 (merged from 2)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 40],
            "end": [50, 60],
        }
    )
    gs = GenomicSet(data)
    # Overlapping intervals should merge to 1
    assert gs.n_intervals() == 1


def test_genomic_set_total_size_empty():
    """Test total_size with empty set.

    Input: (empty set)
    Output: 0
    """
    data = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    gs = GenomicSet(data)
    assert gs.total_size() == 0


def test_genomic_set_total_size_single():
    """Test total_size with single interval.

    Input: chr1:0-100 (size=100)
    Output: 100
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    assert gs.total_size() == 100


def test_genomic_set_total_size_multiple():
    """Test total_size with multiple intervals.

    Input: chr1:0-50 (size=50), chr1:100-150 (size=50), chr2:0-200 (size=200)
    Output: 300 (sum of all sizes)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 100, 0],
            "end": [50, 150, 200],
        }
    )
    gs = GenomicSet(data)
    assert gs.total_size() == 300


def test_genomic_set_total_size_after_merge():
    """Test total_size after operations that merge intervals.

    Input: chr1:0-50 (size=50), chr1:40-60 (size=20, overlaps with first)
    Output: 60 (merged to chr1:0-60, size=60, less than 50+20=70 due to overlap removal)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 40],
            "end": [50, 60],
        }
    )
    gs = GenomicSet(data)
    # Overlapping intervals merge to chr1:0-60, size=60
    # Original sizes were 50+20=70, but overlap reduces total to 60
    assert gs.total_size() == 60
