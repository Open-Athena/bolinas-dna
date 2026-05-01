"""Tests for feature-based sample matching utilities.

Ported from TraitGym (commit e59d612e9; tests/test_matching.py).
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from bolinas.evals.matching import (
    MATCH_GROUP_COL,
    _combine_results,
    _find_closest,
    _match_single_group,
    _scale_features,
    _sort_by_coordinates,
    _validate_columns,
    match_features,
)


def make_variants(
    n: int,
    chrom: str = "1",
    start_pos: int = 100,
    seed: int = 42,
    **extra_cols: list,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    nucs = ["A", "C", "G", "T"]
    data = {
        "chrom": [chrom] * n,
        "pos": list(range(start_pos, start_pos + n)),
        "ref": rng.choice(nucs, n).tolist(),
        "alt": rng.choice(nucs, n).tolist(),
    }
    data.update(extra_cols)
    return pl.DataFrame(data)


class TestValidateColumns:
    def test_passes_when_all_columns_present(self) -> None:
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        _validate_columns(df, ["a", "b"], "test")

    def test_raises_on_missing_columns(self) -> None:
        df = pl.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="missing columns"):
            _validate_columns(df, ["a", "c", "d"], "test")

    def test_error_message_includes_dataframe_name(self) -> None:
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="my_df is missing"):
            _validate_columns(df, ["a", "b"], "my_df")


class TestScaleFeatures:
    def test_returns_scaled_columns(self) -> None:
        pos = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [10, 20, 30]})
        neg = pd.DataFrame({"feat1": [4, 5, 6], "feat2": [40, 50, 60]})
        pos_scaled, neg_scaled, scaled_cols = _scale_features(
            pos, neg, ["feat1", "feat2"]
        )

        assert scaled_cols == ["feat1_scaled", "feat2_scaled"]
        assert "feat1_scaled" in pos_scaled.columns
        assert "feat2_scaled" in neg_scaled.columns

    def test_preserves_original_columns(self) -> None:
        pos = pd.DataFrame({"feat1": [1, 2, 3], "other": ["a", "b", "c"]})
        neg = pd.DataFrame({"feat1": [4, 5, 6], "other": ["d", "e", "f"]})
        pos_scaled, _, _ = _scale_features(pos, neg, ["feat1"])

        assert "feat1" in pos_scaled.columns
        assert "other" in pos_scaled.columns
        pd.testing.assert_series_equal(pos_scaled["feat1"], pos["feat1"])

    def test_robust_scaling_uses_median_and_iqr(self) -> None:
        pos = pd.DataFrame({"feat1": [0.0, 1.0, 2.0]})
        neg = pd.DataFrame({"feat1": [3.0, 4.0, 5.0]})
        pos_scaled, _, _ = _scale_features(pos, neg, ["feat1"])

        all_original = np.array([0, 1, 2, 3, 4, 5])
        median = np.median(all_original)
        q1, q3 = np.percentile(all_original, [25, 75])
        iqr = q3 - q1
        expected_pos = (np.array([0, 1, 2]) - median) / iqr
        np.testing.assert_array_almost_equal(
            pos_scaled["feat1_scaled"].values, expected_pos
        )


class TestFindClosest:
    def test_finds_k_closest(self) -> None:
        pos = pd.DataFrame({"x": [0.0], "y": [0.0]})
        neg = pd.DataFrame(
            {
                "x": [1.0, 2.0, 10.0, 3.0],
                "y": [1.0, 2.0, 10.0, 3.0],
                "id": ["a", "b", "c", "d"],
            }
        )
        result = _find_closest(pos, neg, ["x", "y"], k=2)

        assert len(result) == 2
        assert set(result["id"].tolist()) == {"a", "b"}

    def test_no_replacement_across_positives(self) -> None:
        pos = pd.DataFrame({"x": [0.0, 0.1]})
        neg = pd.DataFrame({"x": [0.0, 0.2, 0.3, 0.4], "id": [0, 1, 2, 3]})
        result = _find_closest(pos, neg, ["x"], k=2)

        assert len(result) == 4
        assert len(set(result["id"].tolist())) == 4

    def test_euclidean_distance_used(self) -> None:
        pos = pd.DataFrame({"x": [0.0], "y": [0.0]})
        neg = pd.DataFrame(
            {
                "x": [3.0, 0.0],
                "y": [0.0, 4.0],
                "id": ["three", "four"],
            }
        )
        result = _find_closest(pos, neg, ["x", "y"], k=1)

        assert result["id"].iloc[0] == "three"


class TestMatchSingleGroup:
    def test_matches_within_group(self) -> None:
        pos = pd.DataFrame({"cat": ["A"], "feat": [1.0]}).set_index("cat")
        neg = pd.DataFrame({"cat": ["A", "A"], "feat": [1.1, 2.0]}).set_index("cat")
        result = _match_single_group(pos, neg, "A", ["feat"], k=1, seed=42)

        assert result is not None
        pos_out, neg_out = result
        assert len(pos_out) == 1
        assert len(neg_out) == 1

    def test_warns_and_subsamples_on_insufficient_negatives(self) -> None:
        pos = pd.DataFrame({"cat": ["A"] * 5, "feat": [1.0] * 5}).set_index("cat")
        neg = pd.DataFrame({"cat": ["A"] * 3, "feat": [1.0, 2.0, 3.0]}).set_index("cat")

        with pytest.warns(UserWarning, match="Insufficient negatives"):
            result = _match_single_group(pos, neg, "A", ["feat"], k=2, seed=42)

        assert result is not None
        pos_out, neg_out = result
        assert len(pos_out) == 1
        assert len(neg_out) == 2

    def test_random_sampling_when_no_continuous_features(self) -> None:
        pos = pd.DataFrame({"cat": ["A"] * 2}).set_index("cat")
        neg = pd.DataFrame({"cat": ["A"] * 10, "id": list(range(10))}).set_index("cat")
        result = _match_single_group(pos, neg, "A", [], k=3, seed=42)

        assert result is not None
        pos_out, neg_out = result
        assert len(neg_out) == 6


class TestCombineResults:
    def test_assigns_match_group(self) -> None:
        pos1 = pd.DataFrame({"x": [1, 2]})
        neg1 = pd.DataFrame({"x": [10, 11, 12, 13]})
        result = _combine_results([pos1], [neg1], k=2)

        assert MATCH_GROUP_COL in result.columns
        assert set(result[MATCH_GROUP_COL].unique()) == {0, 1}
        assert (result[MATCH_GROUP_COL] == 0).sum() == 3
        assert (result[MATCH_GROUP_COL] == 1).sum() == 3

    def test_empty_lists_return_empty_dataframe(self) -> None:
        result = _combine_results([], [], k=2)
        assert len(result) == 0

    def test_multiple_groups_combined(self) -> None:
        pos1 = pd.DataFrame({"x": [1]})
        pos2 = pd.DataFrame({"x": [2]})
        neg1 = pd.DataFrame({"x": [10, 11]})
        neg2 = pd.DataFrame({"x": [20, 21]})
        result = _combine_results([pos1, pos2], [neg1, neg2], k=2)

        assert len(result) == 6


class TestSortByCoordinates:
    def test_sorts_by_chrom_pos_ref_alt(self) -> None:
        df = pl.DataFrame(
            {
                "chrom": ["2", "1", "1"],
                "pos": [100, 200, 100],
                "ref": ["A", "A", "A"],
                "alt": ["T", "T", "G"],
            }
        )
        result = _sort_by_coordinates(df)

        assert result["chrom"].to_list() == ["1", "1", "2"]
        assert result["pos"].to_list() == [100, 200, 100]
        assert result["alt"].to_list() == ["G", "T", "T"]


class TestMatchFeatures:
    def test_output_has_match_group_column(self) -> None:
        pos = make_variants(3, category=["A", "A", "A"], feat=[1.0, 2.0, 3.0])
        neg = make_variants(
            9, chrom="1", start_pos=200, category=["A"] * 9, feat=list(range(9))
        )
        result = match_features(pos, neg, ["feat"], ["category"], k=2)

        assert MATCH_GROUP_COL in result.columns

    def test_correct_row_count(self) -> None:
        n_pos = 4
        k = 3
        pos = make_variants(
            n_pos, category=["A"] * n_pos, feat=[float(i) for i in range(n_pos)]
        )
        neg = make_variants(
            n_pos * k,
            chrom="1",
            start_pos=200,
            category=["A"] * (n_pos * k),
            feat=list(range(n_pos * k)),
        )
        result = match_features(pos, neg, ["feat"], ["category"], k=k)

        assert len(result) == n_pos + n_pos * k

    def test_sorted_by_coordinates(self) -> None:
        pos = pl.DataFrame(
            {
                "chrom": ["2", "1"],
                "pos": [100, 50],
                "ref": ["A", "C"],
                "alt": ["T", "G"],
                "cat": ["A", "A"],
                "feat": [1.0, 2.0],
            }
        )
        neg = pl.DataFrame(
            {
                "chrom": ["1", "1", "2", "2"],
                "pos": [60, 70, 110, 120],
                "ref": ["A", "A", "A", "A"],
                "alt": ["T", "T", "T", "T"],
                "cat": ["A", "A", "A", "A"],
                "feat": [2.1, 2.2, 1.1, 1.2],
            }
        )
        result = match_features(pos, neg, ["feat"], ["cat"], k=2)

        chroms = result["chrom"].to_list()
        assert chroms[0] == "1"

    def test_reproducibility_with_seed(self) -> None:
        pos = make_variants(2, category=["A", "A"], feat=[1.0, 2.0])
        neg = make_variants(
            10, chrom="1", start_pos=200, category=["A"] * 10, feat=list(range(10))
        )

        result1 = match_features(pos, neg, ["feat"], ["category"], k=2, seed=123)
        result2 = match_features(pos, neg, ["feat"], ["category"], k=2, seed=123)

        assert result1.equals(result2)

    def test_different_seeds_give_different_results(self) -> None:
        pos = make_variants(2, category=["A", "A"])
        neg = make_variants(20, chrom="1", start_pos=200, category=["A"] * 20)

        result1 = match_features(pos, neg, [], ["category"], k=3, seed=1)
        result2 = match_features(pos, neg, [], ["category"], k=3, seed=2)

        assert not result1.equals(result2)

    def test_closest_matching_behavior(self) -> None:
        pos = pl.DataFrame(
            {
                "chrom": ["1"],
                "pos": [100],
                "ref": ["A"],
                "alt": ["T"],
                "cat": ["A"],
                "feat": [5.0],
            }
        )
        neg = pl.DataFrame(
            {
                "chrom": ["1", "1", "1"],
                "pos": [200, 300, 400],
                "ref": ["A", "A", "A"],
                "alt": ["T", "T", "T"],
                "cat": ["A", "A", "A"],
                "feat": [5.1, 100.0, 5.2],
            }
        )
        result = match_features(pos, neg, ["feat"], ["cat"], k=2)

        matched_positions = result["pos"].to_list()
        assert 100 in matched_positions
        assert 200 in matched_positions
        assert 400 in matched_positions
        assert 300 not in matched_positions


class TestMatchFeaturesEdgeCases:
    def test_empty_continuous_features_does_random_sampling(self) -> None:
        pos = make_variants(2, category=["A", "A"])
        neg = make_variants(10, chrom="1", start_pos=200, category=["A"] * 10)
        result = match_features(pos, neg, [], ["category"], k=2)

        assert len(result) == 2 + 2 * 2

    def test_multiple_categorical_features(self) -> None:
        pos = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [100, 200],
                "ref": ["A", "A"],
                "alt": ["T", "T"],
                "cat1": ["A", "B"],
                "cat2": ["X", "Y"],
                "feat": [1.0, 2.0],
            }
        )
        neg = pl.DataFrame(
            {
                "chrom": ["1", "1", "1", "1"],
                "pos": [300, 400, 500, 600],
                "ref": ["A", "A", "A", "A"],
                "alt": ["T", "T", "T", "T"],
                "cat1": ["A", "A", "B", "B"],
                "cat2": ["X", "X", "Y", "Y"],
                "feat": [1.1, 1.2, 2.1, 2.2],
            }
        )
        result = match_features(pos, neg, ["feat"], ["cat1", "cat2"], k=2)

        assert len(result) == 2 + 2 * 2

    def test_warns_on_insufficient_negatives(self) -> None:
        pos = make_variants(5, category=["A"] * 5, feat=[float(i) for i in range(5)])
        neg = make_variants(
            3, chrom="1", start_pos=200, category=["A"] * 3, feat=[0.0, 1.0, 2.0]
        )

        with pytest.warns(UserWarning, match="Insufficient negatives"):
            result = match_features(pos, neg, ["feat"], ["category"], k=2)

        assert len(result) == 1 + 1 * 2

    def test_warns_on_missing_category(self) -> None:
        pos = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [100, 200],
                "ref": ["A", "A"],
                "alt": ["T", "T"],
                "cat": ["A", "B"],
                "feat": [1.0, 2.0],
            }
        )
        neg = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [300, 400],
                "ref": ["A", "A"],
                "alt": ["T", "T"],
                "cat": ["A", "A"],
                "feat": [1.1, 1.2],
            }
        )

        with pytest.warns(UserWarning, match="No negatives found"):
            result = match_features(pos, neg, ["feat"], ["cat"], k=2)

        assert len(result) == 1 + 1 * 2

    def test_missing_columns_raises_value_error(self) -> None:
        pos = pl.DataFrame({"chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["T"]})
        neg = pos.clone()

        with pytest.raises(ValueError, match="pos is missing columns"):
            match_features(pos, neg, ["missing_feat"], [], k=1)

    def test_missing_coordinates_raises_value_error(self) -> None:
        pos = pl.DataFrame({"chrom": ["1"], "pos": [100]})
        neg = pos.clone()

        with pytest.raises(ValueError, match="missing columns"):
            match_features(pos, neg, [], [], k=1)

    def test_scale_false_does_not_scale(self) -> None:
        pos = pl.DataFrame(
            {
                "chrom": ["1"],
                "pos": [100],
                "ref": ["A"],
                "alt": ["T"],
                "cat": ["A"],
                "feat": [1000.0],
            }
        )
        neg = pl.DataFrame(
            {
                "chrom": ["1", "1"],
                "pos": [200, 300],
                "ref": ["A", "A"],
                "alt": ["T", "T"],
                "cat": ["A", "A"],
                "feat": [1001.0, 1002.0],
            }
        )
        result = match_features(pos, neg, ["feat"], ["cat"], k=1, scale=False)

        assert len(result) == 2
        assert "_scaled" not in str(result.columns)
