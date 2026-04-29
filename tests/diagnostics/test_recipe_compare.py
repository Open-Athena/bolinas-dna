from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bolinas.data.intervals import GenomicSet
from bolinas.diagnostics.recipe_compare import (
    bp_jaccard,
    classify_distal_vs_proximal,
    interval_recall,
    per_interval_bigwig_aggregates,
    softmask_fraction,
)


def _gset(rows: list[tuple[str, int, int]]) -> GenomicSet:
    df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    return GenomicSet(df)


# ---- interval_recall ----


def test_interval_recall_full_overlap():
    """Every reference interval is hit by some query interval → 1.0."""
    ref = _gset([("chr1", 0, 100), ("chr1", 200, 300), ("chr2", 0, 50)])
    query = _gset([("chr1", 50, 250), ("chr2", 25, 75)])
    assert interval_recall(query, ref) == 1.0


def test_interval_recall_partial():
    """3 of 5 reference intervals overlapped → 0.6.

    Includes the case where one query interval covers two reference intervals
    — that should still count as 2 references hit (not 1).
    """
    ref = _gset(
        [
            ("chr1", 0, 100),
            ("chr1", 150, 250),
            ("chr1", 400, 500),
            ("chr1", 600, 700),
            ("chr1", 800, 900),
        ]
    )
    query = _gset([("chr1", 50, 200), ("chr1", 450, 550)])
    assert interval_recall(query, ref) == pytest.approx(3 / 5)


def test_interval_recall_disjoint():
    ref = _gset([("chr1", 0, 100)])
    query = _gset([("chr2", 0, 100)])
    assert interval_recall(query, ref) == 0.0


def test_interval_recall_empty_reference():
    ref = _gset([])
    query = _gset([("chr1", 0, 100)])
    assert interval_recall(query, ref) == 0.0


# ---- bp_jaccard ----


def test_bp_jaccard_identical():
    a = _gset([("chr1", 0, 100), ("chr2", 50, 150)])
    assert bp_jaccard(a, a) == 1.0


def test_bp_jaccard_disjoint():
    a = _gset([("chr1", 0, 100)])
    b = _gset([("chr2", 0, 100)])
    assert bp_jaccard(a, b) == 0.0


def test_bp_jaccard_partial():
    """A=0..100, B=50..150 → ∩=50bp, ∪=150bp, J=1/3."""
    a = _gset([("chr1", 0, 100)])
    b = _gset([("chr1", 50, 150)])
    assert bp_jaccard(a, b) == pytest.approx(1 / 3)


def test_bp_jaccard_both_empty():
    assert bp_jaccard(_gset([]), _gset([])) == 0.0


# ---- classify_distal_vs_proximal ----


def test_classify_distal_midpoint_inside_promoter_is_proximal():
    intervals = _gset([("chr1", 100, 200)])  # midpoint = 150
    promoters = _gset([("chr1", 140, 160)])
    is_distal = classify_distal_vs_proximal(intervals, promoters)
    assert list(is_distal) == [False]


def test_classify_distal_midpoint_outside_promoter_is_distal():
    """Edge-overlap should NOT count — only midpoint membership."""
    intervals = _gset([("chr1", 100, 200)])  # midpoint = 150
    # Promoter overlaps the interval at its left edge but does NOT contain mid.
    promoters = _gset([("chr1", 95, 110)])
    is_distal = classify_distal_vs_proximal(intervals, promoters)
    assert list(is_distal) == [True]


def test_classify_distal_mixed():
    intervals = _gset(
        [
            ("chr1", 0, 100),  # mid 50
            ("chr1", 200, 300),  # mid 250
            ("chr1", 400, 500),  # mid 450
        ]
    )
    promoters = _gset([("chr1", 40, 60), ("chr1", 440, 460)])
    is_distal = classify_distal_vs_proximal(intervals, promoters)
    assert list(is_distal) == [False, True, False]


def test_classify_distal_empty_promoters_all_distal():
    intervals = _gset([("chr1", 0, 100), ("chr1", 200, 300)])
    promoters = _gset([])
    is_distal = classify_distal_vs_proximal(intervals, promoters)
    assert list(is_distal) == [True, True]


# ---- softmask_fraction ----


def test_softmask_fraction_counts_lowercase():
    """Mock py2bit to verify the lowercase-counting math without needing a 2bit fixture."""
    intervals = _gset([("chr1", 0, 10), ("chr1", 100, 110)])

    fake_tb = MagicMock()
    fake_tb.sequence.side_effect = ["ACGTacgtAC", "aaaaaaaaaa"]
    with patch("bolinas.diagnostics.recipe_compare.py2bit.open", return_value=fake_tb):
        result = softmask_fraction(intervals, "/fake/path.2bit")

    assert result.iloc[0] == pytest.approx(0.4)  # 4 lowercase / 10
    assert result.iloc[1] == pytest.approx(1.0)  # all lowercase
    fake_tb.close.assert_called_once()


def test_softmask_fraction_empty():
    result = softmask_fraction(_gset([]), "/fake/path.2bit")
    assert len(result) == 0


# ---- per_interval_bigwig_aggregates ----


def test_per_interval_bigwig_aggregates_basic():
    """Mock pyBigWig to verify mean + frac math without a real bigwig."""
    import numpy as np

    intervals = _gset([("chr1", 0, 4), ("chr1", 10, 14)])

    fake_bw = MagicMock()
    # Two intervals × 4 bp each. Threshold = 0.5.
    # Interval 1: [0.0, 0.6, 0.8, 0.4] → mean 0.45, 2/4 ≥ 0.5
    # Interval 2: [1.0, 1.0, 0.0, 0.0] → mean 0.5,  2/4 ≥ 0.5
    # Combined: mean = (0.0 + 0.6 + 0.8 + 0.4 + 1.0 + 1.0 + 0.0 + 0.0) / 8 = 0.475
    # frac_ge_0.5 = 4/8 = 0.5
    fake_bw.values.side_effect = [
        np.array([0.0, 0.6, 0.8, 0.4], dtype=np.float32),
        np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    ]
    with patch(
        "bolinas.diagnostics.recipe_compare.pyBigWig.open", return_value=fake_bw
    ):
        result = per_interval_bigwig_aggregates(intervals, "/fake.bw", threshold=0.5)

    assert result["mean"] == pytest.approx(0.475)
    assert result["frac_ge_threshold"] == pytest.approx(0.5)
    assert result["total_bp"] == 8
    assert result["n_unmapped"] == 0


def test_per_interval_bigwig_aggregates_handles_nan():
    """NaN positions: excluded from mean, count toward total bp but not toward ≥ threshold."""
    import numpy as np

    intervals = _gset([("chr1", 0, 4)])

    fake_bw = MagicMock()
    # 4 bp, two NaN. Non-NaN: [1.0, 0.0]. Mean = 0.5. ≥ 0.5: 1 of 4 (NaN excluded).
    fake_bw.values.side_effect = [
        np.array([1.0, np.nan, np.nan, 0.0], dtype=np.float32),
    ]
    with patch(
        "bolinas.diagnostics.recipe_compare.pyBigWig.open", return_value=fake_bw
    ):
        result = per_interval_bigwig_aggregates(intervals, "/fake.bw", threshold=0.5)

    assert result["mean"] == pytest.approx(0.5)
    # 1 bp >= 0.5; total bp = 4; frac = 0.25
    assert result["frac_ge_threshold"] == pytest.approx(0.25)
    assert result["total_bp"] == 4


def test_per_interval_bigwig_aggregates_chrom_map():
    """Intervals with chrom names absent from the map are dropped and counted."""
    import numpy as np

    intervals = _gset([("NC_000001.11", 0, 4), ("NC_999999.99", 0, 4)])

    fake_bw = MagicMock()
    fake_bw.values.side_effect = [np.array([0.5] * 4, dtype=np.float32)]
    with patch(
        "bolinas.diagnostics.recipe_compare.pyBigWig.open", return_value=fake_bw
    ):
        result = per_interval_bigwig_aggregates(
            intervals, "/fake.bw", threshold=0.5, chrom_map={"NC_000001.11": "chr1"}
        )

    assert result["n_unmapped"] == 1
    assert result["total_bp"] == 4
    fake_bw.values.assert_called_once_with("chr1", 0, 4, numpy=True)
