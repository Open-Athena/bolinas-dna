"""Tests for the conservation-score helpers (issue #146)."""

import numpy as np
import pandas as pd
import pyBigWig
import pytest

from bolinas.evals.conservation import (
    CONSERVATION_TRACKS,
    aggregate_traitgym_metrics,
    score_variants_at_positions,
)


def test_conservation_tracks_keys():
    """The three expected tracks are present and URLs look like bigWigs."""
    assert set(CONSERVATION_TRACKS.keys()) == {
        "phyloP_100v",
        "phyloP_241m",
        "phastCons_43p",
    }
    for name, url in CONSERVATION_TRACKS.items():
        assert url.startswith("https://"), f"{name}: URL must be https"
        assert url.endswith((".bw", ".bigWig")), (
            f"{name}: URL must point to a bigWig file, got {url}"
        )


def _write_tiny_bigwig(path, entries, chrom_size=1000):
    """Write a minimal bigWig with the given (chrom, start, end, value) entries."""
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([("chr1", chrom_size)])
    chroms, starts, ends, values = zip(*entries)
    bw.addEntries(list(chroms), list(starts), ends=list(ends), values=list(values))
    bw.close()


def test_score_variants_coordinate_conversion(tmp_path):
    """1-based VCF pos -> 0-based half-open bigWig query.

    bigWig has values at 0-based positions 99 and 199 only; everything
    else should be NaN.
    """
    bw_path = tmp_path / "tiny.bw"
    _write_tiny_bigwig(
        bw_path,
        [
            ("chr1", 99, 100, 1.5),
            ("chr1", 199, 200, 2.5),
        ],
    )

    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "pos": [100, 200, 300, 500],  # 1-based
        }
    )

    scores = score_variants_at_positions(df, bw_path)

    assert scores.shape == (4,)
    assert scores[0] == pytest.approx(1.5)
    assert scores[1] == pytest.approx(2.5)
    assert np.isnan(scores[2])
    assert np.isnan(scores[3])


def test_score_variants_preserves_nan(tmp_path):
    """NaN must be preserved (not silently filled) when no data is present."""
    bw_path = tmp_path / "tiny.bw"
    _write_tiny_bigwig(bw_path, [("chr1", 0, 1, 0.0)])

    df = pd.DataFrame({"chrom": ["chr1"], "pos": [500]})  # outside the entry
    scores = score_variants_at_positions(df, bw_path)
    assert np.isnan(scores[0])


def test_score_variants_chrom_prefix_handling(tmp_path):
    """Chromosome names without 'chr' prefix should be auto-prefixed."""
    bw_path = tmp_path / "tiny.bw"
    _write_tiny_bigwig(bw_path, [("chr1", 99, 100, 3.14)])

    df = pd.DataFrame({"chrom": ["1"], "pos": [100]})  # no 'chr' prefix
    scores = score_variants_at_positions(df, bw_path)
    assert scores[0] == pytest.approx(3.14)


def test_score_variants_unknown_chromosome(tmp_path):
    """Variants on chromosomes not in the bigWig return NaN, don't crash."""
    bw_path = tmp_path / "tiny.bw"
    _write_tiny_bigwig(bw_path, [("chr1", 99, 100, 1.0)])

    df = pd.DataFrame({"chrom": ["chrM"], "pos": [100]})
    scores = score_variants_at_positions(df, bw_path)
    assert np.isnan(scores[0])


def test_score_variants_rejects_non_integer_pos(tmp_path):
    """Float pos column should fail loudly (silent rounding hides bugs)."""
    bw_path = tmp_path / "tiny.bw"
    _write_tiny_bigwig(bw_path, [("chr1", 99, 100, 1.0)])

    df = pd.DataFrame({"chrom": ["chr1"], "pos": [100.0]})
    with pytest.raises(AssertionError, match="pos must be integer"):
        score_variants_at_positions(df, bw_path)


def _scored_parquet(tmp_path, name, scores, labels, subsets):
    """Write a fake scored-variant parquet matching the score_variants output schema."""
    path = tmp_path / f"{name}.parquet"
    n = len(scores)
    df = pd.DataFrame(
        {
            "chrom": ["chr1"] * n,
            "pos": list(range(1, n + 1)),
            "ref": ["A"] * n,
            "alt": ["T"] * n,
            "label": labels,
            "subset": subsets,
            "score": scores,
        }
    )
    df.to_parquet(path, index=False)
    return path


def test_aggregate_traitgym_metrics_nan_accounting(tmp_path):
    """NaN counts should be correct per subset and globally; AUPRC computed
    after fillna(0)."""
    # 6 variants: 3 in subset_A, 3 in subset_B. Score column has 1 NaN
    # in subset_A, 0 in subset_B.
    scores = [0.9, np.nan, 0.1, 0.8, 0.2, 0.5]
    labels = [1, 1, 0, 1, 0, 0]
    subsets = ["A", "A", "A", "B", "B", "B"]
    p = _scored_parquet(tmp_path, "phyloP_241m", scores, labels, subsets)

    metrics, md = aggregate_traitgym_metrics({"phyloP_241m": p})

    # Schema check.
    expected_cols = {
        "metric",
        "score_type",
        "score_name",
        "subset",
        "value",
        "n_pos",
        "n_neg",
        "n_nan",
        "n_total",
    }
    assert expected_cols.issubset(set(metrics.columns))

    # NaN accounting.
    nan_global = metrics[
        (metrics["score_name"] == "phyloP_241m") & (metrics["subset"] == "global")
    ]["n_nan"].iloc[0]
    nan_a = metrics[
        (metrics["score_name"] == "phyloP_241m") & (metrics["subset"] == "A")
    ]["n_nan"].iloc[0]
    nan_b = metrics[
        (metrics["score_name"] == "phyloP_241m") & (metrics["subset"] == "B")
    ]["n_nan"].iloc[0]
    assert nan_global == 1
    assert nan_a == 1
    assert nan_b == 0

    # Total counts.
    tot_global = metrics[
        (metrics["score_name"] == "phyloP_241m") & (metrics["subset"] == "global")
    ]["n_total"].iloc[0]
    assert tot_global == 6

    # Markdown contains the AUPRC and NaN-count tables.
    assert "AUPRC" in md
    assert "NaN" in md
    assert "phyloP_241m" in md


def test_aggregate_traitgym_metrics_multi_score_column_order(tmp_path):
    """Markdown columns appear in the order of parquet_paths.keys()."""
    scores = [0.5, 0.7, 0.3, 0.2]
    labels = [1, 1, 0, 0]
    subsets = ["x", "x", "x", "x"]
    paths = {
        "phyloP_100v": _scored_parquet(
            tmp_path, "phyloP_100v", scores, labels, subsets
        ),
        "phyloP_241m": _scored_parquet(
            tmp_path, "phyloP_241m", scores, labels, subsets
        ),
        "phastCons_43p": _scored_parquet(
            tmp_path, "phastCons_43p", scores, labels, subsets
        ),
    }
    _, md = aggregate_traitgym_metrics(paths)

    # All three names appear; phyloP_100v comes before phyloP_241m (column order).
    pos_100v = md.find("phyloP_100v")
    pos_241m = md.find("phyloP_241m")
    pos_43p = md.find("phastCons_43p")
    assert 0 < pos_100v < pos_241m < pos_43p


def test_score_variants_preserves_input_order(tmp_path):
    """Output is row-aligned with input — important since the pipeline
    concats it back as a column."""
    bw_path = tmp_path / "tiny.bw"
    _write_tiny_bigwig(
        bw_path,
        [
            ("chr1", 9, 10, 10.0),
            ("chr1", 19, 20, 20.0),
            ("chr1", 29, 30, 30.0),
        ],
    )

    # Deliberately unsorted positions
    df = pd.DataFrame({"chrom": ["chr1"] * 3, "pos": [30, 10, 20]})
    scores = score_variants_at_positions(df, bw_path)

    assert scores[0] == pytest.approx(30.0)
    assert scores[1] == pytest.approx(10.0)
    assert scores[2] == pytest.approx(20.0)
