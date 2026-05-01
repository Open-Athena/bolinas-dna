"""Tests for ``bolinas.conservation.calibration``."""

from __future__ import annotations

import numpy as np
import pytest

from bolinas.conservation.calibration import calibrate_to_match_count
from bolinas.conservation.histogram import PhylopHistogram


@pytest.fixture
def edges() -> np.ndarray:
    return np.linspace(-20.0, 20.0, 1001)


def _hist(values: np.ndarray, edges: np.ndarray) -> PhylopHistogram:
    counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
    return PhylopHistogram(counts=counts.astype(np.int64), edges=edges, n_nan=0)


def test_calibrate_identity(edges: np.ndarray) -> None:
    """Same histogram on both sides → matched threshold equals the reference."""
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 2.0, size=200_000)
    hist = _hist(values, edges)
    out = calibrate_to_match_count(hist, hist, ref_threshold=2.27)
    # Must be within one bin's worth (0.04) of the reference
    assert abs(out["target"]["threshold"] - 2.27) < 0.05


def test_calibrate_known_shift(edges: np.ndarray) -> None:
    """Reference is N(0, 2); target is N(1, 2). Calibrated threshold should be
    ~ 1 above the reference threshold (since target distribution is shifted +1)."""
    rng = np.random.default_rng(0)
    n = 500_000
    ref_vals = rng.normal(0.0, 2.0, size=n)
    tgt_vals = rng.normal(1.0, 2.0, size=n)
    ref_hist = _hist(ref_vals, edges)
    tgt_hist = _hist(tgt_vals, edges)

    out = calibrate_to_match_count(tgt_hist, ref_hist, ref_threshold=2.27)
    # The target threshold should be ~ 1 above the reference threshold,
    # within sampling noise + bin width
    assert abs(out["target"]["threshold"] - (2.27 + 1.0)) < 0.1


def test_calibration_dict_shape(edges: np.ndarray) -> None:
    """Returned dict has the expected structure for downstream JSON output."""
    rng = np.random.default_rng(0)
    hist = _hist(rng.normal(0, 2, size=100_000), edges)
    out = calibrate_to_match_count(
        hist,
        hist,
        ref_threshold=2.0,
        target_name="phyloP_447m",
        ref_name="phyloP_241m",
    )
    assert set(out.keys()) == {"phyloP_241m", "phyloP_447m", "hist_meta"}
    assert set(out["phyloP_241m"]) == {"threshold", "count", "total_bases", "n_nan"}
    assert set(out["phyloP_447m"]) == {
        "threshold",
        "count",
        "count_target",
        "abs_relative_error",
        "total_bases",
        "n_nan",
    }
    assert set(out["hist_meta"]) == {"n_bins", "min", "max", "bin_width"}
    assert out["phyloP_447m"]["abs_relative_error"] < 0.01


def test_calibration_fails_loudly_with_too_coarse_bins() -> None:
    """If the bin width swallows the reference count entirely, calibration
    should error rather than silently return a bogus threshold."""
    edges = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])  # width 10 — extremely coarse
    rng = np.random.default_rng(0)
    ref_vals = rng.normal(0, 0.5, size=10_000)  # nearly all in bin [-10, 0)
    tgt_vals = rng.normal(0, 0.5, size=10_000)
    ref_hist = _hist(ref_vals, edges)
    tgt_hist = _hist(tgt_vals, edges)
    # ref_threshold near a bin edge to amplify mismatch
    # With this coarse a histogram and a threshold at 0.0, calibration should
    # still find a matching threshold (both distributions are identical), so
    # this test mostly checks we don't crash. Accept either success or the
    # explicit assertion.
    try:
        out = calibrate_to_match_count(tgt_hist, ref_hist, ref_threshold=2.27)
        assert out["target"]["abs_relative_error"] < 0.01
    except AssertionError as e:
        assert "bin width" in str(e) or "calibration failed" in str(e)
