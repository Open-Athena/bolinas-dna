"""Tests for Evo2 variant scoring helpers.

The real inference path needs ``evo2`` + an H100/GH200 + ~40 GB of downloaded
weights, so this file only exercises the lightweight contract (signature,
import, aggregator math). The heavy lifting is checked at runtime on the
SkyPilot cluster.
"""

import numpy as np

from bolinas.pipelines.evals.evo2 import aggregate_ll_gap, compute_evo2_ll


def test_evo2_helpers_importable():
    """Sanity-check that the public LL-gap functions exist and are importable
    without evo2 installed.

    Variant scoring lives in ``scripts/evo2_eval/_evo2_scoring.py``
    (intentionally not testable from here without an evo2 install)."""
    assert callable(compute_evo2_ll)
    assert callable(aggregate_ll_gap)


def test_aggregate_ll_gap_token_weighted_mean_and_sign():
    """Per-row sums and counts must aggregate to a token-weighted dataset
    mean (sum-then-divide), and the gap convention must be
    ``LL_upper - LL_lower``: positive when uppercase log-probs are
    closer to 0 (easier) than lowercase log-probs.
    """
    # Two rows, one all-upper and one all-lower — the edge case that
    # motivates returning sums+counts (per-row means would NaN here).
    # All-upper row: 4 target tokens, total log p = -2.0
    # All-lower row: 6 target tokens, total log p = -9.0
    pred = np.array(
        [
            [-2.0, 0.0, 4.0, 0.0],
            [0.0, -9.0, 0.0, 6.0],
        ],
        dtype=np.float32,
    )
    out = aggregate_ll_gap(pred)

    # token-weighted mean: -11.0 / 10 = -1.1
    assert out["LL_all"] == -1.1
    # mean LL on uppercase: -2 / 4 = -0.5
    assert out["LL_upper"] == -0.5
    # mean LL on lowercase: -9 / 6 = -1.5
    assert out["LL_lower"] == -1.5
    # gap = LL_upper - LL_lower = -0.5 - (-1.5) = +1.0
    # (positive ⇒ uppercase is easier to predict, the expected sign on CDS)
    assert out["gap"] == 1.0
    assert out["n_upper"] == 4
    assert out["n_lower"] == 6


def test_aggregate_ll_gap_uses_fp64():
    """Aggregator must cast to fp64 before the cross-row sum (PR #18 of
    biofoundation). We verify by feeding a fp32 input large enough that
    a naive fp32 ``sum`` would drop the bottom bits, then checking the
    result has fp64 precision relative to a fully-fp64 reference.
    """
    rng = np.random.default_rng(0)
    n_rows = 50_000
    upper_logp = rng.uniform(-3.0, -0.1, size=n_rows).astype(np.float32)
    lower_logp = rng.uniform(-3.0, -0.1, size=n_rows).astype(np.float32)
    pred = np.stack(
        [
            upper_logp,
            lower_logp,
            np.ones(n_rows, dtype=np.float32),
            np.ones(n_rows, dtype=np.float32),
        ],
        axis=1,
    )

    out = aggregate_ll_gap(pred)
    # All four returned floats are Python floats (fp64).
    for k in ("LL_all", "LL_upper", "LL_lower", "gap"):
        assert isinstance(out[k], float)
    # Reference: same arithmetic but fully fp64 from the start. Any miss
    # in the cast will show up as a >fp32-eps difference here on a 50k-row
    # sample where the per-token mean is ~-1.5.
    ref_upper = upper_logp.astype(np.float64).sum() / n_rows
    ref_lower = lower_logp.astype(np.float64).sum() / n_rows
    assert abs(out["LL_upper"] - ref_upper) < 1e-12
    assert abs(out["LL_lower"] - ref_lower) < 1e-12


def test_aggregate_ll_gap_rejects_bad_shape():
    import pytest

    with pytest.raises(AssertionError):
        aggregate_ll_gap(np.zeros((5, 3)))  # not [N, 4]


def test_aggregate_ll_gap_rejects_zero_count_buckets():
    import pytest

    # all upper, no lower at all
    pred = np.array([[-1.0, 0.0, 5.0, 0.0]], dtype=np.float32)
    with pytest.raises(AssertionError, match="non-functional"):
        aggregate_ll_gap(pred)

    # all lower, no upper at all
    pred = np.array([[0.0, -1.0, 0.0, 5.0]], dtype=np.float32)
    with pytest.raises(AssertionError, match="functional"):
        aggregate_ll_gap(pred)
