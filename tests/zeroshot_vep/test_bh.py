"""Tests for the Benjamini-Hochberg FDR adjustment used in aggregate.smk.

The function is defined inline in the Snakemake rule (no Python module), so
import it from the rule file directly.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np


_HERE = pathlib.Path(__file__).resolve().parents[2]
_AGGREGATE_PATH = _HERE / "snakemake" / "analysis" / "zeroshot_vep" / "workflow" / "rules" / "aggregate.smk"


def _load_bh_adjust():
    """Load ``_bh_adjust`` from aggregate.smk (which is a .smk = Python-ish file).

    Strip Snakemake-specific syntax via a minimal preprocess: we only need the
    top-level helper, so import as a module after slicing off the ``rule`` block.
    """
    text = _AGGREGATE_PATH.read_text()
    # Cut at the first occurrence of "rule " — leave only the imports + helper.
    rule_start = text.find("\nrule ")
    py_src = text[:rule_start] if rule_start >= 0 else text
    spec = importlib.util.spec_from_loader("_aggregate_helpers", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(py_src, mod.__dict__)
    return mod._bh_adjust


_bh_adjust = _load_bh_adjust()


def test_bh_all_significant():
    # All p=0.001, n=10. BH-adjusted = p * n / rank but step-up enforces
    # monotonicity, so they all collapse to p*n/n = 0.001 since they're tied.
    p = np.full(10, 0.001)
    q = _bh_adjust(p)
    np.testing.assert_allclose(q, 0.001)


def test_bh_one_significant_among_many():
    # 1 p=0.001, 99 p=0.5. q for the small one = 0.001 * 100 / 1 = 0.1.
    p = np.array([0.001] + [0.5] * 99)
    q = _bh_adjust(p)
    assert q[0] == pytest.approx(0.1)
    # The bulk of p=0.5 gets q = min(0.5*100/k, 1) for k=2..100, with the largest
    # contribution at k=100 → 0.5. Step-up enforces non-decreasing → all 0.5.
    np.testing.assert_allclose(q[1:], 0.5)


def test_bh_monotonicity():
    # q must be monotone non-decreasing in p (after sorting).
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=50)
    q = _bh_adjust(p)
    order = np.argsort(p)
    q_sorted = q[order]
    assert np.all(np.diff(q_sorted) >= -1e-12), "q must be non-decreasing in sorted p"


def test_bh_caps_at_one():
    p = np.array([0.99, 0.99, 0.99])
    q = _bh_adjust(p)
    assert (q <= 1.0).all()


def test_bh_passes_through_nan():
    p = np.array([0.001, np.nan, 0.5, np.nan, 0.01])
    q = _bh_adjust(p)
    assert np.isnan(q[1]) and np.isnan(q[3])
    # The 3 non-NaN values get BH applied with n=3.
    # sorted p (non-nan) = [0.001, 0.01, 0.5]; q_raw = [0.003, 0.015, 0.5]; step-up → [0.003, 0.015, 0.5]
    assert q[0] == pytest.approx(0.003)
    assert q[4] == pytest.approx(0.015)
    assert q[2] == pytest.approx(0.5)


def test_bh_all_nan():
    p = np.array([np.nan, np.nan])
    q = _bh_adjust(p)
    assert np.isnan(q).all()


# pytest needs to be importable in this file for the approx call.
import pytest  # noqa: E402
