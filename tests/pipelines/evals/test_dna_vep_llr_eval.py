# Copyright The Bolinas Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lm_eval task class ``DnaVepLlrEvalTask``.

Pin down the per-variant aggregation logic — the load-bearing piece for #179
parity with the offline batched VEP path. Avoids loading any HF dataset; the
tests construct synthetic doc dicts and call ``process_results`` /
``aggregation`` directly.
"""

import math

import pytest

pytest.importorskip("lm_eval", reason="install with `uv sync --extra marin` to run")

from bolinas.pipelines.evals.lm_eval.dna_vep_llr_eval import (  # noqa: E402
    METRIC_REGISTRY,
    _PairwiseAccuracyAggregation,
)


def _items_from_per_variant(per_variant_rows):
    """Flatten per-variant rows into the (score, target, subset, variant_id, match_group)
    tuples that ``DnaVepLlrEvalTask.aggregation()`` consumes.

    ``per_variant_rows`` is a list of
        (variant_id, target, subset, match_group, scores)
    where ``scores`` is a list of one or more per-row scores for that variant
    (one score per strand for the FWD+RC dataset; one score for FWD-only).
    """
    items = []
    for variant_id, target, subset, match_group, scores in per_variant_rows:
        for s in scores:
            items.append((s, target, subset, variant_id, match_group))
    return items


def _run_aggregation(items, metric_name="pairwise_accuracy"):
    store: dict[str, float] = {}
    agg = _PairwiseAccuracyAggregation(results_store=store, metric_name=metric_name)
    scalar = agg(items)
    return scalar, store


def test_metric_registry_has_pairwise_accuracy():
    assert "pairwise_accuracy" in METRIC_REGISTRY
    assert METRIC_REGISTRY["pairwise_accuracy"]["higher_is_better"] is True


def test_one_strand_per_variant_no_average():
    """One row per variant: aggregation should match a direct PairwiseAccuracy
    on the per-row scores. 4 perfectly-separable matched pairs => PA = 1.0.
    """
    per_variant = [
        # (variant_id, target, subset, match_group, scores)
        (("1", 11, "G", "A"), 1, "missense", 1, [0.9]),  # pos
        (("1", 12, "T", "A"), 0, "missense", 1, [0.1]),  # neg
        (("1", 13, "C", "G"), 1, "missense", 2, [0.8]),  # pos
        (("1", 14, "A", "T"), 0, "missense", 2, [0.2]),  # neg
        (("2", 11, "G", "A"), 1, "missense", 3, [0.7]),  # pos
        (("2", 12, "T", "A"), 0, "missense", 3, [0.3]),  # neg
    ] + [
        (
            ("3", 100 + i, "A", "T"),
            int(i % 2 == 0),
            "splicing",
            4 + (i // 2),
            [0.6 + 0.05 * i if i % 2 == 0 else 0.4 - 0.05 * i],
        )
        for i in range(60)
    ]
    items = _items_from_per_variant(per_variant)
    scalar, store = _run_aggregation(items)

    # All positives score above their matched negative => global PA = 1.0
    assert scalar == pytest.approx(1.0)
    assert store["_global_/pairwise_accuracy"] == pytest.approx(1.0)
    # Per-subset rows present (missense and splicing).
    assert "missense/pairwise_accuracy" in store
    assert "splicing/pairwise_accuracy" in store
    assert "_macro_avg_/pairwise_accuracy" in store


def test_two_strands_per_variant_collapses_then_metric():
    """Two rows per variant: scores are averaged per (chrom, pos, ref, alt) before PA.

    Set up scores so that the FWD-only PA would be 0.5 (a tie) but the AVG
    (FWD+RC) breaks the tie in favor of the positive => PA = 1.0.
    """
    # match_group 1: pos has FWD=0.5, RC=0.9 (avg 0.7); neg has FWD=0.5, RC=0.1 (avg 0.3)
    per_variant = [
        (("1", 11, "G", "A"), 1, "missense", 1, [0.5, 0.9]),  # pos
        (("1", 12, "T", "A"), 0, "missense", 1, [0.5, 0.1]),  # neg
    ]
    # Pad to >=30 pairs in one subset so compute_pairwise_metrics' macro_avg
    # threshold is satisfied (n_min=30).
    for i in range(2, 31):
        per_variant.append((("9", 100 + i, "A", "T"), 1, "missense", i, [0.5, 0.9]))
        per_variant.append((("9", 200 + i, "A", "T"), 0, "missense", i, [0.5, 0.1]))

    items = _items_from_per_variant(per_variant)
    scalar, store = _run_aggregation(items)

    assert scalar == pytest.approx(1.0)
    assert store["_global_/pairwise_accuracy"] == pytest.approx(1.0)


def test_inconsistent_target_within_variant_fails_loud():
    """If two rows for the same variant_id disagree on target, the assertion fires."""
    items = [
        (0.5, 1, "missense", ("1", 11, "G", "A"), 1),
        (0.6, 0, "missense", ("1", 11, "G", "A"), 1),  # contradicting target
    ]
    with pytest.raises(AssertionError, match="inconsistent target"):
        _run_aggregation(items)


def test_inconsistent_subset_within_variant_fails_loud():
    items = [
        (0.5, 1, "missense", ("1", 11, "G", "A"), 1),
        (0.6, 1, "splicing", ("1", 11, "G", "A"), 1),
    ]
    with pytest.raises(AssertionError, match="inconsistent subset"):
        _run_aggregation(items)


def test_inconsistent_match_group_within_variant_fails_loud():
    items = [
        (0.5, 1, "missense", ("1", 11, "G", "A"), 1),
        (0.6, 1, "missense", ("1", 11, "G", "A"), 99),
    ]
    with pytest.raises(AssertionError, match="inconsistent match_group"):
        _run_aggregation(items)


def test_se_is_finite_and_consistent_with_wald():
    """Standard error matches the Wald binomial form ``sqrt(p*(1-p)/n)``.

    Uses 30 pairs (so compute_pairwise_metrics' n_min=30 macro-avg gate fires
    cleanly) with 20 wins and 10 losses => global PA = 20/30.
    """
    per_variant: list = []
    # 20 winning pairs (positive scores higher).
    for i in range(20):
        per_variant.append((("1", 100 + i, "G", "A"), 1, "missense", i + 1, [0.9]))
        per_variant.append((("1", 200 + i, "T", "A"), 0, "missense", i + 1, [0.1]))
    # 10 losing pairs.
    for i in range(20, 30):
        per_variant.append((("1", 100 + i, "G", "A"), 1, "missense", i + 1, [0.1]))
        per_variant.append((("1", 200 + i, "T", "A"), 0, "missense", i + 1, [0.9]))
    items = _items_from_per_variant(per_variant)
    scalar, store = _run_aggregation(items)
    assert scalar == pytest.approx(20 / 30)
    n_pairs = 30
    expected_se = math.sqrt((20 / 30) * (1 - 20 / 30) / n_pairs)
    assert store["_global_/pairwise_accuracy_se"] == pytest.approx(expected_se)
