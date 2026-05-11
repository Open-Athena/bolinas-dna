"""Tests for the pairwise-accuracy metric (matched-pair within-group)."""

import math

import pandas as pd
import pytest

from bolinas.evals.metrics import compute_pairwise_metrics, pairwise_accuracy


# ---- pairwise_accuracy ----------------------------------------------------


def _frame(label, score, match_group):
    return (
        pd.Series(label),
        pd.Series(score, dtype=float),
        pd.Series(match_group),
    )


def test_pairwise_accuracy_all_wins():
    label, score, mg = _frame([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], [0, 0, 1, 1])
    res = pairwise_accuracy(label, score, mg)
    assert res["value"] == 1.0
    assert res["se"] == 0.0
    assert res["n_pairs"] == 2
    assert res["n_ties"] == 0


def test_pairwise_accuracy_all_losses():
    label, score, mg = _frame([1, 0, 1, 0], [0.1, 0.9, 0.2, 0.8], [0, 0, 1, 1])
    res = pairwise_accuracy(label, score, mg)
    assert res["value"] == 0.0
    assert res["se"] == 0.0
    assert res["n_pairs"] == 2
    assert res["n_ties"] == 0


def test_pairwise_accuracy_fifty_fifty():
    # 4 pairs: 2 wins, 2 losses.
    label = [1, 0, 1, 0, 1, 0, 1, 0]
    score = [0.9, 0.1, 0.1, 0.9, 0.8, 0.2, 0.2, 0.8]
    mg = [0, 0, 1, 1, 2, 2, 3, 3]
    res = pairwise_accuracy(*_frame(label, score, mg))
    assert res["value"] == 0.5
    assert res["se"] == pytest.approx(math.sqrt(0.25 / 4))
    assert res["n_pairs"] == 4
    assert res["n_ties"] == 0


def test_pairwise_accuracy_pure_ties():
    label = [1, 0, 1, 0, 1, 0]
    score = [0.5, 0.5, 0.7, 0.7, 0.0, 0.0]
    mg = [0, 0, 1, 1, 2, 2]
    res = pairwise_accuracy(*_frame(label, score, mg))
    assert res["value"] == 0.5  # ties counted as 0.5
    assert res["n_pairs"] == 3
    assert res["n_ties"] == 3
    assert res["se"] == pytest.approx(math.sqrt(0.5 * 0.5 / 3))


def test_pairwise_accuracy_p_value_all_wins_is_small():
    # 10 wins out of 10 — two-sided binomial test against Binom(10, 0.5)
    # → p = 2 * P(X >= 10) = 2 * (1/1024) ≈ 0.00195.
    label = [1, 0] * 10
    score = [1.0, 0.0] * 10
    mg = [i for i in range(10) for _ in range(2)]
    res = pairwise_accuracy(*_frame(label, score, mg))
    assert res["n_pairs"] == 10
    assert res["n_ties"] == 0
    assert res["p_value"] == pytest.approx(2 / 1024)


def test_pairwise_accuracy_p_value_fifty_fifty_near_one():
    # 5 wins / 5 losses → maximally null, p_value = 1.0.
    label = [1, 0] * 10
    score = ([1.0, 0.0] * 5) + ([0.0, 1.0] * 5)
    mg = [i for i in range(10) for _ in range(2)]
    res = pairwise_accuracy(*_frame(label, score, mg))
    assert res["value"] == 0.5
    assert res["p_value"] == pytest.approx(1.0)


def test_pairwise_accuracy_p_value_drops_ties():
    # 2 wins, 2 ties: effective n=2, both wins → p = 2 * (1/4) = 0.5.
    label = [1, 0, 1, 0, 1, 0, 1, 0]
    score = [1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5]
    mg = [0, 0, 1, 1, 2, 2, 3, 3]
    res = pairwise_accuracy(*_frame(label, score, mg))
    assert res["n_pairs"] == 4
    assert res["n_ties"] == 2
    assert res["p_value"] == pytest.approx(0.5)


def test_pairwise_accuracy_p_value_all_ties_is_one():
    label = [1, 0, 1, 0]
    score = [0.5, 0.5, 0.5, 0.5]
    mg = [0, 0, 1, 1]
    res = pairwise_accuracy(*_frame(label, score, mg))
    assert res["n_ties"] == 2
    assert res["p_value"] == 1.0


def test_pairwise_accuracy_mixed_wins_and_ties():
    # 4 pairs: 2 wins, 1 tie, 1 loss => value = (2 + 0.5*1) / 4 = 0.625
    label = [1, 0, 1, 0, 1, 0, 1, 0]
    score = [0.9, 0.1, 0.5, 0.5, 0.8, 0.2, 0.1, 0.9]
    mg = [0, 0, 1, 1, 2, 2, 3, 3]
    res = pairwise_accuracy(*_frame(label, score, mg))
    expected = (2 + 0.5) / 4
    assert res["value"] == pytest.approx(expected)
    assert res["se"] == pytest.approx(math.sqrt(expected * (1 - expected) / 4))
    assert res["n_pairs"] == 4
    assert res["n_ties"] == 1


def test_pairwise_accuracy_handles_bool_label():
    label = [True, False, True, False]
    score = [0.9, 0.1, 0.8, 0.2]
    mg = [0, 0, 1, 1]
    res = pairwise_accuracy(*_frame(label, score, mg))
    assert res["value"] == 1.0
    assert res["n_pairs"] == 2


def test_pairwise_accuracy_se_decreases_with_n():
    """SE for 50/50 must shrink as 1/sqrt(n)."""
    se_small = pairwise_accuracy(*_frame([1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]))[
        "se"
    ]
    n_big = 100
    label = [1, 0] * n_big
    score = ([1, 0] * (n_big // 2)) + ([0, 1] * (n_big // 2))
    mg = [i for i in range(n_big) for _ in range(2)]
    se_big = pairwise_accuracy(*_frame(label, score, mg))["se"]
    assert se_big < se_small


def test_pairwise_accuracy_rejects_extra_positive_in_group():
    # match_group 0 has 2 positives, 1 negative.
    label = [1, 1, 0, 1, 0]
    score = [0.9, 0.5, 0.1, 0.8, 0.2]
    mg = [0, 0, 0, 1, 1]
    with pytest.raises(AssertionError, match="exactly 1 positive"):
        pairwise_accuracy(*_frame(label, score, mg))


def test_pairwise_accuracy_rejects_missing_negative_in_group():
    # match_group 1 has only a positive.
    label = [1, 0, 1]
    score = [0.9, 0.1, 0.8]
    mg = [0, 0, 1]
    with pytest.raises(AssertionError, match="exactly 1 positive"):
        pairwise_accuracy(*_frame(label, score, mg))


def test_pairwise_accuracy_rejects_length_mismatch():
    with pytest.raises(AssertionError, match="length mismatch"):
        pairwise_accuracy(
            pd.Series([1, 0, 1, 0]),
            pd.Series([0.9, 0.1, 0.8]),
            pd.Series([0, 0, 1, 1]),
        )


def test_pairwise_accuracy_rejects_nan_score():
    """NaN in score is a silent-corruption risk (NaN > NaN and NaN == NaN both
    False -> NaN-vs-NaN silently counts as a loss). Caller must fill upstream."""
    label = [1, 0, 1, 0]
    score = [float("nan"), 0.1, 0.8, 0.2]
    mg = [0, 0, 1, 1]
    with pytest.raises(AssertionError, match="NaN"):
        pairwise_accuracy(*_frame(label, score, mg))


# ---- compute_pairwise_metrics --------------------------------------------


def test_compute_pairwise_metrics_per_subset():
    """Stratifies by ``subset`` and returns one row per (subset, score)."""
    dataset = pd.DataFrame(
        {
            "label": [1, 0, 1, 0, 1, 0, 1, 0],
            "subset": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "match_group": [0, 0, 1, 1, 2, 2, 3, 3],
        }
    )
    # Subset A: both pairs are wins -> 1.0
    # Subset B: one win, one loss -> 0.5
    scores = pd.DataFrame(
        {
            "score": [0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.1, 0.9],
        }
    )
    metrics = compute_pairwise_metrics(
        dataset=dataset, scores=scores, score_columns=["score"]
    )
    assert set(metrics.columns) == {
        "score_type",
        "subset",
        "value",
        "se",
        "n_pairs",
        "n_ties",
        "p_value",
    }
    assert len(metrics) == 2  # 2 subsets x 1 score_column
    by_subset = metrics.set_index("subset")["value"]
    assert by_subset["A"] == 1.0
    assert by_subset["B"] == 0.5


def test_compute_pairwise_metrics_multi_score_columns():
    dataset = pd.DataFrame(
        {
            "label": [1, 0, 1, 0],
            "subset": ["A", "A", "A", "A"],
            "match_group": [0, 0, 1, 1],
        }
    )
    scores = pd.DataFrame(
        {
            "score_a": [0.9, 0.1, 0.8, 0.2],  # all wins -> 1.0
            "score_b": [0.1, 0.9, 0.2, 0.8],  # all losses -> 0.0
        }
    )
    metrics = compute_pairwise_metrics(dataset=dataset, scores=scores)
    by_score = metrics.set_index("score_type")["value"]
    assert by_score["score_a"] == 1.0
    assert by_score["score_b"] == 0.0


def test_compute_pairwise_metrics_rejects_subset_straddle():
    """A match_group spanning two subsets is a silent-corruption risk."""
    dataset = pd.DataFrame(
        {
            "label": [1, 0, 1, 0],
            "subset": ["A", "B", "A", "A"],  # group 0 straddles A/B
            "match_group": [0, 0, 1, 1],
        }
    )
    scores = pd.DataFrame({"score": [0.9, 0.1, 0.8, 0.2]})
    with pytest.raises(AssertionError, match="span multiple subsets"):
        compute_pairwise_metrics(
            dataset=dataset, scores=scores, score_columns=["score"]
        )


def test_compute_pairwise_metrics_default_score_columns():
    dataset = pd.DataFrame(
        {
            "label": [1, 0],
            "subset": ["A", "A"],
            "match_group": [0, 0],
        }
    )
    scores = pd.DataFrame({"foo": [0.9, 0.1], "bar": [0.1, 0.9]})
    metrics = compute_pairwise_metrics(dataset=dataset, scores=scores)
    assert set(metrics["score_type"]) == {"foo", "bar"}


def test_compute_pairwise_metrics_requires_match_group_column():
    dataset = pd.DataFrame({"label": [1, 0], "subset": ["A", "A"]})
    scores = pd.DataFrame({"score": [0.9, 0.1]})
    with pytest.raises(AssertionError, match="match_group"):
        compute_pairwise_metrics(
            dataset=dataset, scores=scores, score_columns=["score"]
        )
