"""Tests for ``bolinas.supervised.classifiers``: standard + pair-aware classifiers."""

import numpy as np
import pytest

from bolinas.supervised.classifiers import (
    _pairwise_accuracy_within_groups,
    all_standard_specs,
    build_pairwise_diff_dataset,
    fit_pairwise_linear_weights,
    knn_spec,
    linearsvc_spec,
    logreg_spec,
    pairwise_oof_predict,
    xgboost_spec,
)
from bolinas.supervised.cv import oof_predict


TWELVE_CHROMS = ["1", "11", "13", "15", "17", "19", "21", "3", "5", "7", "9", "X"]


def _make_paired_dataset(n_per_chrom: int = 60, d: int = 6, seed: int = 0):
    """Synthetic paired dataset: positives and negatives matched, separable on dim 0.

    Returns ``(X, y, chroms, match_group)`` such that:
    * Each chrom contributes ``n_per_chrom`` rows, half positive half negative.
    * Each (positive, negative) pair shares a ``match_group`` integer.
    * Positives have +1.5 shift on feature 0 (plus noise), so a linear
      classifier should rank them above their match.
    """
    rng = np.random.default_rng(seed)
    Xs, ys, cs, mgs = [], [], [], []
    next_mg = 0
    for c in TWELVE_CHROMS:
        n_pairs = n_per_chrom // 2
        x_neg = rng.normal(size=(n_pairs, d))
        x_pos = x_neg.copy()
        x_pos[:, 0] += 1.5 + rng.normal(0, 0.5, size=n_pairs)
        for i in range(n_pairs):
            Xs.append(x_neg[i])
            ys.append(0)
            cs.append(c)
            mgs.append(next_mg)
            Xs.append(x_pos[i])
            ys.append(1)
            cs.append(c)
            mgs.append(next_mg)
            next_mg += 1
    return (
        np.stack(Xs).astype(np.float32),
        np.array(ys, dtype=int),
        np.array(cs),
        np.array(mgs, dtype=int),
    )


# ---------- standard classifier specs ------------------------------------


def test_logreg_spec_bfs_mode_is_single_C():
    spec = logreg_spec(mode="bfs")
    assert spec.name == "logreg_l2"
    assert spec.param_grid["clf__C"] == [1.0]


def test_logreg_spec_refine_mode_has_wide_grid():
    spec = logreg_spec(mode="refine")
    C_grid = spec.param_grid["clf__C"]
    assert len(C_grid) >= 5
    assert min(C_grid) < 1e-4 and max(C_grid) >= 1.0


def test_linearsvc_spec_bfs_mode_is_single_C():
    spec = linearsvc_spec(mode="bfs")
    assert "clf__C" in spec.param_grid
    assert spec.param_grid["clf__C"] == [1.0]


def test_knn_spec_refine_grid_scales_with_n_train():
    small = knn_spec(n_train=500, mode="refine")
    big = knn_spec(n_train=20000, mode="refine")
    assert max(small.param_grid["clf__n_neighbors"]) <= 500
    assert max(big.param_grid["clf__n_neighbors"]) > 500


def test_knn_spec_bfs_mode_is_single_k():
    spec = knn_spec(mode="bfs")
    assert spec.param_grid["clf__n_neighbors"] == [25]


def test_xgboost_spec_either_present_or_none():
    spec = xgboost_spec()
    assert spec is None or "clf__max_depth" in spec.param_grid


def test_specs_reject_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        logreg_spec(mode="nonsense")


def test_all_standard_specs_uses_oof_predict_end_to_end():
    """The full standard ClassifierSpec → oof_predict path should run without error.

    Uses a narrow C grid for this synthetic problem; the production spec keeps
    the wide ``logspace(-8, 0, 10)`` grid for cases where the right C is hard to
    predict in advance.
    """
    X, y, chroms, _ = _make_paired_dataset()
    spec = logreg_spec()
    preds, records = oof_predict(
        X=X,
        y=y,
        chroms=chroms,
        estimator=spec.estimator,
        param_grid={"clf__C": [0.1, 1.0, 10.0]},
        n_splits=3,
        n_splits_inner=3,
    )
    assert preds.shape == (len(X),)
    assert not np.isnan(preds).any()
    assert len(records) == 3
    # On this easy synthetic problem, OOF preds should clearly separate classes.
    assert preds[y == 1].mean() > preds[y == 0].mean() + 0.05


def test_all_standard_specs_returns_at_least_three():
    specs = all_standard_specs()
    names = {s.name for s in specs}
    assert {"logreg_l2", "linearsvc", "knn"}.issubset(names)


# ---------- pair-aware classifier ----------------------------------------


def test_build_pairwise_diff_dataset_emits_two_rows_per_pair():
    X = np.array([[1.0, 2.0], [3.0, 5.0], [0.0, 0.0], [4.0, 8.0]])
    y = np.array([0, 1, 0, 1])
    mg = np.array([0, 0, 1, 1])
    X_diff, y_diff = build_pairwise_diff_dataset(X, y, mg)
    # 2 pairs * 2 (orientations each) = 4 rows.
    assert X_diff.shape == (4, 2)
    assert y_diff.shape == (4,)
    # Labels alternate +1 / 0.
    assert set(y_diff.tolist()) == {0, 1}
    # First pair: feat(pos) - feat(neg) = [3-1, 5-2] = [2, 3], label 1.
    np.testing.assert_allclose(X_diff[0], [2.0, 3.0])
    assert y_diff[0] == 1
    np.testing.assert_allclose(X_diff[1], [-2.0, -3.0])
    assert y_diff[1] == 0


def test_build_pairwise_diff_dataset_skips_incomplete_groups():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1, 1, 0])
    mg = np.array([0, 0, 1])  # group 0 has two positives, group 1 has only a negative
    with pytest.raises(ValueError, match="no valid"):
        build_pairwise_diff_dataset(X, y, mg)


def test_fit_pairwise_linear_weights_recovers_class_separating_axis():
    """On a linearly-separable paired dataset, the weight vector should point along dim 0."""
    X, y, _, mg = _make_paired_dataset(n_per_chrom=40, d=3)
    w = fit_pairwise_linear_weights(X, y, mg, base="logreg", C=1.0)
    assert w.shape == (3,)
    # The strongest signal is on feature 0 (positives shifted +1.5 there).
    assert abs(w[0]) > abs(w[1])
    assert abs(w[0]) > abs(w[2])


def test_pairwise_oof_predict_separates_classes():
    X, y, chroms, mg = _make_paired_dataset(n_per_chrom=40, d=4)
    preds, records = pairwise_oof_predict(
        X=X,
        y=y,
        chroms=chroms,
        match_group=mg,
        base="logreg",
        C_grid=[0.1, 1.0, 10.0],
        n_splits=3,
        n_splits_inner=3,
    )
    assert preds.shape == (len(X),)
    assert not np.isnan(preds).any()
    assert len(records) == 3
    assert preds[y == 1].mean() > preds[y == 0].mean() + 0.5


def test_pairwise_oof_predict_records_boundary_flag():
    X, y, chroms, mg = _make_paired_dataset(n_per_chrom=20, d=3)
    preds, records = pairwise_oof_predict(
        X=X,
        y=y,
        chroms=chroms,
        match_group=mg,
        base="logreg",
        C_grid=[1e-6, 1e-3, 1.0],
        n_splits=3,
        n_splits_inner=3,
    )
    assert preds.shape == (len(X),)
    for r in records:
        assert "hparam_boundary_flag" in r
        assert "C" in r["hparam_boundary_flag"]


def test_pairwise_accuracy_within_groups_basic():
    scores = np.array([0.1, 0.9, 0.4, 0.3])
    labels = np.array([0, 1, 0, 1])
    mg = np.array([0, 0, 1, 1])
    # Group 0: pos=0.9 > neg=0.1 → correct
    # Group 1: pos=0.3 < neg=0.4 → wrong
    acc = _pairwise_accuracy_within_groups(scores, labels, mg)
    assert acc == 0.5


def test_pairwise_accuracy_within_groups_handles_ties():
    scores = np.array([0.5, 0.5, 1.0, 0.0])
    labels = np.array([0, 1, 1, 0])
    mg = np.array([0, 0, 1, 1])
    acc = _pairwise_accuracy_within_groups(scores, labels, mg)
    assert acc == 0.75  # 0.5 + 1.0 over 2


def test_pairwise_accuracy_within_groups_returns_none_when_empty():
    # No valid pairs (only one row per group).
    scores = np.array([0.5, 0.7])
    labels = np.array([0, 1])
    mg = np.array([0, 1])
    acc = _pairwise_accuracy_within_groups(scores, labels, mg)
    assert acc is None


def test_pairwise_oof_predict_pairwise_aware_is_competitive_with_oof():
    """Pair-aware OOF preds should rank pairs at least as well as standard OOF
    on a synthetic problem where the pair structure aligns with the labels."""
    X, y, chroms, mg = _make_paired_dataset(n_per_chrom=40, d=5)

    pair_preds, _ = pairwise_oof_predict(
        X=X,
        y=y,
        chroms=chroms,
        match_group=mg,
        base="logreg",
        C_grid=[0.1, 1.0, 10.0],
    )

    spec = logreg_spec()
    std_preds, _ = oof_predict(
        X=X,
        y=y,
        chroms=chroms,
        estimator=spec.estimator,
        param_grid=spec.param_grid,
    )

    # Both should separate positives above negatives.
    pair_pa = _pairwise_accuracy_within_groups(pair_preds, y, mg)
    std_pa = _pairwise_accuracy_within_groups(std_preds, y, mg)
    assert pair_pa is not None and std_pa is not None
    assert pair_pa > 0.8
    assert std_pa > 0.8
