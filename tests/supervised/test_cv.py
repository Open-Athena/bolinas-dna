"""Tests for ``bolinas.supervised.cv`` chrom-grouped K-fold + OOF predict."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from bolinas.supervised.cv import (
    FoldAssignment,
    _hparam_at_boundary,
    assign_chrom_folds,
    chrom_kfold_indices,
    oof_predict,
)


# 12 train-split chroms in the bolinas eval datasets.
TWELVE_CHROMS = ["1", "11", "13", "15", "17", "19", "21", "3", "5", "7", "9", "X"]


def test_assign_chrom_folds_each_chrom_assigned_exactly_once():
    folds = assign_chrom_folds(TWELVE_CHROMS, n_splits=3)
    assert folds.n_splits == 3
    assert set(folds.chrom_to_fold.keys()) == set(TWELVE_CHROMS)
    fold_counts = [len(folds.fold_to_chroms[f]) for f in range(3)]
    assert sum(fold_counts) == 12
    # With 12 chroms / 3 folds, each fold should hold 4 chroms exactly.
    assert fold_counts == [4, 4, 4]


def test_assign_chrom_folds_is_deterministic():
    """Repeated calls should produce identical assignments."""
    a = assign_chrom_folds(TWELVE_CHROMS, n_splits=3)
    b = assign_chrom_folds(TWELVE_CHROMS, n_splits=3)
    assert a == b


def test_assign_chrom_folds_invariant_to_input_order():
    """The chrom→fold map should not depend on the order of the input rows."""
    a = assign_chrom_folds(TWELVE_CHROMS, n_splits=3)
    shuffled = list(reversed(TWELVE_CHROMS))
    b = assign_chrom_folds(shuffled, n_splits=3)
    assert a.chrom_to_fold == b.chrom_to_fold


def test_assign_chrom_folds_rejects_too_many_splits():
    with pytest.raises(ValueError, match="exceeds number of unique chroms"):
        assign_chrom_folds(["1", "2"], n_splits=3)


def test_chrom_kfold_indices_partitions_rows_with_no_chrom_leakage():
    rng = np.random.default_rng(0)
    n = 200
    chroms = rng.choice(TWELVE_CHROMS, size=n)
    seen_test = np.zeros(n, dtype=bool)
    for train_idx, test_idx in chrom_kfold_indices(chroms, n_splits=3):
        assert len(set(train_idx) & set(test_idx)) == 0
        train_chroms = set(chroms[train_idx])
        test_chroms = set(chroms[test_idx])
        assert train_chroms.isdisjoint(test_chroms), (
            f"chrom leakage: {train_chroms & test_chroms}"
        )
        seen_test[test_idx] = True
    assert seen_test.all(), "every row must appear in exactly one test fold"


def _make_synthetic_classification_problem(
    n_per_chrom: int = 40, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A simple linearly-separable binary problem replicated across chroms.

    Returns ``(X, y, chroms)``. Within each chrom, ``X[:, 0]`` separates the
    classes; chrom assignment is independent of label.
    """
    rng = np.random.default_rng(seed)
    Xs, ys, cs = [], [], []
    for c in TWELVE_CHROMS:
        y_c = np.concatenate([np.zeros(n_per_chrom // 2), np.ones(n_per_chrom // 2)])
        # Class-conditional means with shared noise.
        x0 = y_c * 2.0 + rng.normal(0, 1, size=n_per_chrom)
        x_noise = rng.normal(0, 1, size=(n_per_chrom, 4))
        X_c = np.column_stack([x0, x_noise])
        Xs.append(X_c)
        ys.append(y_c)
        cs.extend([c] * n_per_chrom)
    return np.vstack(Xs), np.concatenate(ys).astype(int), np.array(cs)


def test_oof_predict_produces_one_score_per_row_with_no_nan():
    X, y, chroms = _make_synthetic_classification_problem()
    preds, records = oof_predict(
        X=X,
        y=y,
        chroms=chroms,
        estimator=LogisticRegression(max_iter=200),
        param_grid={"C": [0.1, 1.0, 10.0]},
        n_splits=3,
        n_splits_inner=3,
    )
    assert preds.shape == (X.shape[0],)
    assert not np.isnan(preds).any()
    assert len(records) == 3
    for r in records:
        assert "best_params" in r
        assert "best_inner_score" in r
        assert "hparam_boundary_flag" in r
        # n_train + n_test == N
        assert r["n_train"] + r["n_test"] == X.shape[0]


def test_oof_predict_separates_classes_on_easy_synthetic():
    """OOF predictions should rank positives above negatives on average."""
    X, y, chroms = _make_synthetic_classification_problem()
    preds, _ = oof_predict(
        X=X,
        y=y,
        chroms=chroms,
        estimator=LogisticRegression(max_iter=200),
        param_grid={"C": [1.0]},
    )
    mean_pos = preds[y == 1].mean()
    mean_neg = preds[y == 0].mean()
    assert mean_pos > mean_neg + 0.2, (
        f"OOF preds don't separate classes: pos={mean_pos:.3f}, neg={mean_neg:.3f}"
    )


def test_oof_predict_rejects_size_mismatch():
    X = np.zeros((10, 3))
    y = np.zeros(9)
    chroms = ["1"] * 10
    with pytest.raises(ValueError, match="row count mismatch"):
        oof_predict(
            X=X,
            y=y,
            chroms=chroms,
            estimator=LogisticRegression(),
            param_grid={"C": [1.0]},
        )


def test_hparam_at_boundary_flags_min_and_max():
    grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    assert _hparam_at_boundary({"C": 0.01}, grid) == {"C": "low"}
    assert _hparam_at_boundary({"C": 10.0}, grid) == {"C": "high"}
    assert _hparam_at_boundary({"C": 1.0}, grid) == {"C": None}


def test_hparam_at_boundary_returns_none_for_single_value_grid():
    assert _hparam_at_boundary({"C": 1.0}, {"C": [1.0]}) == {"C": None}


def test_hparam_at_boundary_handles_non_numeric():
    assert _hparam_at_boundary({"k": "linear"}, {"k": ["linear", "rbf"]}) == {"k": None}


def test_fold_assignment_dataclass_is_hashable_frozen():
    folds = assign_chrom_folds(TWELVE_CHROMS, n_splits=3)
    assert isinstance(folds, FoldAssignment)
    # Frozen, so reassignment should raise.
    with pytest.raises(Exception):
        folds.n_splits = 5  # type: ignore[misc]
