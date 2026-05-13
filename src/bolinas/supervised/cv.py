"""Chrom-grouped cross-validation + out-of-fold prediction helpers.

The supervised-VEP investigation evaluates classifier recipes via 3-fold
chrom-grouped CV inside the bolinas eval-dataset *train* splits (test held
out per #175 convention). For each outer fold we get OOF predictions for
every row; those predictions concatenate to one score per variant in the
input order, ready to plug into ``bolinas.evals.metrics.compute_pairwise_metrics``.

Inner CV (hyperparameter selection) runs on the outer-fold training set with
chroms again as groups, so no chrom appears in both the inner train and
inner validation.

This module is sklearn-only — no torch, no pandas-dependent IO. Pair-aware
classifiers (which operate on pair-difference features) live in
``bolinas.supervised.classifiers``.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GridSearchCV, GroupKFold


@dataclass(frozen=True)
class FoldAssignment:
    """Chrom-grouped fold split.

    Attributes:
        n_splits: number of outer folds.
        chrom_to_fold: ``{chrom: fold_index}`` for every chrom appearing in
            the dataset. Each chrom maps to exactly one fold.
        fold_to_chroms: inverse mapping, ``{fold_index: [chroms...]}``.
    """

    n_splits: int
    chrom_to_fold: dict[str, int]
    fold_to_chroms: dict[int, list[str]]

    def fold_index(self, chrom: str) -> int:
        return self.chrom_to_fold[chrom]


def assign_chrom_folds(chroms: Sequence[str], n_splits: int) -> FoldAssignment:
    """Deterministically assign each unique chrom to one of ``n_splits`` folds.

    Uses ``sklearn.model_selection.GroupKFold`` to do the partitioning — its
    behaviour is deterministic given the *order* of unique groups, so we
    sort chroms first for reproducibility across pandas versions / dtype
    inferences.

    Args:
        chroms: per-row chrom labels (length = N). Repeated values are fine.
        n_splits: number of folds. Must be ≤ number of unique chroms.

    Returns:
        FoldAssignment with every unique chrom assigned to exactly one fold.
    """
    chroms_arr = np.asarray(chroms)
    unique_chroms = sorted(np.unique(chroms_arr).tolist(), key=lambda c: str(c))
    if n_splits > len(unique_chroms):
        raise ValueError(
            f"n_splits={n_splits} exceeds number of unique chroms "
            f"({len(unique_chroms)})"
        )

    # Treat each unique chrom as one synthetic sample, with itself as the
    # group. GroupKFold then produces a partition over unique groups.
    dummy_X = np.zeros((len(unique_chroms), 1))
    gkf = GroupKFold(n_splits=n_splits)
    chrom_to_fold: dict[str, int] = {}
    for fold_idx, (_, test_idx) in enumerate(gkf.split(dummy_X, groups=unique_chroms)):
        for i in test_idx:
            chrom_to_fold[unique_chroms[i]] = fold_idx

    assert len(chrom_to_fold) == len(unique_chroms), (
        "every unique chrom must be assigned exactly once"
    )

    fold_to_chroms: dict[int, list[str]] = {f: [] for f in range(n_splits)}
    for c, f in chrom_to_fold.items():
        fold_to_chroms[f].append(c)

    return FoldAssignment(
        n_splits=n_splits,
        chrom_to_fold=chrom_to_fold,
        fold_to_chroms=fold_to_chroms,
    )


def chrom_kfold_indices(
    chroms: Sequence[str], n_splits: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` per outer fold for the given chrom labels.

    Indices are positional into the original ``chroms`` array. No chrom
    appears in both the train and test fold.
    """
    chroms_arr = np.asarray(chroms)
    folds = assign_chrom_folds(chroms_arr, n_splits=n_splits)
    all_idx = np.arange(len(chroms_arr))
    fold_per_row = np.array([folds.fold_index(c) for c in chroms_arr])
    for f in range(folds.n_splits):
        test_mask = fold_per_row == f
        yield all_idx[~test_mask], all_idx[test_mask]


def oof_predict(
    *,
    X: np.ndarray,
    y: np.ndarray,
    chroms: Sequence[str],
    estimator,
    param_grid: dict,
    n_splits: int = 3,
    n_splits_inner: int = 3,
    scoring: str = "average_precision",
    n_jobs: int = 1,
) -> tuple[np.ndarray, list[dict]]:
    """Run nested chrom-grouped CV and return one OOF prediction per row.

    For each outer fold:

    1. Build the train/test row sets so no chrom is in both.
    2. Run ``GridSearchCV`` over ``param_grid`` on the outer-train set, with
       inner CV = ``GroupKFold(n_splits_inner)`` on the outer-train chroms.
    3. Re-fit the best estimator on the full outer-train set (default for
       GridSearchCV) and predict on the outer-test set.

    Args:
        X: feature matrix, shape [N, F].
        y: binary labels in {0, 1}, shape [N].
        chroms: per-row chrom labels, length N.
        estimator: a sklearn-compatible estimator (or Pipeline).
        param_grid: passed to GridSearchCV.
        n_splits: outer folds.
        n_splits_inner: inner-CV folds for hyperparameter selection.
        scoring: GridSearchCV scoring string.
        n_jobs: GridSearchCV parallelism.

    Returns:
        ``(predictions, fold_records)``:

        * predictions: shape [N], score per row from the fold that *didn't*
          see this row in its training data. Higher = more likely positive
          (predict_proba[:, 1] when available, else decision_function).
        * fold_records: one dict per fold with ``{fold, best_params,
          best_inner_score, n_train, n_test, hparam_boundary_flag}``.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    chroms_arr = np.asarray(chroms)
    if X.shape[0] != y.shape[0] or X.shape[0] != len(chroms_arr):
        raise ValueError(
            f"row count mismatch: X={X.shape[0]} y={y.shape[0]} "
            f"chroms={len(chroms_arr)}"
        )

    predictions = np.full(X.shape[0], np.nan, dtype=np.float64)
    fold_records: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        chrom_kfold_indices(chroms_arr, n_splits=n_splits)
    ):
        assert set(chroms_arr[train_idx]).isdisjoint(set(chroms_arr[test_idx])), (
            f"fold {fold_idx} has chrom leakage between train and test"
        )

        inner_cv = GroupKFold(n_splits=n_splits_inner)
        search = GridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(
            X[train_idx],
            y[train_idx],
            groups=chroms_arr[train_idx],
        )

        best = search.best_estimator_
        if hasattr(best, "predict_proba"):
            scores = best.predict_proba(X[test_idx])[:, 1]
        elif hasattr(best, "decision_function"):
            scores = best.decision_function(X[test_idx])
        else:
            raise TypeError(
                f"estimator {type(best).__name__} has neither predict_proba "
                "nor decision_function"
            )
        predictions[test_idx] = scores

        boundary = _hparam_at_boundary(search.best_params_, param_grid)
        fold_records.append(
            {
                "fold": fold_idx,
                "best_params": dict(search.best_params_),
                "best_inner_score": float(search.best_score_),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "hparam_boundary_flag": boundary,
            }
        )

    assert not np.isnan(predictions).any(), "every row must have an OOF prediction"
    return predictions, fold_records


def _hparam_at_boundary(best_params: dict, param_grid: dict) -> dict[str, str | None]:
    """Per-hparam flag: 'low', 'high', or None depending on whether the
    selected value is at the min/max of its swept grid.

    Only flags numeric parameters where the grid is a list/array with at
    least 2 distinct values. Non-numeric or single-value grids report None.
    """
    flags: dict[str, str | None] = {}
    for k, chosen in best_params.items():
        grid_vals = param_grid.get(k, None)
        if grid_vals is None:
            flags[k] = None
            continue
        try:
            numeric = sorted(float(v) for v in grid_vals)
        except (TypeError, ValueError):
            flags[k] = None
            continue
        if len(numeric) < 2:
            flags[k] = None
        elif float(chosen) == numeric[0]:
            flags[k] = "low"
        elif float(chosen) == numeric[-1]:
            flags[k] = "high"
        else:
            flags[k] = None
    return flags
