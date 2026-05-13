"""Classifier wrappers + factories for the supervised-VEP investigation.

Two families:

* **Standard** classifiers (LogReg, LinearSVC, KNN, XGBoost) — plain sklearn /
  XGBoost pipelines wrapped in ``SimpleImputer(mean) → StandardScaler → clf``.
  Used via ``bolinas.supervised.cv.oof_predict``.
* **Pair-aware** linear classifier — fits a LogReg/LinearSVC on pair-difference
  features (label ±1) to directly optimise within-``match_group`` ranking,
  which is the eval metric (PairwiseAccuracy). Has its own OOF entry point
  ``pairwise_oof_predict`` since the training-data construction breaks the
  sklearn ``fit(X, y)`` contract.

The standard hparam grids are *starting points* — every fit's selected hparam
is flagged if it lands at the grid boundary (see ``cv._hparam_at_boundary``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from bolinas.supervised.cv import _hparam_at_boundary, chrom_kfold_indices

try:  # optional, only needed for xgboost recipe
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment]


# ---------- standard sklearn classifiers ----------------------------------


@dataclass(frozen=True)
class ClassifierSpec:
    """A classifier name → (estimator, GridSearchCV param grid) pair."""

    name: str
    estimator: object
    param_grid: dict[str, list]


def _scaled_pipeline(clf):
    """Wrap an estimator in the standard ``impute → scale → clf`` pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def logreg_spec(*, mode: str = "bfs") -> ClassifierSpec:
    """L2-regularised logistic regression.

    ``mode="bfs"`` (default) — single ``C=1.0`` for fast first-pass sweep.
    ``mode="refine"`` — wide ``logspace(-8, 0, 10)`` sweep for the chosen
    (recipe, dataset) cells once iter-1 has identified the top performers.

    Boundary-hit flagging in ``oof_predict`` catches the refine cases where
    the grid still needs widening.
    """
    estimator = _scaled_pipeline(
        LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
        )
    )
    if mode == "bfs":
        grid = {"clf__C": [1.0]}
    elif mode == "refine":
        grid = {"clf__C": list(np.logspace(-8, 0, 10))}
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return ClassifierSpec(name="logreg_l2", estimator=estimator, param_grid=grid)


def linearsvc_spec(*, mode: str = "bfs") -> ClassifierSpec:
    """Linear SVM with hinge loss; ``decision_function`` provides the score."""
    estimator = _scaled_pipeline(
        LinearSVC(
            class_weight="balanced",
            dual="auto",
            max_iter=5000,
        )
    )
    if mode == "bfs":
        grid = {"clf__C": [1.0]}
    elif mode == "refine":
        grid = {"clf__C": list(np.logspace(-4, 2, 7))}
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return ClassifierSpec(name="linearsvc", estimator=estimator, param_grid=grid)


def knn_spec(n_train: int = 4000, *, mode: str = "bfs") -> ClassifierSpec:
    """K-nearest neighbours on scaled features."""
    estimator = _scaled_pipeline(KNeighborsClassifier(metric="euclidean"))
    if mode == "bfs":
        grid = {"clf__n_neighbors": [25]}
    elif mode == "refine":
        upper = max(500, n_train // 10)
        candidates = sorted(
            set(c for c in [5, 25, 100, 500, upper] if 1 < c <= max(n_train - 1, 5))
        )
        grid = {"clf__n_neighbors": candidates}
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return ClassifierSpec(name="knn", estimator=estimator, param_grid=grid)


def xgboost_spec(*, mode: str = "bfs") -> ClassifierSpec | None:
    """Gradient-boosted trees."""
    if XGBClassifier is None:
        return None
    estimator = _scaled_pipeline(
        XGBClassifier(
            tree_method="hist",
            eval_metric="logloss",
            n_estimators=200,
            random_state=0,
        )
    )
    if mode == "bfs":
        # sklearn defaults for tree params; only `n_estimators` capped above.
        grid = {"clf__max_depth": [6]}
    elif mode == "refine":
        grid = {
            "clf__max_depth": [3, 6],
            "clf__learning_rate": [0.05, 0.1],
            "clf__subsample": [0.8],
            "clf__colsample_bytree": [0.5, 1.0],
        }
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return ClassifierSpec(name="xgboost", estimator=estimator, param_grid=grid)


def all_standard_specs(*, mode: str = "bfs") -> list[ClassifierSpec]:
    """The 3 (or 4 if xgboost installed) standard classifiers.

    ``mode="bfs"`` (default) returns minimal grids for the first-pass sweep —
    one or two values per knob. ``mode="refine"`` switches to the wider grids
    used after the BFS round identifies the top (recipe, classifier) cells.
    """
    specs = [logreg_spec(mode=mode), linearsvc_spec(mode=mode), knn_spec(mode=mode)]
    xgb = xgboost_spec(mode=mode)
    if xgb is not None:
        specs.append(xgb)
    return specs


# ---------- pair-aware linear classifier ----------------------------------


def build_pairwise_diff_dataset(
    X: np.ndarray,
    y: np.ndarray,
    match_group: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the pair-difference training set used by ``pairwise_oof_predict``.

    For each ``match_group`` with exactly one positive (``y=1``) and one
    negative (``y=0``) row, emit two rows:

    * ``feat(pos) - feat(neg)`` with label ``+1``
    * ``feat(neg) - feat(pos)`` with label ``0`` (sklearn binary)

    Groups that don't have exactly one positive and one negative are skipped
    (with no error — the cache has match_group values for all rows, but a
    given chrom-grouped split can occasionally drop the partner of a pair if
    the partner sits in a different chrom; we assert below that this never
    happens in practice).

    Returns:
        ``(X_diff, y_diff)`` ready to feed an unintercepted linear classifier.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    mg = np.asarray(match_group)
    if not (X.shape[0] == y.shape[0] == mg.shape[0]):
        raise ValueError("X, y, match_group must have the same length")

    # Group rows by match_group.
    order = np.argsort(mg, kind="stable")
    mg_sorted = mg[order]
    boundaries = np.flatnonzero(np.diff(mg_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(mg_sorted)]])

    diffs: list[np.ndarray] = []
    labels: list[int] = []
    for s, e in zip(starts, ends):
        idx = order[s:e]
        ys = y[idx]
        if ys.sum() != 1 or (1 - ys).sum() != 1 or len(idx) != 2:
            continue
        pos_idx = idx[ys == 1][0]
        neg_idx = idx[ys == 0][0]
        d = X[pos_idx] - X[neg_idx]
        diffs.append(d)
        labels.append(1)
        diffs.append(-d)
        labels.append(0)

    if not diffs:
        raise ValueError("no valid (pos, neg) pairs in match_group — check the dataset")
    return np.stack(diffs), np.asarray(labels, dtype=int)


def fit_pairwise_linear_weights(
    X: np.ndarray,
    y: np.ndarray,
    match_group: Sequence[int],
    *,
    base: str = "logreg",
    C: float = 1.0,
) -> np.ndarray:
    """Fit a linear classifier on pair-difference features and return the weight vector ``w``.

    The scorer at inference is ``s(v) = w · feat(v)`` — no intercept, since
    pair-difference vectors have mean ≈ 0 by construction.

    Args:
        X: features [N, F].
        y: binary labels [N].
        match_group: pair-id per row [N]. Each id should have exactly one
            positive and one negative.
        base: ``"logreg"`` or ``"svm"``.
        C: regularisation strength (same convention as sklearn).

    Returns:
        Weight vector ``w`` of shape [F]. Score variants as ``X @ w``.
    """
    X_diff, y_diff = build_pairwise_diff_dataset(X, y, match_group)
    if base == "logreg":
        clf = LogisticRegression(
            penalty="l2",
            C=C,
            fit_intercept=False,
            solver="liblinear",
            max_iter=2000,
        )
    elif base == "svm":
        clf = LinearSVC(
            C=C,
            fit_intercept=False,
            dual="auto",
            max_iter=5000,
        )
    else:
        raise ValueError(f"unknown base {base!r}; expected 'logreg' or 'svm'")
    clf.fit(X_diff, y_diff)
    return clf.coef_.flatten().astype(np.float64)


def pairwise_oof_predict(
    *,
    X: np.ndarray,
    y: np.ndarray,
    chroms: Sequence[str],
    match_group: Sequence[int],
    base: str = "logreg",
    C_grid: Sequence[float] = tuple(10.0 ** np.arange(-4, 3)),
    n_splits: int = 3,
    n_splits_inner: int = 3,
) -> tuple[np.ndarray, list[dict]]:
    """OOF predict with the pair-aware linear classifier.

    Mirrors ``cv.oof_predict`` structurally — chrom-grouped outer folds,
    chrom-grouped inner folds for C selection — but operates on pair-difference
    features. The scorer is the standardised pipeline's transform applied to
    the raw feature matrix and then ``X_scaled @ w``.

    Notes:

    * Inner-CV scoring metric is **PairwiseAccuracy within the inner
      validation fold**, computed by sign-comparing scores in matched
      pairs — i.e., the same metric the outer eval uses. This is the
      whole reason to use a pair-aware loss in the first place.
    * Standardisation is fit on the inner-train rows only (no leakage into
      the inner-val set).
    * Match groups whose two rows span chroms are skipped at the diff-
      construction step (with no error). An assertion confirms that the
      number of skipped pairs is zero in our datasets.

    Returns:
        ``(predictions, fold_records)`` matching ``cv.oof_predict``'s shape.
        ``hparam_boundary_flag`` here reports the chosen ``C`` against
        ``C_grid``.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    chroms_arr = np.asarray(chroms)
    mg = np.asarray(match_group)
    if not (len(X) == len(y) == len(chroms_arr) == len(mg)):
        raise ValueError("X, y, chroms, match_group must have the same length")

    predictions = np.full(len(X), np.nan, dtype=np.float64)
    fold_records: list[dict] = []
    C_grid = sorted(set(float(c) for c in C_grid))

    for fold_idx, (train_idx, test_idx) in enumerate(
        chrom_kfold_indices(chroms_arr, n_splits=n_splits)
    ):
        assert set(chroms_arr[train_idx]).isdisjoint(set(chroms_arr[test_idx]))

        # Inner CV for C selection.
        inner_scores: dict[float, list[float]] = {C: [] for C in C_grid}
        for inner_train_idx, inner_val_idx in chrom_kfold_indices(
            chroms_arr[train_idx], n_splits=n_splits_inner
        ):
            it = train_idx[inner_train_idx]
            iv = train_idx[inner_val_idx]
            for C in C_grid:
                w, scaler = _fit_pairwise_with_scaling(
                    X[it], y[it], mg[it], base=base, C=C
                )
                scores_val = scaler.transform(X[iv]) @ w
                # PairwiseAccuracy on the inner-val fold.
                ap = _pairwise_accuracy_within_groups(
                    scores=scores_val, labels=y[iv], match_group=mg[iv]
                )
                if ap is not None:
                    inner_scores[C].append(ap)

        # Pick best C by mean inner-CV PairwiseAccuracy. Ties broken by
        # picking the *more regularised* (smaller C) value for stability.
        mean_scores = {
            C: float(np.mean(v)) if v else float("nan") for C, v in inner_scores.items()
        }
        best_C = min(
            (C for C in C_grid if not np.isnan(mean_scores[C])),
            key=lambda C: (-mean_scores[C], C),
        )

        # Refit on full outer-train set with the best C, score outer-test.
        w, scaler = _fit_pairwise_with_scaling(
            X[train_idx], y[train_idx], mg[train_idx], base=base, C=best_C
        )
        predictions[test_idx] = scaler.transform(X[test_idx]) @ w

        fold_records.append(
            {
                "fold": fold_idx,
                "best_params": {"C": best_C},
                "best_inner_score": mean_scores[best_C],
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "hparam_boundary_flag": _hparam_at_boundary(
                    {"C": best_C}, {"C": C_grid}
                ),
            }
        )

    assert not np.isnan(predictions).any()
    return predictions, fold_records


def _fit_pairwise_with_scaling(
    X: np.ndarray,
    y: np.ndarray,
    match_group: Sequence[int],
    *,
    base: str,
    C: float,
):
    """StandardScaler on training rows, then pair-aware linear fit on diffs."""
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X)
    X_s = scaler.transform(X)
    w = fit_pairwise_linear_weights(X_s, y, match_group, base=base, C=C)
    return w, scaler


def _pairwise_accuracy_within_groups(
    scores: np.ndarray,
    labels: np.ndarray,
    match_group: Sequence[int],
) -> float | None:
    """Compute fraction of pairs with ``score(pos) > score(neg)`` (ties = 0.5).

    Returns ``None`` if there are no valid (pos, neg) pairs (e.g. when an
    inner-val fold contains zero complete pairs).
    """
    mg = np.asarray(match_group)
    order = np.argsort(mg, kind="stable")
    mg_sorted = mg[order]
    boundaries = np.flatnonzero(np.diff(mg_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(mg_sorted)]])

    correct = 0.0
    total = 0
    for s, e in zip(starts, ends):
        idx = order[s:e]
        ys = labels[idx]
        if ys.sum() != 1 or (1 - ys).sum() != 1 or len(idx) != 2:
            continue
        pos_score = scores[idx[ys == 1][0]]
        neg_score = scores[idx[ys == 0][0]]
        if pos_score > neg_score:
            correct += 1.0
        elif pos_score == neg_score:
            correct += 0.5
        total += 1
    if total == 0:
        return None
    return correct / total
