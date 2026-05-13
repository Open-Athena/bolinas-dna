"""High-level entry points used by ``snakemake/analysis/supervised_vep/`` rules.

Keeps the Snakemake ``run:`` blocks thin (per CLAUDE.md) — they load a
parquet, call one of these helpers, write a parquet. All logic lives here so
``pytest`` can reach it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bolinas.evals.metrics import compute_pairwise_metrics
from bolinas.supervised.classifiers import (
    all_standard_specs,
    pairwise_oof_predict,
)
from bolinas.supervised.cv import oof_predict
from bolinas.supervised.features import RECIPE_IS_SYMMETRIC, build_features


# Names used as wildcards in the snakemake rules.
STANDARD_CLASSIFIERS = ("logreg_l2", "linearsvc", "knn", "xgboost")
PAIRWISE_CLASSIFIERS = ("pairwise_logreg",)
ALL_CLASSIFIERS = STANDARD_CLASSIFIERS + PAIRWISE_CLASSIFIERS


def fit_oof_predictions(
    *,
    features_df: pd.DataFrame,
    recipe: str,
    classifier: str,
    n_splits: int = 3,
    n_splits_inner: int = 3,
    mode: str = "bfs",
) -> tuple[pd.DataFrame, list[dict]]:
    """Run nested chrom-grouped CV and produce one OOF score per variant.

    Args:
        features_df: cache DataFrame with the columns produced by
            ``compute_pooled_features`` plus the carried variant metadata
            (``chrom, pos, ref, alt, label, subset, match_group, ...``).
        recipe: name in ``bolinas.supervised.features.RECIPE_BUILDERS``.
        classifier: one of ``STANDARD_CLASSIFIERS`` or ``PAIRWISE_CLASSIFIERS``.
        n_splits / n_splits_inner: outer / inner CV folds.
        mode: ``"bfs"`` (small grids) or ``"refine"`` (wide grids). Pair-aware
            classifiers get a similar 1-value vs 7-value grid.

    Returns:
        ``(predictions_df, fold_records)``.

        * ``predictions_df`` has columns ``[chrom, pos, ref, alt, label,
          subset, match_group, score]`` aligned with ``features_df`` by index.
          ``score`` is the OOF prediction for the fold that didn't train on
          this variant.
        * ``fold_records`` is the list returned by the underlying
          ``oof_predict`` / ``pairwise_oof_predict``; serialise to JSON.
    """
    X = build_features(features_df, recipe)
    y = features_df["label"].to_numpy()
    chroms = features_df["chrom"].to_numpy()
    mg = features_df["match_group"].to_numpy()

    if classifier in STANDARD_CLASSIFIERS:
        spec = _spec_by_name(classifier, n_train=X.shape[0], mode=mode)
        predictions, fold_records = oof_predict(
            X=X,
            y=y,
            chroms=chroms,
            estimator=spec.estimator,
            param_grid=spec.param_grid,
            n_splits=n_splits,
            n_splits_inner=n_splits_inner,
        )
    elif classifier == "pairwise_logreg":
        if mode == "bfs":
            C_grid = (1.0,)
        elif mode == "refine":
            C_grid = tuple(10.0 ** np.arange(-4, 3))
        else:
            raise ValueError(f"unknown mode {mode!r}")
        predictions, fold_records = pairwise_oof_predict(
            X=X,
            y=y,
            chroms=chroms,
            match_group=mg,
            base="logreg",
            C_grid=C_grid,
            n_splits=n_splits,
            n_splits_inner=n_splits_inner,
        )
    else:
        raise ValueError(
            f"unknown classifier {classifier!r}; expected one of {ALL_CLASSIFIERS}"
        )

    preds_df = features_df[
        ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
    ].copy()
    preds_df["score"] = predictions
    return preds_df, fold_records


def compute_metrics_from_oof(
    predictions_df: pd.DataFrame,
    score_column: str = "score",
) -> pd.DataFrame:
    """Score OOF predictions through ``compute_pairwise_metrics``.

    Returns a long-form DataFrame with PairwiseAccuracy per subset, plus the
    ``_global_`` and ``_macro_avg_`` rows. Identical schema to ``evals_v2``
    metrics, which lets us drop these into the same leaderboards.
    """
    meta_cols = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
    return compute_pairwise_metrics(
        dataset=predictions_df[[c for c in meta_cols if c in predictions_df.columns]],
        scores=predictions_df[[score_column]],
        score_columns=[score_column],
    )


def compute_zeroshot_baseline_metrics(
    features_df: pd.DataFrame,
    *,
    score_columns: tuple[str, ...] = ("minus_llr", "abs_llr", "embed_last_l2"),
) -> pd.DataFrame:
    """Run each zero-shot scalar score directly through ``compute_pairwise_metrics``.

    The supervised investigation's headline comparison is "supervised OOF vs.
    zero-shot baseline" on the same train split, so this is essentially the
    no-supervised-head row in the leaderboard.
    """
    score_cols = [c for c in score_columns if c in features_df.columns]
    meta_cols = ["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]
    return compute_pairwise_metrics(
        dataset=features_df[[c for c in meta_cols if c in features_df.columns]],
        scores=features_df[score_cols],
        score_columns=score_cols,
    )


def write_fold_records_json(path: str, fold_records: list[dict]) -> None:
    """Serialise fold-records to JSON with non-trivial dtypes coerced to Python primitives."""
    with open(path, "w") as f:
        json.dump(fold_records, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"unhandled type for JSON: {type(obj)}")


def _spec_by_name(name: str, n_train: int, mode: str):
    """Look up a ``ClassifierSpec`` factory by name and instantiate it."""
    specs = {s.name: s for s in all_standard_specs(mode=mode)}
    if name == "knn":
        # KNN's grid scales with n_train, so re-instantiate explicitly.
        from bolinas.supervised.classifiers import knn_spec

        return knn_spec(n_train=n_train, mode=mode)
    if name not in specs:
        raise ValueError(f"no standard spec named {name!r}; available: {sorted(specs)}")
    return specs[name]


def is_recipe_dataset_compatible(recipe: str, dataset_name: str) -> bool:
    """Asymmetric recipes only apply to ``mendelian_traits`` (ref/alt direction is defined there).

    For ``complex_traits`` and ``eqtl`` the ref/alt designation is arbitrary
    in fine-mapping, so features must be ref↔alt-symmetric.
    """
    if recipe not in RECIPE_IS_SYMMETRIC:
        raise KeyError(f"unknown recipe {recipe!r}")
    if RECIPE_IS_SYMMETRIC[recipe]:
        return True
    return dataset_name == "mendelian_traits"
