"""Tests for ``bolinas.supervised.pipeline`` — the snakemake entry points.

Uses synthetic cache-shaped DataFrames so the test is fast and self-contained.
"""

import json
import tempfile

import numpy as np
import pandas as pd
import pytest

from bolinas.supervised.pipeline import (
    ALL_CLASSIFIERS,
    PAIRWISE_CLASSIFIERS,
    STANDARD_CLASSIFIERS,
    compute_metrics_from_oof,
    compute_zeroshot_baseline_metrics,
    fit_oof_predictions,
    is_recipe_dataset_compatible,
    write_fold_records_json,
)


TWELVE_CHROMS = ["1", "11", "13", "15", "17", "19", "21", "3", "5", "7", "9", "X"]


def _make_paired_cache_df(
    n_per_chrom: int = 40, d: int = 6, seed: int = 0, subsets: int = 2
) -> pd.DataFrame:
    """Synthetic supervised-VEP feature cache for testing.

    Each chrom contributes ``n_per_chrom`` rows, organised as
    ``n_per_chrom // 2`` matched pairs. Positives are shifted +1.5 on
    embedding channel 0 (after pooling) so a linear classifier ranks them
    above their matched negatives.
    """
    rng = np.random.default_rng(seed)
    rows = []
    next_mg = 0
    for c in TWELVE_CHROMS:
        n_pairs = n_per_chrom // 2
        for i in range(n_pairs):
            mean_ref_neg = rng.normal(size=d).astype(np.float32)
            mean_alt_neg = rng.normal(size=d).astype(np.float32)
            # Positive: same mean_ref, mean_alt shifted on dim 0.
            mean_alt_pos = mean_alt_neg.copy()
            mean_alt_pos[0] += 1.5 + rng.normal(0, 0.5)
            innerprod_neg = mean_ref_neg * mean_alt_neg
            innerprod_pos = mean_ref_neg * mean_alt_pos
            llr_neg = float(rng.normal(0, 0.1))
            llr_pos = float(rng.normal(0.5, 0.3))
            subset_name = f"sub{i % subsets}"
            rows.append(
                {
                    "chrom": c,
                    "pos": 1000 + i,
                    "ref": "A",
                    "alt": "G",
                    "label": False,
                    "subset": subset_name,
                    "match_group": next_mg,
                    "llr": llr_neg,
                    "minus_llr": -llr_neg,
                    "abs_llr": abs(llr_neg),
                    "embed_last_l2": float(rng.uniform(0.0, 0.5)),
                    "mean_ref": mean_ref_neg,
                    "mean_alt": mean_alt_neg,
                    "traitgym_innerprod": innerprod_neg,
                }
            )
            rows.append(
                {
                    "chrom": c,
                    "pos": 1000 + i,
                    "ref": "A",
                    "alt": "G",
                    "label": True,
                    "subset": subset_name,
                    "match_group": next_mg,
                    "llr": llr_pos,
                    "minus_llr": -llr_pos,
                    "abs_llr": abs(llr_pos),
                    "embed_last_l2": float(rng.uniform(0.5, 2.0)),
                    "mean_ref": mean_ref_neg,
                    "mean_alt": mean_alt_pos,
                    "traitgym_innerprod": innerprod_pos,
                }
            )
            next_mg += 1
    return pd.DataFrame(rows)


@pytest.mark.parametrize("classifier", STANDARD_CLASSIFIERS)
def test_fit_oof_predictions_standard_classifier_returns_expected_schema(classifier):
    if classifier == "xgboost":
        pytest.importorskip("xgboost")
    df = _make_paired_cache_df(n_per_chrom=20, d=4)
    preds, records = fit_oof_predictions(
        features_df=df,
        recipe="abs_delta",
        classifier=classifier,
        n_splits=3,
        n_splits_inner=3,
        mode="bfs",
    )
    assert set(preds.columns) == {
        "chrom",
        "pos",
        "ref",
        "alt",
        "label",
        "subset",
        "match_group",
        "score",
    }
    assert len(preds) == len(df)
    assert not preds["score"].isna().any()
    assert len(records) == 3
    for r in records:
        assert "best_params" in r
        assert "hparam_boundary_flag" in r


@pytest.mark.parametrize("classifier", PAIRWISE_CLASSIFIERS)
def test_fit_oof_predictions_pairwise_classifier(classifier):
    df = _make_paired_cache_df(n_per_chrom=20, d=4)
    preds, records = fit_oof_predictions(
        features_df=df,
        recipe="abs_delta",
        classifier=classifier,
        n_splits=3,
        n_splits_inner=3,
        mode="bfs",
    )
    assert len(preds) == len(df)
    assert not preds["score"].isna().any()
    assert len(records) == 3


def test_fit_oof_predictions_rejects_unknown_classifier():
    df = _make_paired_cache_df(n_per_chrom=20, d=3)
    with pytest.raises(ValueError, match="unknown classifier"):
        fit_oof_predictions(
            features_df=df,
            recipe="abs_delta",
            classifier="lightgbm",
        )


def test_compute_metrics_from_oof_returns_pairwise_table():
    df = _make_paired_cache_df(n_per_chrom=20, d=4)
    preds, _ = fit_oof_predictions(
        features_df=df,
        recipe="abs_delta",
        classifier="logreg_l2",
        mode="bfs",
    )
    metrics = compute_metrics_from_oof(preds)
    # Schema from compute_pairwise_metrics.
    expected_cols = {"score_type", "subset", "value", "se", "n_pairs", "n_ties"}
    assert expected_cols.issubset(metrics.columns)
    # _global_ and _macro_avg_ rows present.
    assert "_global_" in metrics["subset"].tolist()


def test_compute_zeroshot_baseline_metrics_returns_one_row_per_score_column():
    df = _make_paired_cache_df(n_per_chrom=20, d=4)
    metrics = compute_zeroshot_baseline_metrics(
        df, score_columns=("minus_llr", "abs_llr", "embed_last_l2")
    )
    score_types = set(metrics["score_type"].tolist())
    assert {"minus_llr", "abs_llr", "embed_last_l2"} <= score_types


def test_is_recipe_dataset_compatible_for_symmetric_recipes():
    for ds in ["mendelian_traits", "complex_traits", "eqtl"]:
        assert is_recipe_dataset_compatible("abs_delta", ds)
        assert is_recipe_dataset_compatible("sym_concat", ds)


def test_is_recipe_dataset_compatible_for_asymmetric_recipes():
    assert is_recipe_dataset_compatible("mean_delta", "mendelian_traits")
    assert not is_recipe_dataset_compatible("mean_delta", "complex_traits")
    assert not is_recipe_dataset_compatible("mean_delta", "eqtl")


def test_is_recipe_dataset_compatible_rejects_unknown_recipe():
    with pytest.raises(KeyError, match="unknown recipe"):
        is_recipe_dataset_compatible("nonsense", "mendelian_traits")


def test_write_fold_records_json_round_trips():
    records = [
        {
            "fold": 0,
            "best_params": {"clf__C": 1.0},
            "best_inner_score": np.float64(0.78),
            "n_train": np.int64(400),
            "n_test": 100,
            "hparam_boundary_flag": {"clf__C": None},
        },
    ]
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
        write_fold_records_json(f.name, records)
        with open(f.name) as g:
            loaded = json.load(g)
    assert loaded == [
        {
            "fold": 0,
            "best_params": {"clf__C": 1.0},
            "best_inner_score": 0.78,
            "n_train": 400,
            "n_test": 100,
            "hparam_boundary_flag": {"clf__C": None},
        },
    ]


def test_all_classifiers_constant_covers_standard_and_pairwise():
    assert set(ALL_CLASSIFIERS) == set(STANDARD_CLASSIFIERS) | set(PAIRWISE_CLASSIFIERS)
