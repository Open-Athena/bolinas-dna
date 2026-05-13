"""Tests for ``bolinas.supervised.features`` recipe builders + symmetry invariants."""

import numpy as np
import pandas as pd
import pytest

from bolinas.supervised.features import (
    ASYMMETRIC_RECIPES,
    RECIPE_BUILDERS,
    RECIPE_IS_SYMMETRIC,
    SYMMETRIC_RECIPES,
    build_features,
    ref_alt_swap,
)


def _make_cache_df(n: int = 8, d: int = 5, seed: int = 0) -> pd.DataFrame:
    """Synthetic supervised-VEP cache DataFrame: 4 scalar cols + 3 list cols."""
    rng = np.random.default_rng(seed)
    mean_ref = rng.normal(size=(n, d)).astype(np.float32)
    mean_alt = rng.normal(size=(n, d)).astype(np.float32)
    innerprod = rng.normal(size=(n, d)).astype(np.float32)
    llr = rng.normal(size=n).astype(np.float32)
    return pd.DataFrame(
        {
            "ref": ["A"] * n,
            "alt": ["G"] * n,
            "llr": llr,
            "minus_llr": -llr,
            "abs_llr": np.abs(llr),
            "embed_last_l2": rng.uniform(0, 5, size=n).astype(np.float32),
            "mean_ref": list(mean_ref),
            "mean_alt": list(mean_alt),
            "traitgym_innerprod": list(innerprod),
        }
    )


@pytest.mark.parametrize("recipe", sorted(RECIPE_BUILDERS))
def test_recipe_returns_2d_float_array_with_correct_row_count(recipe):
    df = _make_cache_df(n=12, d=4)
    X = build_features(df, recipe)
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    assert X.shape[0] == len(df)
    assert np.issubdtype(X.dtype, np.floating)
    assert not np.isnan(X).any()


@pytest.mark.parametrize("recipe", SYMMETRIC_RECIPES)
def test_symmetric_recipes_are_invariant_under_ref_alt_swap(recipe):
    """The whole point of the symmetric recipes: swap(ref,alt) → same features."""
    df = _make_cache_df(n=12, d=5)
    X_orig = build_features(df, recipe)
    X_swap = build_features(ref_alt_swap(df), recipe)
    np.testing.assert_allclose(X_orig, X_swap, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("recipe", ASYMMETRIC_RECIPES)
def test_asymmetric_recipes_change_under_ref_alt_swap(recipe):
    """Asymmetric recipes should NOT match after swap (otherwise they're mislabeled)."""
    df = _make_cache_df(n=12, d=5)
    X_orig = build_features(df, recipe)
    X_swap = build_features(ref_alt_swap(df), recipe)
    diff = np.abs(X_orig - X_swap).max()
    assert diff > 1e-4, f"asymmetric recipe {recipe!r} looks invariant under swap"


def test_build_abs_delta_shape_and_values():
    df = _make_cache_df(n=4, d=3)
    X = build_features(df, "abs_delta")
    assert X.shape == (4, 3)
    expected = np.abs(np.stack(df["mean_alt"]) - np.stack(df["mean_ref"]))
    np.testing.assert_allclose(X, expected)


def test_build_prod_shape_and_values():
    df = _make_cache_df(n=4, d=3)
    X = build_features(df, "prod")
    assert X.shape == (4, 3)
    expected = np.stack(df["mean_ref"]) * np.stack(df["mean_alt"])
    np.testing.assert_allclose(X, expected)


def test_build_sym_concat_doubles_dim():
    df = _make_cache_df(n=4, d=6)
    X = build_features(df, "sym_concat")
    assert X.shape == (4, 12)


def test_build_traitgym_innerprod_passes_through():
    df = _make_cache_df(n=4, d=3)
    X = build_features(df, "traitgym_innerprod")
    expected = np.stack(df["traitgym_innerprod"])
    np.testing.assert_allclose(X, expected)


def test_build_abs_delta_plus_scalars_appends_two_columns():
    df = _make_cache_df(n=4, d=3)
    X = build_features(df, "abs_delta_plus_scalars")
    assert X.shape == (4, 3 + 2)
    # Last two columns are abs_llr and embed_last_l2.
    np.testing.assert_allclose(X[:, -2], np.abs(df["llr"].to_numpy()))
    np.testing.assert_allclose(X[:, -1], df["embed_last_l2"].to_numpy())


def test_build_mean_delta_triples_dim():
    df = _make_cache_df(n=4, d=5)
    X = build_features(df, "mean_delta")
    assert X.shape == (4, 15)


def test_build_mean_concat_plus_llr_adds_one_column():
    df = _make_cache_df(n=4, d=5)
    X = build_features(df, "mean_concat_plus_llr")
    assert X.shape == (4, 11)  # 2*5 + 1
    np.testing.assert_allclose(X[:, -1], df["llr"].to_numpy())


def test_unknown_recipe_raises():
    df = _make_cache_df()
    with pytest.raises(KeyError, match="unknown recipe"):
        build_features(df, "does_not_exist")


def test_recipe_tables_are_consistent():
    """Every recipe in RECIPE_BUILDERS must have a symmetry flag."""
    assert set(RECIPE_BUILDERS) == set(RECIPE_IS_SYMMETRIC)
    assert set(SYMMETRIC_RECIPES) | set(ASYMMETRIC_RECIPES) == set(RECIPE_BUILDERS)
    assert set(SYMMETRIC_RECIPES).isdisjoint(set(ASYMMETRIC_RECIPES))


def test_ref_alt_swap_negates_llr_and_swaps_pools():
    df = _make_cache_df(n=3, d=4)
    swapped = ref_alt_swap(df)
    np.testing.assert_allclose(swapped["llr"].to_numpy(), -df["llr"].to_numpy())
    np.testing.assert_allclose(swapped["minus_llr"].to_numpy(), df["llr"].to_numpy())
    for i in range(len(df)):
        np.testing.assert_allclose(swapped["mean_ref"].iloc[i], df["mean_alt"].iloc[i])
        np.testing.assert_allclose(swapped["mean_alt"].iloc[i], df["mean_ref"].iloc[i])
    # abs_llr, embed_last_l2, traitgym_innerprod don't depend on direction.
    np.testing.assert_allclose(swapped["abs_llr"].to_numpy(), df["abs_llr"].to_numpy())
    np.testing.assert_allclose(
        swapped["embed_last_l2"].to_numpy(), df["embed_last_l2"].to_numpy()
    )
