"""Feature builders that turn the supervised-VEP feature cache into sklearn-ready arrays.

Each recipe consumes the cache columns produced by
``bolinas.supervised.inference.compute_pooled_features`` (a pandas DataFrame
with scalar columns ``llr / abs_llr / minus_llr / embed_last_l2`` and three
D-dim list columns ``mean_ref / mean_alt / traitgym_innerprod``) and returns a
dense ``np.ndarray`` of shape ``[N, F]`` ready to feed into sklearn.

**Symmetric vs asymmetric.** ``complex_traits`` and ``eqtl`` have no
biological ref/alt direction; features used on them must be invariant under
ref↔alt swap. The ``RECIPE_IS_SYMMETRIC`` table flags each recipe; the
``ref_alt_swap`` helper here is also used in the unit tests to verify
the symmetric recipes literally satisfy ``recipe(swap(df)) == recipe(df)``.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd


def _stack_list_column(df: pd.DataFrame, col: str) -> np.ndarray:
    """Convert a parquet list column into an ``[N, D]`` numpy array."""
    arr = np.stack(df[col].to_numpy())
    assert arr.ndim == 2, f"column {col!r} did not stack to 2-D (got {arr.shape})"
    return arr.astype(np.float32, copy=False)


# ---------- symmetric recipes (apply to all 3 datasets) -------------------


def build_abs_delta(df: pd.DataFrame) -> np.ndarray:
    """``|mean_alt - mean_ref|``, D-dim, symmetric."""
    return np.abs(
        _stack_list_column(df, "mean_alt") - _stack_list_column(df, "mean_ref")
    )


def build_prod(df: pd.DataFrame) -> np.ndarray:
    """``mean_ref ∘ mean_alt`` (element-wise), D-dim, symmetric."""
    return _stack_list_column(df, "mean_ref") * _stack_list_column(df, "mean_alt")


def build_sym_concat(df: pd.DataFrame) -> np.ndarray:
    """``concat(mean_ref + mean_alt, |mean_alt - mean_ref|)`` — 2 D-dim, symmetric.

    Captures both "where the variant lives" (sum) and "what changed" (abs delta).
    """
    mref = _stack_list_column(df, "mean_ref")
    malt = _stack_list_column(df, "mean_alt")
    return np.concatenate([mref + malt, np.abs(malt - mref)], axis=1)


def build_traitgym_innerprod(df: pd.DataFrame) -> np.ndarray:
    """TraitGym recipe: ``(emb_ref ∘ emb_alt).sum(seq_axis)`` per channel.

    Symmetric in (ref, alt) by commutativity of the element-wise product.
    """
    return _stack_list_column(df, "traitgym_innerprod")


def build_abs_delta_with_scalars(df: pd.DataFrame) -> np.ndarray:
    """``abs_delta`` + symmetric zero-shot scalars (``abs_llr``, ``embed_last_l2``).

    Mendelian convention also has ``minus_llr`` (asymmetric) as its zero-shot
    headline; that one is excluded here so the recipe stays symmetric.
    """
    return np.column_stack(
        [
            build_abs_delta(df),
            np.abs(df["llr"].to_numpy(dtype=np.float32)),
            df["embed_last_l2"].to_numpy(dtype=np.float32),
        ]
    )


# ---------- asymmetric probe recipes (mendelian-only) ---------------------


def build_mean_delta(df: pd.DataFrame) -> np.ndarray:
    """``concat(mean_ref, mean_alt, mean_alt - mean_ref)`` — 3 D-dim, asymmetric."""
    mref = _stack_list_column(df, "mean_ref")
    malt = _stack_list_column(df, "mean_alt")
    return np.concatenate([mref, malt, malt - mref], axis=1)


def build_mean_concat_with_llr(df: pd.DataFrame) -> np.ndarray:
    """``concat(mean_ref, mean_alt, signed_llr)`` — 2 D + 1, asymmetric."""
    mref = _stack_list_column(df, "mean_ref")
    malt = _stack_list_column(df, "mean_alt")
    llr = df["llr"].to_numpy(dtype=np.float32).reshape(-1, 1)
    return np.concatenate([mref, malt, llr], axis=1)


# ---------- registry ------------------------------------------------------


RECIPE_BUILDERS: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    # symmetric — usable on all 3 datasets
    "abs_delta": build_abs_delta,
    "prod": build_prod,
    "sym_concat": build_sym_concat,
    "traitgym_innerprod": build_traitgym_innerprod,
    "abs_delta_plus_scalars": build_abs_delta_with_scalars,
    # asymmetric — mendelian-only probe
    "mean_delta": build_mean_delta,
    "mean_concat_plus_llr": build_mean_concat_with_llr,
}


RECIPE_IS_SYMMETRIC: dict[str, bool] = {
    "abs_delta": True,
    "prod": True,
    "sym_concat": True,
    "traitgym_innerprod": True,
    "abs_delta_plus_scalars": True,
    "mean_delta": False,
    "mean_concat_plus_llr": False,
}


SYMMETRIC_RECIPES: tuple[str, ...] = tuple(
    name for name, sym in RECIPE_IS_SYMMETRIC.items() if sym
)
ASYMMETRIC_RECIPES: tuple[str, ...] = tuple(
    name for name, sym in RECIPE_IS_SYMMETRIC.items() if not sym
)


assert set(RECIPE_BUILDERS) == set(RECIPE_IS_SYMMETRIC), (
    "RECIPE_BUILDERS and RECIPE_IS_SYMMETRIC must cover the same recipe names"
)


def build_features(df: pd.DataFrame, recipe: str) -> np.ndarray:
    """Look up and apply the named recipe."""
    if recipe not in RECIPE_BUILDERS:
        raise KeyError(
            f"unknown recipe {recipe!r}; available: {sorted(RECIPE_BUILDERS)}"
        )
    return RECIPE_BUILDERS[recipe](df)


def ref_alt_swap(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ref↔alt exchanged in every relevant column.

    Used for symmetry tests and (later, optionally) for ref/alt-augmented
    training on complex_traits / eqtl. Swaps:

    * scalar ``ref`` ↔ ``alt`` columns
    * dense ``mean_ref`` ↔ ``mean_alt`` columns
    * ``llr`` → ``-llr`` (and ``minus_llr`` accordingly)

    The TraitGym ``innerprod`` column is already symmetric in ref↔alt so
    it doesn't need swapping. ``embed_last_l2``, ``abs_llr`` likewise.
    """
    out = df.copy()
    if "ref" in out.columns and "alt" in out.columns:
        out["ref"], out["alt"] = df["alt"], df["ref"]
    if "mean_ref" in out.columns and "mean_alt" in out.columns:
        out["mean_ref"], out["mean_alt"] = df["mean_alt"], df["mean_ref"]
    if "llr" in out.columns:
        out["llr"] = -df["llr"]
    if "minus_llr" in out.columns:
        out["minus_llr"] = -df["minus_llr"]
    return out
