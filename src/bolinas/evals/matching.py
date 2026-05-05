"""Feature-based sample matching utilities.

Ported from TraitGym (commit e59d612e9; src/traitgym/matching.py).
"""

import warnings

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler

from bolinas.evals.variants import COORDINATES

MATCH_GROUP_COL = "match_group"

# Bin schemes locked in issue #156 (iter 22 mendelian, iter 24 complex).
# https://github.com/Open-Athena/bolinas-dna/issues/156
TSS_DIST_BIN_EDGES = [0, 50, 100, 200, 500, 1000]
EXON_DIST_BIN_EDGES = [0, 5, 20, 30]
MAF_BIN_EDGES = [
    0.0,
    0.0005,
    0.001,
    0.0015,
    0.002,
    0.0025,
    0.003,
    0.0035,
    0.004,
    0.005,
    0.007,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.07,
    0.1,
    0.15,
    0.2,
    0.5,
]


def bin_feature(
    feature: str,
    edges: list[float],
    *,
    right_closed: bool = False,
) -> pl.Expr:
    """Polars expression binning ``feature`` into ``len(edges) - 1`` string buckets.

    Returns ``"b0".."b{n-1}"`` for in-range values, ``"OOR"`` for out-of-range
    or null values.

    With ``right_closed=False`` (default; used for ``tss_dist`` / ``exon_dist``):
        bins are ``[lo, hi)`` for ``i < n-1``, ``[lo, hi]`` for the last bin.
    With ``right_closed=True`` (used for ``MAF``):
        bins are ``(lo, hi]`` for ``i > 0``, ``[lo, hi]`` for the first bin.
    """
    assert len(edges) >= 2, f"need at least 2 edges, got {edges!r}"
    n = len(edges) - 1
    expr = pl.lit("OOR")
    col = pl.col(feature)
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        if right_closed:
            cond = ((col >= lo) & (col <= hi)) if i == 0 else ((col > lo) & (col <= hi))
        else:
            cond = (
                ((col >= lo) & (col <= hi))
                if i == n - 1
                else ((col >= lo) & (col < hi))
            )
        expr = pl.when(cond).then(pl.lit(f"b{i}")).otherwise(expr)
    return expr


def match_features(
    pos: pl.DataFrame,
    neg: pl.DataFrame,
    continuous_features: list[str],
    categorical_features: list[str],
    k: int,
    scale: bool = True,
    seed: int | None = 42,
) -> pl.DataFrame:
    """Match positive samples to k negative samples based on features.

    For each unique combination of categorical features, finds the k closest
    negative samples for each positive sample based on continuous features.

    Args:
        pos: Positive samples DataFrame with COORDINATES columns.
        neg: Negative samples DataFrame with COORDINATES columns.
        continuous_features: Columns to use for distance-based matching.
        categorical_features: Columns for exact matching (grouping).
        k: Number of negative samples to match per positive sample.
        scale: Whether to scale continuous features before matching.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with matched samples, match_group column, sorted by COORDINATES.

    Raises:
        ValueError: If required columns are missing from pos or neg, or if
            null values are present in match features.
    """
    required_cols = COORDINATES + continuous_features + categorical_features
    _validate_columns(pos, required_cols, "pos")
    _validate_columns(neg, required_cols, "neg")
    all_features = continuous_features + categorical_features
    _validate_no_nulls(pos, all_features, "pos")
    _validate_no_nulls(neg, all_features, "neg")

    pos_pd = pos.to_pandas()
    neg_pd = neg.to_pandas()

    if scale and len(continuous_features) > 0:
        pos_pd, neg_pd, match_cols = _scale_features(
            pos_pd, neg_pd, continuous_features
        )
    else:
        match_cols = continuous_features

    # Use polars partition_by for fast group splitting. Pandas multi-index
    # .loc on millions of negatives is the per-call hot spot otherwise.
    pos_groups = pl.from_pandas(pos_pd).partition_by(categorical_features, as_dict=True)
    neg_groups = pl.from_pandas(neg_pd).partition_by(categorical_features, as_dict=True)

    pos_list: list[pd.DataFrame] = []
    neg_list: list[pd.DataFrame] = []

    for group_key, pos_group_pl in pos_groups.items():
        # partition_by returns single-element tuples even for one-column keys;
        # strip when match_single_group expects the unwrapped scalar in messages.
        display_key = group_key[0] if len(group_key) == 1 else group_key
        neg_group_pl = neg_groups.get(group_key)
        if neg_group_pl is None:
            warnings.warn(f"No negatives found for category: {display_key}")
            continue
        result = _match_single_group(
            pos_group_pl.to_pandas(),
            neg_group_pl.to_pandas(),
            display_key,
            match_cols,
            k,
            seed,
        )
        if result is not None:
            pos_group, neg_group = result
            pos_list.append(pos_group)
            neg_list.append(neg_group)

    result_df = _combine_results(pos_list, neg_list, k)

    if scale and len(continuous_features) > 0:
        scaled_cols = [f"{c}_scaled" for c in continuous_features]
        result_df = result_df.drop(columns=scaled_cols, errors="ignore")

    return _sort_by_coordinates(pl.from_pandas(result_df))


def _validate_columns(
    df: pl.DataFrame,
    required: list[str],
    name: str,
) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _validate_no_nulls(
    df: pl.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    for col in columns:
        null_count = df[col].null_count()
        if null_count > 0:
            raise ValueError(f"{name} has {null_count} null values in column '{col}'")


def _scale_features(
    pos: pd.DataFrame,
    neg: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    scaler = RobustScaler()
    all_data = pd.concat([pos[features], neg[features]])
    scaler.fit(all_data)

    scaled_cols = [f"{c}_scaled" for c in features]
    pos = pos.copy()
    neg = neg.copy()
    pos[scaled_cols] = scaler.transform(pos[features])
    neg[scaled_cols] = scaler.transform(neg[features])

    return pos, neg, scaled_cols


def _match_single_group(
    pos_group: pd.DataFrame,
    neg_group: pd.DataFrame,
    group_key: tuple | str,
    match_cols: list[str],
    k: int,
    seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    n_pos_needed = len(pos_group)
    n_neg_available = len(neg_group)
    n_neg_needed = n_pos_needed * k

    if n_neg_needed > n_neg_available:
        warnings.warn(
            f"Insufficient negatives for category {group_key}: "
            f"need {n_neg_needed}, have {n_neg_available}. Subsampling positives."
        )
        n_pos_possible = n_neg_available // k
        pos_group = pos_group.sample(n_pos_possible, random_state=seed)

    if len(match_cols) == 0:
        n_neg_to_sample = len(pos_group) * k
        neg_matched = neg_group.sample(n_neg_to_sample, random_state=seed)
    else:
        neg_matched = _find_closest(pos_group, neg_group, match_cols, k)

    return pos_group, neg_matched


def _find_closest(
    pos: pd.DataFrame,
    neg: pd.DataFrame,
    cols: list[str],
    k: int,
) -> pd.DataFrame:
    """Find k closest negatives for each positive (Euclidean), without replacement."""
    dist_matrix = cdist(pos[cols], neg[cols])
    closest_indices: list[int] = []

    for i in range(len(pos)):
        sorted_indices = np.argsort(dist_matrix[i])[:k].tolist()
        closest_indices.extend(sorted_indices)
        dist_matrix[:, sorted_indices] = np.inf

    return neg.iloc[closest_indices]


def _combine_results(
    pos_list: list[pd.DataFrame],
    neg_list: list[pd.DataFrame],
    k: int,
) -> pd.DataFrame:
    if not pos_list:
        return pd.DataFrame()

    all_pos = pd.concat(pos_list, ignore_index=True)
    all_pos[MATCH_GROUP_COL] = np.arange(len(all_pos))

    all_neg = pd.concat(neg_list, ignore_index=True)
    all_neg[MATCH_GROUP_COL] = np.repeat(all_pos[MATCH_GROUP_COL].values, k)

    return pd.concat([all_pos, all_neg], ignore_index=True)


def _sort_by_coordinates(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(COORDINATES)
