"""Feature-based sample matching utilities.

Ported from TraitGym (commit e59d612e9; src/traitgym/matching.py).
"""

import warnings

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler

from bolinas.pipelines.evals.variants import COORDINATES

MATCH_GROUP_COL = "match_group"

# Sentinels for bin labels.
BIN_OOR = "OOR"  # value outside any bin (or null) — emitted by `bin_feature`
BIN_NA = "NA"  # subset-conditional bin not applicable for this row

# Bin schemes locked in issue #156 — iter 33 final design covering all three
# eval datasets (mendelian / complex / eqtl). https://github.com/Open-Athena/bolinas-dna/issues/156
TSS_DIST_BIN_EDGES = [0, 50, 100, 200, 500, 1000]
EXON_DIST_BIN_EDGES = [0, 5, 20, 30]
# Wider edges for `non_coding_transcript_exon_variant` × `distance_tss_nc`,
# added in round-2 of #156 after the Catalogue source switch exposed a
# residual leak (PA=0.401, p=1.2e-5). ncRNA-exon variants span a much
# broader distance-to-TSS range than tss_proximal (the iter-33 design's
# only consumer of `distance_tss_nc_bin`) — empirical pos quantiles run
# from 9 bp (q=0.05) to ~8.7 kb (q=0.95). 4 bins (+ OOR for >5 kb).
NCRNA_TSS_NC_DIST_BIN_EDGES = [0, 200, 1000, 5000]
# Three MAF bin granularities used by the per-subset tiered scheme.
MAF_BIN_EDGES_20 = [
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
MAF_BIN_EDGES_10 = [0.0, 0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5]
MAF_BIN_EDGES_5 = [0.0, 0.001, 0.01, 0.05, 0.2, 0.5]
# Back-compat alias: existing tests / scratch scripts still import MAF_BIN_EDGES.
MAF_BIN_EDGES = MAF_BIN_EDGES_20

# Per-subset MAF bin schemes (iter 33 production).
#
# `LOG_LOCAL` is a sentinel that means "use a local equal-width log10(MAF) bin
# computed per categorical match group (joint pos+neg ref) with `LOG_LOCAL_N`
# buckets". Used for eqtl/distal where fixed global edges left a small but
# Bonferroni-significant residual MAF leak (PA ≈ 0.532) and the per-group
# adaptation closes it.
LOG_LOCAL = "log_local"
LOG_LOCAL_N = 8

MAF_TIERED_V1: dict[str, list[float] | str] = {
    # Big subsets with strong leak in the no-bin baseline → fine 20bin.
    "distal": MAF_BIN_EDGES_20,
    "tss_proximal": MAF_BIN_EDGES_20,
    "non_coding_transcript_exon_variant": MAF_BIN_EDGES_20,
    # Medium subsets → 10bin.
    "3_prime_UTR_variant": MAF_BIN_EDGES_10,
    "5_prime_UTR_variant": MAF_BIN_EDGES_10,
    "missense_variant": MAF_BIN_EDGES_10,
    # Small subsets → 5bin to preserve pair count.
    "synonymous_variant": MAF_BIN_EDGES_5,
    "splicing": MAF_BIN_EDGES_5,
    "mature_miRNA_variant": MAF_BIN_EDGES_5,
    "stop_retained_variant": MAF_BIN_EDGES_5,
    "coding_sequence_variant": MAF_BIN_EDGES_5,
}
MAF_TIERED_LOG8_DISTAL_ONLY: dict[str, list[float] | str] = {
    **MAF_TIERED_V1,
    "distal": LOG_LOCAL,
}


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
    expr = pl.lit(BIN_OOR)
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


# Iter-33 categorical match-key shared by all three eval datasets. The
# trailing per-subset / per-dataset bin columns (`distance_*_bin`, `MAF_bin`)
# are appended by each *_dataset rule; this is the chrom + consequence_final +
# closest-gene-id block they all share.
CAT_BASE: list[str] = [
    "chrom",
    "consequence_final",
    "tss_closest_pc_gene_id",
    "tss_closest_nc_gene_id",
    "exon_closest_pc_gene_id",
    "exon_closest_nc_gene_id",
]


def add_subset_distance_bins(
    df: pl.DataFrame,
    *,
    include_ncrna_tss_nc_bin: bool = False,
) -> pl.DataFrame:
    """Iter-33 per-biotype distance bins as exact-match categoricals.

    - ``distance_tss_pc_bin`` and ``distance_tss_nc_bin``: meaningful only for
      ``tss_proximal`` (else ``BIN_NA``); edges = ``TSS_DIST_BIN_EDGES``.
    - ``distance_exon_pc_bin``: meaningful only for ``splicing`` (else
      ``BIN_NA``); edges = ``EXON_DIST_BIN_EDGES``.

    ``distance_exon_nc_bin`` is intentionally not added — splicing/exon_nc
    was clean in the iter-33 baseline so there's no leak to fix and the
    extra bin column would over-tighten splicing matching.

    Args:
        df: dataframe with ``consequence_group`` + distance columns.
        include_ncrna_tss_nc_bin: round-2 opt-in for eqtl. When True,
            also applies ``distance_tss_nc_bin`` to the
            ``non_coding_transcript_exon_variant`` subset using
            ``NCRNA_TSS_NC_DIST_BIN_EDGES`` (wider than ``TSS_DIST_BIN_EDGES``
            because ncRNA-exon variants span a much broader
            distance-to-nc-TSS range than tss_proximal). Off by default to
            keep mendelian/complex iter-33 outputs byte-equivalent.
    """
    tss_nc_bin_expr = pl.when(pl.col("consequence_group") == "tss_proximal").then(
        bin_feature("distance_tss_nc", TSS_DIST_BIN_EDGES)
    )
    if include_ncrna_tss_nc_bin:
        tss_nc_bin_expr = tss_nc_bin_expr.when(
            pl.col("consequence_group") == "non_coding_transcript_exon_variant"
        ).then(bin_feature("distance_tss_nc", NCRNA_TSS_NC_DIST_BIN_EDGES))
    tss_nc_bin_expr = tss_nc_bin_expr.otherwise(pl.lit(BIN_NA)).alias(
        "distance_tss_nc_bin"
    )

    return df.with_columns(
        pl.when(pl.col("consequence_group") == "tss_proximal")
        .then(bin_feature("distance_tss_pc", TSS_DIST_BIN_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_tss_pc_bin"),
        tss_nc_bin_expr,
        pl.when(pl.col("consequence_group") == "splicing")
        .then(bin_feature("distance_exon_pc", EXON_DIST_BIN_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_exon_pc_bin"),
    )


def add_tiered_maf_bin(
    df: pl.DataFrame,
    scheme: dict[str, list[float] | str],
    *,
    log_local_group_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Add a ``MAF_bin`` column whose edges depend on each row's
    ``consequence_group``.

    ``scheme[consequence_group]`` is either:
      - ``list[float]`` — fixed right-closed bin edges (e.g. ``MAF_BIN_EDGES_20``).
      - ``LOG_LOCAL`` sentinel — local equal-width log10(MAF) bins, ``LOG_LOCAL_N``
        buckets, joint pos+neg reference within each group defined by
        ``log_local_group_cols`` (typically the categorical match features).

    Bin labels are prefixed with the subset name (``"missense:b3"``,
    ``"distal:OOR"``, ``"ll:5"`` for log_local) so labels never collide across
    subsets even though ``consequence_final`` already separates them at
    ``match_features``' partitioning step. Rows whose consequence_group is not
    in ``scheme`` get ``"UNKNOWN"`` (and won't match anything since no other
    row will share that label within their consequence_final).

    Args:
        df: dataframe with at least ``MAF`` and ``consequence_group`` columns.
        scheme: per-subset bin choice.
        log_local_group_cols: required iff any scheme value is ``LOG_LOCAL``;
            usually the 6-element ``CAT_BASE`` list (chrom, consequence_final,
            and the four ``*_closest_*_gene_id`` columns).

    Returns:
        New dataframe with a ``MAF_bin`` column appended.
    """
    # Defensive: subset names get truncated to 8 chars in the bin label below
    # to keep the column compact. If a future scheme adds two consequence
    # groups whose first 8 chars collide (e.g. `non_coding_transcript_exon_*`
    # variants), labels would silently merge across subsets — fail loudly
    # instead. consequence_final is also in the categorical match key so
    # cross-subset matching is impossible regardless, but the label-confusion
    # would surface as a confusing diagnostic rather than wrong matches.
    fixed_edge_subsets = [s for s, v in scheme.items() if v != LOG_LOCAL]
    prefixes = {s[:8] for s in fixed_edge_subsets}
    assert len(prefixes) == len(fixed_edge_subsets), (
        f"first-8-char collision among scheme keys: {sorted(fixed_edge_subsets)}"
    )

    needs_log = any(v == LOG_LOCAL for v in scheme.values())
    if needs_log:
        if log_local_group_cols is None:
            raise ValueError("log_local_group_cols required when scheme uses LOG_LOCAL")
        log_maf = pl.col("MAF").clip(1e-10, 1.0).log10()
        df = df.with_columns(
            log_maf.min().over(log_local_group_cols).alias("_lo"),
            log_maf.max().over(log_local_group_cols).alias("_hi"),
        )
        width = (pl.col("_hi") - pl.col("_lo")) / LOG_LOCAL_N
        log_local_idx = (
            ((log_maf - pl.col("_lo")) / width)
            .floor()
            .cast(pl.Int64, strict=False)
            .fill_null(0)  # width=0 (constant-MAF group) → NaN → null → b0
            .clip(0, LOG_LOCAL_N - 1)
        )
        # Emit `ll:OOR` for null / NaN MAF rows so they only match each other
        # within their categorical group — matches `bin_feature`'s null
        # handling (also OOR) so the two scheme types behave consistently.
        # Without this, the strict_cast=False above produces a null index
        # which then formats as `ll:null`, silently distinct from the OOR
        # label other subsets emit for the same input.
        log_local_label = (
            pl.when(pl.col("MAF").is_null() | pl.col("MAF").is_nan())
            .then(pl.lit("ll:OOR"))
            .otherwise(pl.format("ll:{}", log_local_idx))
        )

    expr = pl.lit("UNKNOWN")
    for subset, val in scheme.items():
        if val == LOG_LOCAL:
            bin_expr = log_local_label
        else:
            bin_expr = pl.format(
                "{}:{}",
                pl.lit(subset[:8]),
                bin_feature("MAF", val, right_closed=True),
            )
        expr = (
            pl.when(pl.col("consequence_group") == subset)
            .then(bin_expr)
            .otherwise(expr)
        )

    df = df.with_columns(expr.alias("MAF_bin"))
    if needs_log:
        df = df.drop(["_lo", "_hi"])
    return df


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

    # Pre-filter: drop neg rows whose categorical combo doesn't match any
    # positive's. Behaviorally identical to the current loop (those negs would
    # never enter `neg_groups[group_key]` for any positive's group_key), but
    # cuts the next two steps' cost by ~50-100× on real eval datasets where
    # most negatives sit far from any positive in (gene_id × MAF_bin × …)
    # space. complex_traits goes from ~12M neg rows to ~100k after this join,
    # which makes the to_pandas/partition_by stack run in seconds instead of
    # minutes (and on a 16 GB box instead of needing 256 GB).
    # `partition_by` (used below) and the semi-join both require at least one
    # categorical key. Pre-iter-33 the function had the same limitation
    # (partition_by raises on an empty key list); this assertion just makes
    # the failure mode obvious instead of buried polars-internal.
    assert categorical_features, "categorical_features must be non-empty"
    pos_keys = pos.select(categorical_features).unique()
    neg = neg.join(pos_keys, on=categorical_features, how="semi")

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
