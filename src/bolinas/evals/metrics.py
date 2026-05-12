"""Metric utilities for variant-effect evaluations.

Two metric families live here:

- ``pairwise_accuracy`` / ``compute_pairwise_metrics``: matched-pair within-
  ``match_group`` accuracy (ties = 0.5) with Wald-binomial SE. Used by the
  matched-pair eval datasets in ``snakemake/evals/`` and the
  ``conservation_eval`` pipeline.
- ``METRIC_FUNCTIONS`` / ``compute_metrics``: classical AUPRC / AUROC /
  Spearman over (label, score) pairs. Still used by older pipelines
  (``snakemake/analysis/evals_v1/``, ``scripts/evo2_eval/``).
"""

import math
from typing import Callable

import pandas as pd
from scipy.stats import binom, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


METRIC_FUNCTIONS: dict[str, Callable[[pd.Series, pd.Series], float]] = {
    "AUPRC": lambda label, score: average_precision_score(label, score),
    "AUROC": lambda label, score: roc_auc_score(label, score),
    "Spearman": lambda label, score: spearmanr(label, score)[0],
}


def pairwise_accuracy(
    label: pd.Series,
    score: pd.Series,
    match_group: pd.Series,
    alternative: str = "two-sided",
) -> dict[str, float | int]:
    """Within-``match_group`` accuracy: fraction of pairs where the positive
    scores higher than the negative (ties = 0.5).

    Asserts each ``match_group`` has exactly one positive and one negative.

    Args:
        label: 0/1 (or bool) per row. Cast to int internally.
        score: numeric score per row. **Must not contain NaN** — fill
            upstream with a semantically appropriate value (the
            ``conservation_eval`` pipeline does ``.fillna(0)``). Without
            this rule, a NaN-vs-NaN pair would silently count as a loss
            for the positive (since ``NaN > NaN`` and ``NaN == NaN`` are
            both False) — that's exactly the kind of silent-corruption
            risk we want to fail loud on.
        match_group: integer group id; positives and negatives are paired
            within a group.
        alternative: ``"two-sided"`` (default) tests the null ``acc = 0.5``
            against any deviation; ``"greater"`` tests against ``acc > 0.5``
            (use when the score's sign is fixed by assumption so "higher =
            class 1" is the only direction of interest — gives 2x the power
            of a two-sided test); ``"less"`` is the mirror image.

    Returns:
        ``{"value", "se", "n_pairs", "n_ties", "p_value"}``. ``se`` is the Wald
        binomial form ``sqrt(value * (1 - value) / n_pairs)``. ``p_value`` is
        the closed-form sign-test p-value under the null
        ``P(pos > neg | not tied) = 0.5`` — direction set by ``alternative``.
        Returns 1.0 when ``n_pairs == n_ties`` (all ties → no information).
    """
    assert len(label) == len(score) == len(match_group), (
        f"length mismatch: label={len(label)} score={len(score)} "
        f"match_group={len(match_group)}"
    )

    label_int = pd.Series(label).astype(int).reset_index(drop=True)
    score_arr = pd.Series(score).reset_index(drop=True)
    mg = pd.Series(match_group).reset_index(drop=True)
    assert not score_arr.isna().any(), (
        f"score has {int(score_arr.isna().sum())} NaN values; fill upstream "
        f"with a semantically appropriate default before scoring"
    )

    df = pd.DataFrame({"label": label_int, "score": score_arr, "match_group": mg})

    # Each group must have exactly 1 pos + 1 neg.
    counts = df.groupby("match_group")["label"].agg(["sum", "count"])
    bad = counts[(counts["sum"] != 1) | (counts["count"] != 2)]
    assert bad.empty, (
        f"pairwise_accuracy expects exactly 1 positive + 1 negative per "
        f"match_group, got {len(bad)} bad groups; first: {bad.head().to_dict()}"
    )

    pos = df[df["label"] == 1].set_index("match_group")["score"].sort_index()
    neg = df[df["label"] == 0].set_index("match_group")["score"].sort_index()
    assert pos.index.equals(neg.index), "positive/negative match_group sets differ"

    diff = pos.values - neg.values
    n = len(diff)
    wins = int((diff > 0).sum())
    ties = int((diff == 0).sum())
    value = (wins + 0.5 * ties) / n
    se = math.sqrt(value * (1 - value) / n)

    # Sign-test p-value: ties drop out (each side gains 0.5 in expectation
    # under H0 so they don't shift the test statistic), and ``wins`` over
    # ``n_eff = n - ties`` non-tied pairs is Binom(n_eff, 0.5).
    n_eff = n - ties
    if n_eff == 0:
        p_value = 1.0
    else:
        # P(X >= wins) under Binom(n_eff, 0.5) is the one-sided p for H1: acc > 0.5.
        # binom.sf(k, n, p) = P(X > k) = P(X >= k+1), so use wins - 1.
        p_greater = float(binom.sf(wins - 1, n_eff, 0.5))
        p_less = float(binom.cdf(wins, n_eff, 0.5))
        if alternative == "greater":
            p_value = p_greater
        elif alternative == "less":
            p_value = p_less
        elif alternative == "two-sided":
            extreme = max(wins, n_eff - wins)
            p_one_tail = float(binom.sf(extreme - 1, n_eff, 0.5))
            p_value = min(2.0 * p_one_tail, 1.0)
        else:
            raise ValueError(
                f"alternative must be 'two-sided', 'greater', or 'less'; got {alternative!r}"
            )

    return {
        "value": float(value),
        "se": float(se),
        "n_pairs": int(n),
        "n_ties": ties,
        "p_value": float(p_value),
    }


def paired_score_comparison(
    label: pd.Series,
    score_a: pd.Series,
    score_b: pd.Series,
    match_group: pd.Series,
    alternative: str = "two-sided",
) -> dict[str, float | int]:
    """Paired McNemar-style sign test: is score A better than score B on the
    *same* matched-pair set?

    For each ``match_group`` (pos, neg) pair, both scores produce an ordering
    outcome ∈ {1.0 (pos > neg), 0.5 (tie), 0.0 (pos < neg)} — same convention
    as :func:`pairwise_accuracy`. Per-pair advantage = ``out_A - out_B`` ∈
    {-1, -0.5, 0, +0.5, +1}. Discordant pairs are those with nonzero advantage.

    Under ``H0: A == B``, advantages are symmetric around 0, so the count of
    "A-wins-this-pair" (positive advantage) is ``Binom(n_discordant, 0.5)``.
    Half-advantage pairs (`±0.5`) are counted with their natural sign — a
    ±0.5 contributes 0.5 to wins / losses (mirrors the tie convention in the
    parent test). For the binomial test, we use the rounded `wins` count over
    the integer-rounded discordant count; this is the standard handling.

    Args:
        label, score_a, score_b, match_group: parallel arrays of equal length.
            Same constraints as :func:`pairwise_accuracy` (one pos + one neg
            per group, no NaNs in either score).
        alternative: ``'two-sided'`` (default), ``'greater'`` (H1: A > B), or
            ``'less'`` (H1: A < B).

    Returns:
        ``{value, se, n_pairs, n_a_wins, n_b_wins, n_concordant, n_half, p_value}``.
        ``value = n_a_wins / (n_a_wins + n_b_wins)`` (with halves counted as
        0.5 to each side) — the fraction of discordant pairs where A wins.
        ``n_pairs`` is total match_groups; ``n_concordant`` is pairs where A
        and B agree (both right or both wrong); ``n_half`` is pairs where one
        side has a tie (advantage = ±0.5).
    """
    assert len(label) == len(score_a) == len(score_b) == len(match_group), (
        f"length mismatch: label={len(label)} a={len(score_a)} "
        f"b={len(score_b)} match_group={len(match_group)}"
    )
    sa = pd.Series(score_a).reset_index(drop=True)
    sb = pd.Series(score_b).reset_index(drop=True)
    assert not sa.isna().any() and not sb.isna().any(), "scores must not contain NaN"

    df = pd.DataFrame({
        "label": pd.Series(label).astype(int).reset_index(drop=True),
        "a": sa,
        "b": sb,
        "match_group": pd.Series(match_group).reset_index(drop=True),
    })
    counts = df.groupby("match_group")["label"].agg(["sum", "count"])
    bad = counts[(counts["sum"] != 1) | (counts["count"] != 2)]
    assert bad.empty, (
        f"paired_score_comparison expects exactly 1 pos + 1 neg per match_group; "
        f"got {len(bad)} bad groups"
    )

    pos = df[df["label"] == 1].set_index("match_group")[["a", "b"]].sort_index()
    neg = df[df["label"] == 0].set_index("match_group")[["a", "b"]].sort_index()
    assert pos.index.equals(neg.index), "positive/negative match_group sets differ"

    # outcome ∈ {0, 0.5, 1} per (match_group, score).
    diff_a = pos["a"].values - neg["a"].values
    diff_b = pos["b"].values - neg["b"].values
    import numpy as _np
    out_a = _np.where(diff_a > 0, 1.0, _np.where(diff_a < 0, 0.0, 0.5))
    out_b = _np.where(diff_b > 0, 1.0, _np.where(diff_b < 0, 0.0, 0.5))
    advantage = out_a - out_b  # ∈ {-1, -0.5, 0, +0.5, +1}

    n_pairs = len(advantage)
    full_wins_a = int((advantage == 1.0).sum())
    full_wins_b = int((advantage == -1.0).sum())
    half_wins_a = int((advantage == 0.5).sum())
    half_wins_b = int((advantage == -0.5).sum())
    n_concordant = int((advantage == 0).sum())
    n_half = half_wins_a + half_wins_b
    n_discordant_full = full_wins_a + full_wins_b
    # Soft "wins" weighting halves at 0.5 each.
    soft_wins_a = full_wins_a + 0.5 * half_wins_a
    soft_wins_b = full_wins_b + 0.5 * half_wins_b
    soft_n = soft_wins_a + soft_wins_b
    if soft_n == 0:
        value = 0.5
        se = 0.0
    else:
        value = float(soft_wins_a / soft_n)
        se = math.sqrt(value * (1 - value) / soft_n)

    # Binomial p-value: use the integer-rounded full-advantage counts (ignore
    # halves for the formal test — McNemar standard). If there are no full
    # discordant pairs, fall back to the half counts as integer wins.
    n_for_test = n_discordant_full if n_discordant_full > 0 else (half_wins_a + half_wins_b)
    wins_for_test = full_wins_a if n_discordant_full > 0 else half_wins_a
    if n_for_test == 0:
        p_value = 1.0
    else:
        if alternative == "greater":
            p_value = float(binom.sf(wins_for_test - 1, n_for_test, 0.5))
        elif alternative == "less":
            p_value = float(binom.cdf(wins_for_test, n_for_test, 0.5))
        elif alternative == "two-sided":
            extreme = max(wins_for_test, n_for_test - wins_for_test)
            p_value = float(min(2.0 * binom.sf(extreme - 1, n_for_test, 0.5), 1.0))
        else:
            raise ValueError(
                f"alternative must be 'two-sided', 'greater', or 'less'; got {alternative!r}"
            )

    return {
        "value": float(value),
        "se": float(se),
        "n_pairs": int(n_pairs),
        "n_a_wins": full_wins_a,
        "n_b_wins": full_wins_b,
        "n_concordant": n_concordant,
        "n_half": n_half,
        "p_value": float(p_value),
    }


def compute_pairwise_metrics(
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    score_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute PairwiseAccuracy + SE per ``subset`` for one or more score columns.

    Aligns ``dataset`` with ``scores`` by row index (assumes same order).
    Stratifies by the ``subset`` column — one row per (subset, score_column).
    Asserts no ``match_group`` straddles subsets.

    Args:
        dataset: DataFrame with columns ``[label, subset, match_group]`` (and
            optionally other passthrough columns).
        scores: DataFrame whose columns are the model scores; row-aligned with
            ``dataset``.
        score_columns: Score column names to evaluate. Defaults to all columns
            of ``scores``.

    Returns:
        DataFrame with columns ``[score_type, subset, value, se, n_pairs,
        n_ties, p_value]``. ``score_type`` is the column name from ``scores``.
    """
    for col in ("label", "subset", "match_group"):
        assert col in dataset.columns, f"dataset missing required column {col!r}"

    if score_columns is None:
        score_columns = list(scores.columns)

    merged = pd.concat(
        [dataset.reset_index(drop=True), scores.reset_index(drop=True)], axis=1
    )

    # No match_group may straddle subsets — would silently double-count or
    # drop pairs depending on filter order.
    subset_per_group = merged.groupby("match_group")["subset"].nunique()
    bad_groups = subset_per_group[subset_per_group > 1]
    assert bad_groups.empty, (
        f"{len(bad_groups)} match_group(s) span multiple subsets; first: "
        f"{bad_groups.head().to_dict()}"
    )

    rows: list[dict] = []
    for subset_name, subset_df in merged.groupby("subset", sort=False):
        for score_col in score_columns:
            res = pairwise_accuracy(
                label=subset_df["label"],
                score=subset_df[score_col],
                match_group=subset_df["match_group"],
            )
            rows.append(
                {
                    "score_type": score_col,
                    "subset": str(subset_name),
                    "value": res["value"],
                    "se": res["se"],
                    "n_pairs": res["n_pairs"],
                    "n_ties": res["n_ties"],
                    "p_value": res["p_value"],
                }
            )
    return pd.DataFrame(rows)


def compute_metrics(
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    metrics: list[str],
    score_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute classical (AUPRC/AUROC/Spearman) metrics for variant predictions.

    Aligns ``dataset`` with ``scores`` by row index. For datasets with a
    ``subset`` column, computes metrics on the full dataset (``subset='global'``)
    and on each individual subset.

    Args:
        dataset: DataFrame with ``[chrom, pos, ref, alt, label]`` and optionally
            ``subset``.
        scores: DataFrame whose columns are the model scores; row-aligned with
            ``dataset``.
        metrics: List of metric names from ``METRIC_FUNCTIONS``.
        score_columns: Score column names to evaluate. Defaults to
            ``['minus_llr', 'abs_llr']``.

    Returns:
        DataFrame with columns ``[metric, score_type, subset, value, n_pos,
        n_neg]``.
    """
    if score_columns is None:
        score_columns = ["minus_llr", "abs_llr"]

    merged = pd.concat(
        [dataset.reset_index(drop=True), scores.reset_index(drop=True)], axis=1
    )

    subsets_to_evaluate = [("global", merged)]
    if "subset" in merged.columns:
        for subset_name in merged["subset"].unique():
            subset_data = merged[merged["subset"] == subset_name]
            subsets_to_evaluate.append((subset_name, subset_data))

    results = []
    for subset_name, subset_data in subsets_to_evaluate:
        n_pos = int((subset_data["label"] == 1).sum())
        n_neg = int((subset_data["label"] == 0).sum())
        for metric_name in metrics:
            metric_func = METRIC_FUNCTIONS[metric_name]
            for score_col in score_columns:
                value = metric_func(subset_data["label"], subset_data[score_col])
                results.append(
                    {
                        "metric": metric_name,
                        "score_type": score_col,
                        "subset": subset_name,
                        "value": value,
                        "n_pos": n_pos,
                        "n_neg": n_neg,
                    }
                )
    return pd.DataFrame(results)


def aggregate_metrics(
    metric_files: list[str], dataset_names: list[str], model_steps: list[str]
) -> pd.DataFrame:
    """Aggregate metrics from multiple evaluation runs into a single DataFrame.

    Args:
        metric_files: List of paths to metric parquet files.
        dataset_names: List of dataset names corresponding to each file.
        model_steps: List of model training steps corresponding to each file.

    Returns:
        DataFrame with all per-file rows plus ``[step, dataset]`` columns,
        sorted by step and dataset.
    """
    all_metrics = []
    for file_path, dataset_name, step in zip(metric_files, dataset_names, model_steps):
        df = pd.read_parquet(file_path)
        df["dataset"] = dataset_name
        df["step"] = int(step)
        all_metrics.append(df)
    result = pd.concat(all_metrics, ignore_index=True)
    return result.sort_values(["step", "dataset"]).reset_index(drop=True)
