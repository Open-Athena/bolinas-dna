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
from scipy.stats import spearmanr
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

    Returns:
        ``{"value", "se", "n_pairs", "n_ties"}``. ``se`` is the Wald binomial
        form ``sqrt(value * (1 - value) / n_pairs)``.
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
    return {"value": float(value), "se": float(se), "n_pairs": int(n), "n_ties": ties}


GLOBAL_SUBSET = "_global_"
MACRO_AVG_SUBSET = "_macro_avg_"


def compute_pairwise_metrics(
    dataset: pd.DataFrame,
    scores: pd.DataFrame,
    score_columns: list[str] | None = None,
    n_min: int = 30,
) -> pd.DataFrame:
    """Compute PairwiseAccuracy + SE per ``subset`` for one or more score columns.

    Aligns ``dataset`` with ``scores`` by row index (assumes same order).
    Stratifies by the ``subset`` column — one row per (subset, score_column).
    Asserts no ``match_group`` straddles subsets.

    Additionally emits two aggregate rows per ``score_col`` (sentinel subset
    values; underscore-bracketed to avoid collision with real consequence-group
    names which never start with ``_``):

    - ``subset="_global_"``: PA over **all** ``match_group``s in the input,
      regardless of subset size. Computed by a direct ``pairwise_accuracy``
      call on the full frame — not by recombining per-subset values — so the
      number is provably "score every pair, average". Same Wald SE.
    - ``subset="_macro_avg_"``: unweighted mean of per-subset values across
      subsets with ``n_pairs >= n_min``. SE = ``sqrt(Σ SE_s²) / K`` (SE of
      the unweighted mean of K independent estimates). ``n_pairs`` on this
      row is **repurposed** to record K, the count of qualifying subsets;
      ``n_ties`` is the sum across qualifying subsets.

    Args:
        dataset: DataFrame with columns ``[label, subset, match_group]`` (and
            optionally other passthrough columns).
        scores: DataFrame whose columns are the model scores; row-aligned with
            ``dataset``.
        score_columns: Score column names to evaluate. Defaults to all columns
            of ``scores``.
        n_min: Minimum ``n_pairs`` per subset to include in the macro average.
            Defaults to 30 (project-wide convention for the leaderboard
            issues). Subsets below this threshold still contribute to the
            global row.

    Returns:
        DataFrame with columns ``[score_type, subset, value, se, n_pairs,
        n_ties]``. ``score_type`` is the column name from ``scores``.
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
    per_score_rows: dict[str, list[dict]] = {sc: [] for sc in score_columns}
    for subset_name, subset_df in merged.groupby("subset", sort=False):
        for score_col in score_columns:
            res = pairwise_accuracy(
                label=subset_df["label"],
                score=subset_df[score_col],
                match_group=subset_df["match_group"],
            )
            row = {
                "score_type": score_col,
                "subset": str(subset_name),
                "value": res["value"],
                "se": res["se"],
                "n_pairs": res["n_pairs"],
                "n_ties": res["n_ties"],
            }
            rows.append(row)
            per_score_rows[score_col].append(row)

    for score_col in score_columns:
        # Global: PA over every match_group, regardless of subset.
        global_res = pairwise_accuracy(
            label=merged["label"],
            score=merged[score_col],
            match_group=merged["match_group"],
        )
        rows.append(
            {
                "score_type": score_col,
                "subset": GLOBAL_SUBSET,
                "value": global_res["value"],
                "se": global_res["se"],
                "n_pairs": global_res["n_pairs"],
                "n_ties": global_res["n_ties"],
            }
        )

        # Macro avg: unweighted mean of per-subset PAs with n_pairs >= n_min.
        qualifying = [r for r in per_score_rows[score_col] if r["n_pairs"] >= n_min]
        assert qualifying, (
            f"no subsets meet n_min={n_min} for score_type={score_col!r}; "
            f"per-subset sizes: "
            f"{ {r['subset']: r['n_pairs'] for r in per_score_rows[score_col]} }"
        )
        k = len(qualifying)
        macro_value = sum(r["value"] for r in qualifying) / k
        macro_se = math.sqrt(sum(r["se"] ** 2 for r in qualifying)) / k
        rows.append(
            {
                "score_type": score_col,
                "subset": MACRO_AVG_SUBSET,
                "value": float(macro_value),
                "se": float(macro_se),
                "n_pairs": k,
                "n_ties": sum(r["n_ties"] for r in qualifying),
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
