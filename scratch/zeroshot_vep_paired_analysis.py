"""Paired-test analysis for zeroshot_vep iter-1 — answers comparison questions
("is A better than B?") rather than significance-vs-chance.

For each comparison family we apply :func:`bolinas.evals.metrics.paired_score_comparison`
(McNemar-style closed-form sign test on discordant matched-pair outcomes) and
report:
  - per-(model, window, dataset, subset) wins/losses + one-sided p
  - BH-FDR-adjusted q-values within the comparison family
  - a summary: "across {N} cells tested, A beats B in {n_a} (q<0.05), B beats A
    in {n_b}, indeterminate in {n_indet}"

Comparison families (iter 1):
1. **Last vs middle layer** — paired across (pool, distance, model, window, dataset, subset)
2. **Pool strategies** — pairwise among {flat, mean, varpos, lastpos}, paired across the rest
3. **Window size** — pairwise among each model's {128, native, 512}
4. **Best single score per family vs `minus_llr`/`abs_llr` baseline**
5. **Best likelihood vs best embedding** per (model, window, dataset, subset)

Inputs:
  scratch/iter1/scores/ — the 45 per-variant score parquets
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from bolinas.evals.metrics import paired_score_comparison
from bolinas.zeroshot_vep.scores import (
    DISTANCES,
    LAYERS,
    POOLS,
    SCORE_DIRECTIONS,
    apply_score_directions,
)


# BH FDR helper (copied from aggregate.smk so this script is standalone).
def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    mask = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if mask.sum() == 0:
        return out
    valid = p[mask]
    order = np.argsort(valid)
    n = len(valid)
    ranks = np.arange(1, n + 1)
    adj_sorted = valid[order] * n / ranks
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adjusted = np.empty_like(valid)
    adjusted[order] = adj_sorted
    out[mask] = adjusted
    return out


# ---------------------------------------------------------------------------
# Load all 45 score parquets and apply the locked sign convention.
# ---------------------------------------------------------------------------


def load_signed_scores(scores_dir: Path) -> pd.DataFrame:
    """Load all per-variant score parquets and apply SCORE_DIRECTIONS.

    Returns a long-format DataFrame indexed by (model, window, dataset, subset,
    match_group, label) with one column per FINAL score name.
    """
    rows = []
    for parquet in sorted(scores_dir.glob("*.parquet")):
        # Filename: {model}__win{w}__{dataset}.parquet
        stem = parquet.stem
        model, win_part, dataset = stem.split("__")
        window = int(win_part.replace("win", ""))
        df = pd.read_parquet(parquet)
        signed = apply_score_directions(df)
        meta = df[["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]].reset_index(drop=True)
        out = pd.concat([meta, signed.reset_index(drop=True)], axis=1)
        out["model"] = model
        out["window"] = window
        out["dataset"] = dataset
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Comparison helpers.
# ---------------------------------------------------------------------------


def compare_pair_across_cells(
    df: pd.DataFrame,
    score_a: str,
    score_b: str,
    cell_columns: list[str],
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """For each unique `cell_columns` group in df, paired-test (score_a vs
    score_b). Returns one row per cell with McNemar stats."""
    records = []
    for cell_vals, sub in df.groupby(cell_columns, sort=False):
        if isinstance(cell_vals, tuple):
            cell_dict = dict(zip(cell_columns, cell_vals))
        else:
            cell_dict = {cell_columns[0]: cell_vals}
        res = paired_score_comparison(
            label=sub["label"],
            score_a=sub[score_a],
            score_b=sub[score_b],
            match_group=sub["match_group"],
            alternative=alternative,
        )
        records.append({**cell_dict, "score_a": score_a, "score_b": score_b, **res})
    return pd.DataFrame(records)


def summarize_family(df: pd.DataFrame, q_cutoff: float = 0.05) -> dict:
    """Summary: A wins / B wins / tied + median value across cells."""
    n = len(df)
    n_a_wins_sig = int(((df["q_value"] < q_cutoff) & (df["value"] > 0.5)).sum())
    n_b_wins_sig = int(((df["q_value"] < q_cutoff) & (df["value"] < 0.5)).sum())
    n_indet = n - n_a_wins_sig - n_b_wins_sig
    return {
        "n_cells": n,
        "n_a_wins_sig": n_a_wins_sig,
        "n_b_wins_sig": n_b_wins_sig,
        "n_indet": n_indet,
        "median_value": float(df["value"].median()),
        "mean_value": float(df["value"].mean()),
    }


def summarize_at_three_levels(
    df: pd.DataFrame, extra_group_cols: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Return {global, per_dataset, per_dataset_subset} summary tables.

    ``extra_group_cols``: prepend additional grouping columns (e.g. the pair
    identifiers like ``score_a, score_b`` or ``dist_a, dist_b``) so the
    summary breaks out by both the pair AND the breakdown axis.
    """
    extra = list(extra_group_cols or [])
    out: dict[str, pd.DataFrame] = {}

    # Global (collapses over everything except the pair identifiers).
    if extra:
        out["global"] = (
            df.groupby(extra)
            .apply(lambda g: pd.Series(summarize_family(g)), include_groups=False)
            .reset_index()
        )
    else:
        out["global"] = pd.DataFrame([summarize_family(df)])

    # Per dataset.
    by_ds_cols = extra + ["dataset"]
    out["per_dataset"] = (
        df.groupby(by_ds_cols)
        .apply(lambda g: pd.Series(summarize_family(g)), include_groups=False)
        .reset_index()
    )

    # Per (dataset, subset).
    by_dss_cols = extra + ["dataset", "subset"]
    out["per_dataset_subset"] = (
        df.groupby(by_dss_cols)
        .apply(lambda g: pd.Series(summarize_family(g)), include_groups=False)
        .reset_index()
    )
    return out


# ---------------------------------------------------------------------------
# Comparison families.
# ---------------------------------------------------------------------------


def family_last_vs_middle(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each (pool, distance, model, window, dataset, subset), compare
    embed_{distance}_{pool}_last vs embed_{distance}_{pool}_middle."""
    rows = []
    for pool in POOLS:
        for dist in DISTANCES:
            score_last = (
                f"embed_minus_dot_{pool}_last" if dist == "dot" else f"embed_{dist}_{pool}_last"
            )
            score_middle = (
                f"embed_minus_dot_{pool}_middle" if dist == "dot" else f"embed_{dist}_{pool}_middle"
            )
            sub_rows = compare_pair_across_cells(
                scored,
                score_a=score_last,
                score_b=score_middle,
                cell_columns=["model", "window", "dataset", "subset"],
                alternative="two-sided",
            )
            sub_rows["pool"] = pool
            sub_rows["distance"] = dist
            rows.append(sub_rows)
    df = pd.concat(rows, ignore_index=True)
    df["q_value"] = bh_adjust(df["p_value"].values)
    summary = pd.DataFrame([{
        "comparison": "last vs middle (across pool × distance × model × window × dataset × subset)",
        **summarize_family(df),
    }])
    return df, summary


def family_pool_pairwise(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise among {flat, mean, varpos, lastpos}, for each (layer, distance,
    model, window, dataset, subset)."""
    rows = []
    for pool_a, pool_b in combinations(POOLS, 2):
        for dist in DISTANCES:
            for layer in LAYERS:
                score_a = (
                    f"embed_minus_dot_{pool_a}_{layer}" if dist == "dot"
                    else f"embed_{dist}_{pool_a}_{layer}"
                )
                score_b = (
                    f"embed_minus_dot_{pool_b}_{layer}" if dist == "dot"
                    else f"embed_{dist}_{pool_b}_{layer}"
                )
                sub_rows = compare_pair_across_cells(
                    scored,
                    score_a=score_a,
                    score_b=score_b,
                    cell_columns=["model", "window", "dataset", "subset"],
                    alternative="two-sided",
                )
                sub_rows["pool_a"] = pool_a
                sub_rows["pool_b"] = pool_b
                sub_rows["distance"] = dist
                sub_rows["layer"] = layer
                rows.append(sub_rows)
    df = pd.concat(rows, ignore_index=True)
    df["q_value"] = bh_adjust(df["p_value"].values)
    # Summary per (pool_a, pool_b).
    grouped = df.groupby(["pool_a", "pool_b"]).apply(
        lambda g: pd.Series(summarize_family(g)), include_groups=False
    ).reset_index()
    return df, grouped


LIKELIHOOD_SIGN_CORRECT = (
    "minus_llr",
    "abs_llr",
    "minus_entropy",
    "logp_ref",
    "minus_logp_alt",
)


# Per-model "home" consequence subsets — the regions each model was designed to handle.
HOME_SUBSETS: dict[str, tuple[str, ...]] = {
    "exp55-mammals":   ("tss_proximal", "5_prime_UTR_variant"),               # promoter
    "exp58-mammals":   ("missense_variant", "synonymous_variant", "splicing"), # CDS, mammals
    "exp58-animals":   ("missense_variant", "synonymous_variant", "splicing"), # CDS, animals
    "exp59-mammals":   ("3_prime_UTR_variant",),                               # downstream
    "exp136-proj_v30": ("distal",),                                            # enhancer
}


def restrict_to_home(scored: pd.DataFrame) -> pd.DataFrame:
    """Filter the per-variant scored DataFrame to (model, subset) pairs in HOME_SUBSETS."""
    mask = pd.Series(False, index=scored.index)
    for model, subsets in HOME_SUBSETS.items():
        mask |= (scored["model"] == model) & (scored["subset"].isin(subsets))
    return scored[mask].reset_index(drop=True)


def family_likelihood_pairwise(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise among the 5 sign-correct likelihood scores, for each (model,
    window, dataset, subset). Answers "LLR vs abs(LLR) vs entropy vs logp_*?"."""
    rows = []
    for score_a, score_b in combinations(LIKELIHOOD_SIGN_CORRECT, 2):
        sub_rows = compare_pair_across_cells(
            scored,
            score_a=score_a,
            score_b=score_b,
            cell_columns=["model", "window", "dataset", "subset"],
            alternative="two-sided",
        )
        sub_rows["score_a"] = score_a
        sub_rows["score_b"] = score_b
        rows.append(sub_rows)
    df = pd.concat(rows, ignore_index=True)
    df["q_value"] = bh_adjust(df["p_value"].values)
    summary = df.groupby(["score_a", "score_b"]).apply(
        lambda g: pd.Series(summarize_family(g)), include_groups=False
    ).reset_index()
    return df, summary


def family_distance_pairwise(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise among {l2, cosine, minus_dot}, for each (pool, layer, model,
    window, dataset, subset)."""
    rows = []
    for dist_a, dist_b in combinations(DISTANCES, 2):
        for pool in POOLS:
            for layer in LAYERS:
                score_a = (
                    f"embed_minus_dot_{pool}_{layer}" if dist_a == "dot"
                    else f"embed_{dist_a}_{pool}_{layer}"
                )
                score_b = (
                    f"embed_minus_dot_{pool}_{layer}" if dist_b == "dot"
                    else f"embed_{dist_b}_{pool}_{layer}"
                )
                sub_rows = compare_pair_across_cells(
                    scored,
                    score_a=score_a,
                    score_b=score_b,
                    cell_columns=["model", "window", "dataset", "subset"],
                    alternative="two-sided",
                )
                sub_rows["dist_a"] = dist_a
                sub_rows["dist_b"] = dist_b
                sub_rows["pool"] = pool
                sub_rows["layer"] = layer
                rows.append(sub_rows)
    df = pd.concat(rows, ignore_index=True)
    df["q_value"] = bh_adjust(df["p_value"].values)
    summary = df.groupby(["dist_a", "dist_b"]).apply(
        lambda g: pd.Series(summarize_family(g)), include_groups=False
    ).reset_index()
    return df, summary


def family_window(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise comparison among windows for each (model, score, dataset, subset)."""
    rows = []
    score_cols = list(SCORE_DIRECTIONS.keys())
    # The exp136 model uses 255 (not 256) as its native. Treat 255 as middle window.
    # For other models, native = 256. Just iterate over each model's actual windows.
    for model, windows_df in scored.groupby("model"):
        windows = sorted(windows_df["window"].unique())
        for wa, wb in combinations(windows, 2):
            sub_df = windows_df[windows_df["window"].isin([wa, wb])]
            # Pivot to columns: score_a from window=wa, score_b from window=wb,
            # aligned on (dataset, subset, match_group, score).
            for score in score_cols:
                rec_a = sub_df[sub_df["window"] == wa][["dataset", "subset", "label", "match_group", score]]
                rec_b = sub_df[sub_df["window"] == wb][["dataset", "subset", "label", "match_group", score]]
                merged = rec_a.merge(
                    rec_b,
                    on=["dataset", "subset", "label", "match_group"],
                    suffixes=("_a", "_b"),
                )
                if len(merged) == 0:
                    continue
                # Compare per (dataset, subset).
                for (ds, subset), cell in merged.groupby(["dataset", "subset"]):
                    res = paired_score_comparison(
                        label=cell["label"],
                        score_a=cell[f"{score}_a"],
                        score_b=cell[f"{score}_b"],
                        match_group=cell["match_group"],
                        alternative="two-sided",
                    )
                    rows.append({
                        "model": model,
                        "window_a": wa,
                        "window_b": wb,
                        "score": score,
                        "dataset": ds,
                        "subset": subset,
                        **res,
                    })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df, pd.DataFrame()
    df["q_value"] = bh_adjust(df["p_value"].values)
    summary = df.groupby(["model", "window_a", "window_b"]).apply(
        lambda g: pd.Series(summarize_family(g)), include_groups=False
    ).reset_index()
    return df, summary


def family_likelihood_vs_embedding(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best likelihood vs best embedding per (model, window, dataset, subset).
    Pick the per-cell winners from each family (by PairwiseAccuracy) and run
    paired test on them."""
    LIKELIHOOD = ("llr", "minus_llr", "abs_llr", "logp_ref", "minus_logp_alt", "minus_entropy")
    EMBEDDING = [s for s in SCORE_DIRECTIONS if s.startswith("embed_")]

    rows = []
    for (m, w, d, s), cell in scored.groupby(["model", "window", "dataset", "subset"]):
        # Find best likelihood score and best embedding score by PairwiseAccuracy on
        # this cell. Then paired-test them.
        def acc(col):
            from bolinas.evals.metrics import pairwise_accuracy
            return pairwise_accuracy(cell["label"], cell[col], cell["match_group"])["value"]
        best_l = max(LIKELIHOOD, key=acc)
        best_e = max(EMBEDDING, key=acc)
        res = paired_score_comparison(
            label=cell["label"],
            score_a=cell[best_l],
            score_b=cell[best_e],
            match_group=cell["match_group"],
            alternative="two-sided",
        )
        rows.append({
            "model": m, "window": w, "dataset": d, "subset": s,
            "best_likelihood": best_l, "best_embedding": best_e,
            **res,
        })
    df = pd.DataFrame(rows)
    df["q_value"] = bh_adjust(df["p_value"].values)
    summary = pd.DataFrame([{
        "comparison": "best likelihood vs best embedding per cell",
        **summarize_family(df),
    }])
    return df, summary


NATIVE_WINDOW: dict[str, int] = {
    "exp55-mammals": 256,
    "exp58-mammals": 256,
    "exp58-animals": 256,
    "exp59-mammals": 256,
    "exp136-proj_v30": 255,
}


def family_home_model_vs_others(scored: pd.DataFrame) -> pd.DataFrame:
    """For each home (model, subset) pair and each dataset, compare the home
    model's best score (at its native window) against each other model's best
    score (at that model's native window) on the same matched pairs.

    "Best score" per (model, dataset, subset) is by PairwiseAccuracy.
    """
    from bolinas.evals.metrics import pairwise_accuracy
    score_cols = list(SCORE_DIRECTIONS.keys())

    def best_score_col(cell: pd.DataFrame) -> str:
        return max(
            score_cols,
            key=lambda s: pairwise_accuracy(
                cell["label"], cell[s], cell["match_group"]
            )["value"],
        )

    records = []
    home_pairs = [(m, s) for m, ss in HOME_SUBSETS.items() for s in ss]
    for home_model, subset in home_pairs:
        w_home = NATIVE_WINDOW[home_model]
        for dataset in scored["dataset"].unique():
            cell_home = scored[
                (scored["model"] == home_model)
                & (scored["window"] == w_home)
                & (scored["dataset"] == dataset)
                & (scored["subset"] == subset)
            ]
            if len(cell_home) == 0:
                continue
            s_home = best_score_col(cell_home)
            for other_model in scored["model"].unique():
                if other_model == home_model:
                    continue
                w_other = NATIVE_WINDOW[other_model]
                cell_other = scored[
                    (scored["model"] == other_model)
                    & (scored["window"] == w_other)
                    & (scored["dataset"] == dataset)
                    & (scored["subset"] == subset)
                ]
                if len(cell_other) == 0:
                    continue
                s_other = best_score_col(cell_other)
                # Align both cells on (match_group, label).
                a = cell_home[["match_group", "label", s_home]].rename(columns={s_home: "score"})
                b = cell_other[["match_group", "label", s_other]].rename(columns={s_other: "score"})
                merged = a.merge(b, on=["match_group", "label"], suffixes=("_home", "_other"))
                if len(merged) == 0:
                    continue
                res = paired_score_comparison(
                    label=merged["label"],
                    score_a=merged["score_home"],
                    score_b=merged["score_other"],
                    match_group=merged["match_group"],
                    alternative="two-sided",
                )
                records.append({
                    "subset": subset, "dataset": dataset,
                    "home_model": home_model, "home_window": w_home, "home_score": s_home,
                    "other_model": other_model, "other_window": w_other, "other_score": s_other,
                    **res,
                })
    df = pd.DataFrame(records)
    if len(df) == 0:
        return df
    df["q_value"] = bh_adjust(df["p_value"].values)
    return df


def family_leaderboard_baseline(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each non-leaderboard score vs the leaderboard score for that dataset.
    Leaderboard: minus_llr for mendelian, abs_llr for complex/eqtl."""
    LEADERBOARD_PER_DATASET = {
        "mendelian_traits": "minus_llr",
        "complex_traits": "abs_llr",
        "eqtl": "abs_llr",
    }
    rows = []
    score_cols = list(SCORE_DIRECTIONS.keys())
    for (m, w, d, s), cell in scored.groupby(["model", "window", "dataset", "subset"]):
        leaderboard = LEADERBOARD_PER_DATASET[d]
        for score in score_cols:
            if score == leaderboard:
                continue
            res = paired_score_comparison(
                label=cell["label"],
                score_a=cell[score],
                score_b=cell[leaderboard],
                match_group=cell["match_group"],
                alternative="two-sided",
            )
            rows.append({
                "model": m, "window": w, "dataset": d, "subset": s,
                "score_a": score, "leaderboard": leaderboard, **res,
            })
    df = pd.DataFrame(rows)
    df["q_value"] = bh_adjust(df["p_value"].values)
    # Summary per dataset.
    summary = df.groupby("dataset").apply(
        lambda g: pd.Series({
            "n_cells": len(g),
            "n_beats_leaderboard_sig": int(((g["q_value"] < 0.05) & (g["value"] > 0.5)).sum()),
            "n_loses_to_leaderboard_sig": int(((g["q_value"] < 0.05) & (g["value"] < 0.5)).sum()),
            "n_indet": len(g) - int(((g["q_value"] < 0.05)).sum()),
        }), include_groups=False
    ).reset_index()
    # Top-10 scores that beat the leaderboard most often (across cells).
    leader_wins = df[(df["q_value"] < 0.05) & (df["value"] > 0.5)]
    top_beats = leader_wins.groupby("score_a").size().reset_index(name="n_wins_vs_leaderboard").sort_values("n_wins_vs_leaderboard", ascending=False)
    return df, summary, top_beats


def main(argv: list[str] | None = None) -> int:
    scores_dir = Path("scratch/iter1/scores")
    print(f"[load] scanning {scores_dir}")
    scored = load_signed_scores(scores_dir)
    print(f"[load] loaded {len(scored)} rows across "
          f"{scored.groupby(['model','window','dataset']).ngroups} (model,window,dataset) cells")

    out_dir = Path("scratch/iter1/paired")
    out_dir.mkdir(parents=True, exist_ok=True)

    def emit(name: str, df: pd.DataFrame, extra: list[str] | None, out_filename: str):
        df.to_parquet(out_dir / out_filename, index=False)
        levels = summarize_at_three_levels(df, extra_group_cols=extra)
        print(f"\n=== {name} ===")
        print(f"-- global --")
        print(levels["global"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\n-- per dataset --")
        print(levels["per_dataset"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\n-- per (dataset, subset) --")
        print(levels["per_dataset_subset"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        # Save the 3-level summaries.
        levels["global"].to_parquet(out_dir / f"{out_filename}.global.parquet", index=False)
        levels["per_dataset"].to_parquet(out_dir / f"{out_filename}.per_dataset.parquet", index=False)
        levels["per_dataset_subset"].to_parquet(out_dir / f"{out_filename}.per_dataset_subset.parquet", index=False)

    print("\n========== 1. Last vs middle layer ==========")
    df_lm, _ = family_last_vs_middle(scored)
    emit("Last vs middle (overall)", df_lm, extra=None, out_filename="last_vs_middle.parquet")
    print("\n-- by distance × layer comparison --")
    emit("Last vs middle by distance", df_lm, extra=["distance"], out_filename="last_vs_middle_by_distance.parquet")

    print("\n========== 2. Pool pairwise (flat / mean / varpos / lastpos) ==========")
    df_p, _ = family_pool_pairwise(scored)
    emit("Pool pairwise", df_p, extra=["pool_a", "pool_b"], out_filename="pool_pairwise.parquet")

    print("\n========== 3. Distance pairwise (l2 / cosine / minus_dot) ==========")
    df_d, _ = family_distance_pairwise(scored)
    emit("Distance pairwise", df_d, extra=["dist_a", "dist_b"], out_filename="distance_pairwise.parquet")

    print("\n========== 4. Likelihood pairwise ==========")
    df_l, _ = family_likelihood_pairwise(scored)
    emit("Likelihood pairwise", df_l, extra=["score_a", "score_b"], out_filename="likelihood_pairwise.parquet")

    print("\n========== 5. Window pairwise (within each model) ==========")
    df_w, _ = family_window(scored)
    emit("Window pairwise", df_w, extra=["model", "window_a", "window_b"], out_filename="window_pairwise.parquet")

    print("\n========== 6. Best likelihood vs best embedding per cell ==========")
    df_le, _ = family_likelihood_vs_embedding(scored)
    emit("Best likelihood vs best embedding", df_le, extra=None, out_filename="likelihood_vs_embedding.parquet")

    # -------- HOME-SUBSET-RESTRICTED RE-RUNS --------
    print("\n\n############################################################")
    print("## HOME-SUBSET ANALYSIS — each model on its target consequence subsets")
    print("############################################################")
    print(f"HOME_SUBSETS = {dict(HOME_SUBSETS)}")
    home = restrict_to_home(scored)
    print(f"[home] {len(home)} rows after restriction (was {len(scored)})")
    print(f"[home] cells: {home.groupby(['model','window','dataset','subset']).ngroups}")

    print("\n========== 1H. Last vs middle (HOME) ==========")
    df_lmh, _ = family_last_vs_middle(home)
    emit("Last vs middle (HOME)", df_lmh, extra=None, out_filename="home_last_vs_middle.parquet")

    print("\n========== 2H. Pool pairwise (HOME) ==========")
    df_ph, _ = family_pool_pairwise(home)
    emit("Pool pairwise (HOME)", df_ph, extra=["pool_a", "pool_b"], out_filename="home_pool_pairwise.parquet")

    print("\n========== 3H. Distance pairwise (HOME) ==========")
    df_dh, _ = family_distance_pairwise(home)
    emit("Distance pairwise (HOME)", df_dh, extra=["dist_a", "dist_b"], out_filename="home_distance_pairwise.parquet")

    print("\n========== 4H. Likelihood pairwise (HOME) ==========")
    df_lh, _ = family_likelihood_pairwise(home)
    emit("Likelihood pairwise (HOME)", df_lh, extra=["score_a", "score_b"], out_filename="home_likelihood_pairwise.parquet")

    print("\n========== 5H. Window pairwise (HOME) ==========")
    df_wh, _ = family_window(home)
    emit("Window pairwise (HOME)", df_wh, extra=["model", "window_a", "window_b"], out_filename="home_window_pairwise.parquet")

    print("\n========== 6H. Best likelihood vs best embedding (HOME) ==========")
    df_leh, _ = family_likelihood_vs_embedding(home)
    emit("Best likelihood vs best embedding (HOME)", df_leh, extra=None, out_filename="home_likelihood_vs_embedding.parquet")

    print("\n========== 8. HOME vs OTHER models on each home subset ==========")
    print("For each (home_model, home_subset, dataset), compare the home model's best score "
          "(at its native window) against every OTHER model's best score (at its native window) "
          "on the SAME matched pairs. value > 0.5 means HOME model wins discordant pairs more often.")
    df_homevsother = family_home_model_vs_others(scored)
    df_homevsother.to_parquet(out_dir / "home_vs_other.parquet", index=False)
    print(f"-- summary per (subset, dataset) --")
    if len(df_homevsother) > 0:
        per_subset_ds = df_homevsother.groupby(["subset", "dataset"]).apply(
            lambda g: pd.Series({
                "n_other_models": len(g),
                "n_home_wins": int(((g['q_value']<0.05) & (g['value']>0.5)).sum()),
                "n_other_wins": int(((g['q_value']<0.05) & (g['value']<0.5)).sum()),
                "indet": int((g['q_value']>=0.05).sum()),
                "median_value": float(g['value'].median()),
                "mean_value": float(g['value'].mean()),
            }), include_groups=False
        ).reset_index()
        print(per_subset_ds.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"\n-- significant losses for the home model (sorted) --")
        losses = df_homevsother[(df_homevsother['q_value']<0.05) & (df_homevsother['value']<0.5)].sort_values('value')
        if len(losses):
            print(losses[['subset','dataset','home_model','home_score','other_model','other_score','value','q_value']].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        else:
            print("(none — home model is never significantly beaten on its home subset)")

    print("\n========== 7. Each score vs leaderboard baseline ==========")
    df_lb, _, top_beats = family_leaderboard_baseline(scored)
    df_lb.to_parquet(out_dir / "leaderboard_baseline.parquet", index=False)
    levels = summarize_at_three_levels(df_lb, extra_group_cols=None)
    print("-- global --")
    print(levels["global"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n-- per dataset --")
    print(levels["per_dataset"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n-- per (dataset, subset) --")
    print(levels["per_dataset_subset"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nTop scores that beat the leaderboard most often across cells:")
    print(top_beats.head(15).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
