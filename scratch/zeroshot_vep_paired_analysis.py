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

    print("\n=== 1. Last vs middle layer ===")
    df_lm, sum_lm = family_last_vs_middle(scored)
    df_lm.to_parquet(out_dir / "last_vs_middle.parquet", index=False)
    print(sum_lm.to_string(index=False))
    print()
    print("Per-distance breakdown (last vs middle):")
    breakdown_lm = df_lm.groupby("distance").apply(
        lambda g: pd.Series(summarize_family(g)), include_groups=False
    ).reset_index()
    print(breakdown_lm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 2. Pool pairwise (flat / mean / varpos / lastpos) ===")
    df_p, sum_p = family_pool_pairwise(scored)
    df_p.to_parquet(out_dir / "pool_pairwise.parquet", index=False)
    print(sum_p.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 3. Window pairwise (within each model) ===")
    df_w, sum_w = family_window(scored)
    df_w.to_parquet(out_dir / "window_pairwise.parquet", index=False)
    print(sum_w.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 4. Best likelihood vs best embedding per cell ===")
    df_le, sum_le = family_likelihood_vs_embedding(scored)
    df_le.to_parquet(out_dir / "likelihood_vs_embedding.parquet", index=False)
    print(sum_le.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    # Per-dataset breakdown.
    per_ds = df_le.groupby("dataset").apply(
        lambda g: pd.Series(summarize_family(g)), include_groups=False
    ).reset_index()
    print("Per-dataset breakdown:")
    print(per_ds.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 5. Each score vs leaderboard baseline ===")
    df_lb, sum_lb, top_beats = family_leaderboard_baseline(scored)
    df_lb.to_parquet(out_dir / "leaderboard_baseline.parquet", index=False)
    print("Per-dataset summary (how often does a non-leaderboard score beat the leaderboard?):")
    print(sum_lb.to_string(index=False))
    print("\nTop scores that beat the leaderboard most often across cells:")
    print(top_beats.head(15).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
