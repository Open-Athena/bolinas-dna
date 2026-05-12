"""Iter 2: rank-based composite scores from iter 1's surviving choices.

Builds composite scores on top of the per-variant signed scores from iter 1.
Composites are rank averages within each (dataset, subset) — comparable across
heterogeneous scoring rules without requiring a learned weighting (zero-shot).

Composites considered:
1. ``rk_minus_llr_plus_l2flat_last`` — `minus_llr` + `embed_l2_flat_last`
2. ``rk_minus_llr_plus_top_emb`` — `minus_llr` + 4 strongest iter-1 embeddings
   (`embed_l2_flat_last`, `embed_cosine_flat_last`, `embed_l2_mean_last`,
   `embed_cosine_mean_last`)
3. ``rk_subset_tailored`` — for each variant, pick the iter-1 per-subset winner:
   - splicing + 5_prime_UTR_variant → `minus_llr`
   - missense_variant + synonymous_variant → `embed_l2_flat_last`
   - distal → `minus_entropy`
   - tss_proximal → `logp_ref`
   - 3_prime_UTR_variant → `embed_minus_dot_flat_middle`
   - non_coding_transcript_exon_variant → `embed_cosine_lastpos_middle`
   This is NOT a single zero-shot score (different per subset) but encodes
   "best-of-iter-1" — useful as an upper bound estimate.

We also drop varpos + dot scores from the base 30 (iter 1 showed they're
strictly worse than alternatives) for the composite ranking.

Then we re-run the metrics + paired-test analysis with the composites added.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bolinas.evals.metrics import paired_score_comparison, pairwise_accuracy
from bolinas.zeroshot_vep.scores import SCORE_DIRECTIONS, apply_score_directions


# ---------------------------------------------------------------------------
# Composite definitions.
# ---------------------------------------------------------------------------


TOP_EMBEDDINGS = (
    "embed_l2_flat_last",
    "embed_cosine_flat_last",
    "embed_l2_mean_last",
    "embed_cosine_mean_last",
)


SUBSET_TAILORED: dict[str, str] = {
    "splicing": "minus_llr",
    "5_prime_UTR_variant": "minus_llr",
    "missense_variant": "embed_l2_flat_last",
    "synonymous_variant": "embed_l2_flat_last",
    "distal": "minus_entropy",
    "tss_proximal": "logp_ref",
    "3_prime_UTR_variant": "embed_minus_dot_flat_middle",
    "non_coding_transcript_exon_variant": "embed_cosine_lastpos_middle",
}


def add_rank_composites(scored: pd.DataFrame) -> pd.DataFrame:
    """Add 3 composite score columns to a per-variant signed-score DataFrame.

    Ranks computed within (model, window, dataset, subset) groups so the
    composites are comparable across heterogeneous scoring rules.
    """
    out = scored.copy()
    group_cols = ["model", "window", "dataset", "subset"]

    # Helper: rank a column within groups (1 = lowest, ties=mean).
    def _rank(col: str) -> pd.Series:
        return out.groupby(group_cols)[col].rank(method="average")

    # Composite 1: minus_llr + embed_l2_flat_last (simple pair).
    r1 = _rank("minus_llr")
    r2 = _rank("embed_l2_flat_last")
    out["rk_minus_llr_plus_l2flat_last"] = (r1 + r2).values

    # Composite 2: minus_llr + top-4 embeddings (mean of 5 ranks).
    ranks = [_rank("minus_llr")] + [_rank(c) for c in TOP_EMBEDDINGS]
    out["rk_minus_llr_plus_top_emb"] = np.mean(np.column_stack(ranks), axis=1)

    # Composite 3: subset-tailored — use the iter-1 per-subset winner.
    # Implementation: for each row, look up the score column dictated by its
    # subset, then convert to a rank within the same group. The "rank" here
    # is just the score itself's rank (single-score per subset), giving us
    # a hybrid score that uses different rules per subset.
    tailored_score = np.empty(len(out), dtype=np.float32)
    for subset, score_col in SUBSET_TAILORED.items():
        mask = (out["subset"] == subset).values
        tailored_score[mask] = out.loc[mask, score_col].values
    out["rk_subset_tailored"] = tailored_score

    return out


def composite_columns() -> list[str]:
    return ["rk_minus_llr_plus_l2flat_last", "rk_minus_llr_plus_top_emb", "rk_subset_tailored"]


# ---------------------------------------------------------------------------
# BH adjustment + summary helpers (copied from iter-1 analysis for standalone use).
# ---------------------------------------------------------------------------


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
# Per-cell metrics + paired comparisons.
# ---------------------------------------------------------------------------


def per_cell_pairwise_accuracy(
    scored: pd.DataFrame, score_cols: list[str]
) -> pd.DataFrame:
    """For each (model, window, dataset, subset, score), compute PairwiseAccuracy
    + closed-form one-sided sign-test p_value (H1: acc > 0.5).
    """
    rows = []
    for (m, w, d, s), cell in scored.groupby(["model", "window", "dataset", "subset"]):
        for score in score_cols:
            res = pairwise_accuracy(
                cell["label"], cell[score], cell["match_group"],
                alternative="greater",
            )
            rows.append({
                "model": m, "window": w, "dataset": d, "subset": s, "score": score,
                **res,
            })
    df = pd.DataFrame(rows)
    # BH within (dataset).
    df["q_value"] = np.nan
    for ds, idx in df.groupby("dataset").groups.items():
        df.loc[idx, "q_value"] = bh_adjust(df.loc[idx, "p_value"].values)
    return df


def paired_composite_vs_components(
    scored: pd.DataFrame,
    composite: str,
    components: list[str],
) -> pd.DataFrame:
    """For each (model, window, dataset, subset), paired-test composite vs each
    of its components. Returns one row per (cell, component) with McNemar stats.
    """
    rows = []
    for (m, w, d, s), cell in scored.groupby(["model", "window", "dataset", "subset"]):
        for component in components:
            res = paired_score_comparison(
                label=cell["label"],
                score_a=cell[composite],
                score_b=cell[component],
                match_group=cell["match_group"],
                alternative="two-sided",
            )
            rows.append({
                "model": m, "window": w, "dataset": d, "subset": s,
                "composite": composite, "component": component,
                **res,
            })
    df = pd.DataFrame(rows)
    df["q_value"] = bh_adjust(df["p_value"].values)
    return df


def main() -> int:
    scores_dir = Path("scratch/iter1/scores")
    out_dir = Path("scratch/iter2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load + sign-correct.
    print(f"[load] reading 45 score parquets from {scores_dir}")
    frames = []
    for parquet in sorted(scores_dir.glob("*.parquet")):
        stem = parquet.stem
        model, win_part, dataset = stem.split("__")
        window = int(win_part.replace("win", ""))
        raw = pd.read_parquet(parquet)
        signed = apply_score_directions(raw)
        meta = raw[["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]].reset_index(drop=True)
        out = pd.concat([meta, signed.reset_index(drop=True)], axis=1)
        out["model"] = model
        out["window"] = window
        out["dataset"] = dataset
        frames.append(out)
    scored = pd.concat(frames, ignore_index=True)
    print(f"[load] {len(scored)} rows, {scored.groupby(['model','window','dataset']).ngroups} cells")

    # Add rank composites.
    print("[composites] adding 3 rank-based composites")
    scored = add_rank_composites(scored)
    composites = composite_columns()

    # Per-cell metrics for composites + key components (PairwiseAccuracy + p + q).
    base_for_comparison = [
        "minus_llr", "abs_llr",
        "embed_l2_flat_last", "embed_cosine_flat_last", "embed_l2_mean_last",
        "minus_entropy", "logp_ref",
    ]
    print("[metrics] computing per-cell PairwiseAccuracy for composites + key components")
    metrics_df = per_cell_pairwise_accuracy(scored, composites + base_for_comparison)
    metrics_df.to_parquet(out_dir / "iter2_per_cell_metrics.parquet", index=False)

    # Headline: how do composites rank vs single scores globally?
    print("\n=== Iter-2 composite vs best single-score baselines, global_macro per dataset ===")
    macros = (
        metrics_df.groupby(["dataset", "score"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "global_macro"})
    )
    macros = macros.sort_values(["dataset", "global_macro"], ascending=[True, False])
    print(macros.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Per-subset best score (mendelian).
    print("\n=== Best score per (dataset, subset), q < 0.05 — composites highlighted ===")
    best = (
        metrics_df[metrics_df["q_value"] < 0.05]
        .sort_values("value", ascending=False)
        .groupby(["dataset", "subset"])
        .head(1)
    )
    print(best.sort_values(["dataset", "subset"])[
        ["dataset", "subset", "score", "model", "window", "value", "n_pairs", "q_value"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Paired: composite vs each of its components.
    print("\n=== Paired (composite vs component) ===")
    for composite, components in [
        ("rk_minus_llr_plus_l2flat_last", ["minus_llr", "embed_l2_flat_last"]),
        ("rk_minus_llr_plus_top_emb", ["minus_llr"] + list(TOP_EMBEDDINGS)),
        ("rk_subset_tailored", ["minus_llr", "embed_l2_flat_last"]),
    ]:
        df = paired_composite_vs_components(scored, composite, components)
        df.to_parquet(out_dir / f"paired_{composite}.parquet", index=False)
        per_ds = df.groupby(["composite", "component", "dataset"]).apply(
            lambda g: pd.Series({
                "n_cells": len(g),
                "n_composite_wins": int(((g["q_value"]<0.05) & (g["value"]>0.5)).sum()),
                "n_component_wins": int(((g["q_value"]<0.05) & (g["value"]<0.5)).sum()),
                "median_value": float(g["value"].median()),
                "mean_value": float(g["value"].mean()),
            }), include_groups=False
        ).reset_index()
        print(f"\n--- {composite} ---")
        print(per_ds.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n[done] all outputs in scratch/iter2/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
