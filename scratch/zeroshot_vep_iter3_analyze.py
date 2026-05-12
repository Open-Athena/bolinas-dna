"""Iter 3 analysis: downstream-effect scores from exp55-mammals on
tss_proximal + 5_prime_UTR_variant only.

For each (dataset, subset) cell, compute:
  - PairwiseAccuracy + one-sided sign-test p_value for each of the 8 down_* scores
  - Paired McNemar vs the iter-1 winners on the SAME matched-pair set:
      - minus_llr        (leaderboard baseline)
      - logp_ref         (iter-1 winner on tss_proximal mendelian)
      - embed_cosine_mean_last (iter-1 winner on 5'UTR mendelian)
      - embed_l2_flat_last (iter-1 missense / mendelian-distal-non-coding-go-to)

The iter-1 score columns are read from the iter-1 scores parquets
(scratch/iter1/scores/exp55-mammals__win256__{dataset}.parquet).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bolinas.evals.metrics import paired_score_comparison, pairwise_accuracy
from bolinas.zeroshot_vep.scores import apply_score_directions

DOWN_COLS = [
    "down_jsd_mean", "down_jsd_max",
    "down_l1_mean",  "down_l1_max",
    "down_l2_mean",  "down_l2_max",
    "down_linf_mean","down_linf_max",
]
ITER1_COMPARISONS = ["minus_llr", "logp_ref", "embed_cosine_mean_last", "embed_l2_flat_last"]
TARGET_SUBSETS = ["tss_proximal", "5_prime_UTR_variant"]


def bh_adjust(p):
    p = np.asarray(p, dtype=np.float64)
    mask = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if mask.sum() == 0:
        return out
    valid = p[mask]
    order = np.argsort(valid)
    n = len(valid)
    ranks = np.arange(1, n + 1)
    adj = valid[order] * n / ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out[mask] = adj[np.argsort(order)]
    return out


def load_merged(dataset: str) -> pd.DataFrame:
    """Load iter-3 down_* parquet, merge with iter-1 signed scores on (chrom,pos,ref,alt)."""
    iter3 = pd.read_parquet(f"scratch/iter3/iter3_exp55-mammals__win256__{dataset}.parquet")
    iter1_raw = pd.read_parquet(f"scratch/iter1/scores/exp55-mammals__win256__{dataset}.parquet")
    iter1_signed = apply_score_directions(iter1_raw)
    iter1 = pd.concat([
        iter1_raw[["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]].reset_index(drop=True),
        iter1_signed.reset_index(drop=True),
    ], axis=1)
    merged = iter3.merge(
        iter1.drop(columns=["label", "subset", "match_group"]),
        on=["chrom", "pos", "ref", "alt"],
        how="inner",
    )
    return merged


def main() -> int:
    print("[iter3-analyze] loading 3 datasets, merging iter3 ↔ iter1 scores")
    all_records_solo = []
    all_records_paired = []
    for dataset in ("mendelian_traits", "complex_traits", "eqtl"):
        df = load_merged(dataset)
        df = df[df["subset"].isin(TARGET_SUBSETS)]
        print(f"  {dataset}: {len(df)} variants in target subsets "
              f"({dict(df.groupby('subset').size())})")

        # Solo: PairwiseAccuracy + p-value for each down_* score per subset.
        for subset, sub in df.groupby("subset"):
            for sc in DOWN_COLS:
                res = pairwise_accuracy(
                    sub["label"], sub[sc], sub["match_group"], alternative="greater"
                )
                all_records_solo.append({
                    "dataset": dataset, "subset": subset, "score": sc, **res,
                })

        # Paired: each down_* vs each iter-1 comparison.
        for subset, sub in df.groupby("subset"):
            for down in DOWN_COLS:
                for base in ITER1_COMPARISONS:
                    if base not in sub.columns:
                        continue
                    res = paired_score_comparison(
                        label=sub["label"],
                        score_a=sub[down],
                        score_b=sub[base],
                        match_group=sub["match_group"],
                        alternative="two-sided",
                    )
                    all_records_paired.append({
                        "dataset": dataset, "subset": subset,
                        "down": down, "baseline": base, **res,
                    })

    solo = pd.DataFrame(all_records_solo)
    paired = pd.DataFrame(all_records_paired)
    solo["q_value"] = bh_adjust(solo["p_value"].values)
    paired["q_value"] = bh_adjust(paired["p_value"].values)

    Path("scratch/iter3").mkdir(parents=True, exist_ok=True)
    solo.to_parquet("scratch/iter3/iter3_solo_metrics.parquet", index=False)
    paired.to_parquet("scratch/iter3/iter3_paired_vs_baselines.parquet", index=False)

    print("\n=== Solo PairwiseAccuracy per (dataset, subset) ===")
    print(solo.sort_values(["dataset", "subset", "value"], ascending=[True, True, False])[
        ["dataset", "subset", "score", "value", "n_pairs", "p_value", "q_value"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Best down_* score per (dataset, subset).
    print("\n=== Best down_* per (dataset, subset) ===")
    best = solo.sort_values("value", ascending=False).groupby(["dataset", "subset"]).head(1)
    print(best.sort_values(["dataset", "subset"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Paired summary: how many down_* × baseline cells show significant difference?
    print("\n=== Paired comparison summary: how does each down_* compare to each baseline? ===")
    # For each (down, baseline, dataset, subset): value > 0.5 = down wins; value < 0.5 = baseline wins.
    print(paired.sort_values("value", ascending=False)[
        ["dataset", "subset", "down", "baseline", "value", "p_value", "q_value", "n_a_wins", "n_b_wins"]
    ].head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Significant cells (q < 0.05): down_* beats baseline ===")
    wins = paired[(paired["q_value"] < 0.05) & (paired["value"] > 0.5)]
    if len(wins):
        print(wins[["dataset", "subset", "down", "baseline", "value", "q_value"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("(none)")

    print("\n=== Significant cells (q < 0.05): down_* LOSES to baseline ===")
    losses = paired[(paired["q_value"] < 0.05) & (paired["value"] < 0.5)]
    if len(losses):
        print(losses[["dataset", "subset", "down", "baseline", "value", "q_value"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("(none)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
