"""Iter 2 follow-up: investigate why eqtl shows no significant signal.

Hypothesis space:
(a) Matched-pair construction is too tight — positives and matched negatives
    are so similar (in sequence context) that no LLR or embedding distance
    can separate them.
(b) Zero-shot bolinas gLMs genuinely don't capture eqtl signal.
(c) bf16 noise drowns out a small true effect size.

Diagnostics to distinguish:

1. **Per-variant signed-score distribution, eqtl positives vs matched
   negatives**: if pos / neg distributions are nearly identical (mean shift <<
   per-pair within-class variance), (a) is supported.
2. **Position-bias check**: does the same model on the same dataset show
   meaningful eqtl LLR distribution shape, or are all eqtl variants showing
   LLR ≈ 0? If LLR is genuinely tiny for eqtl positives → (b).
3. **eqtl vs mendelian**: same model + same window + same matched-pair
   construction. If mendelian shows a clear pos > neg shift but eqtl doesn't,
   on the same subset, it's NOT a methodology issue.

We use `exp58-animals` at win=256 (the iter-1 strongest model) and inspect the
distal subset (largest eqtl bucket: ~1,678 pairs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bolinas.zeroshot_vep.scores import apply_score_directions


MODEL = "exp58-animals"
WINDOW = 256


def load(model: str, window: int, dataset: str) -> pd.DataFrame:
    p = Path(f"scratch/iter1/scores/{model}__win{window}__{dataset}.parquet")
    raw = pd.read_parquet(p)
    signed = apply_score_directions(raw)
    meta = raw[["label", "subset", "match_group"]].reset_index(drop=True)
    return pd.concat([meta, signed.reset_index(drop=True)], axis=1)


def main() -> int:
    print(f"[load] {MODEL} win={WINDOW}, all 3 datasets")
    dfs = {
        ds: load(MODEL, WINDOW, ds)
        for ds in ("mendelian_traits", "complex_traits", "eqtl")
    }

    # For each (dataset, subset, score), summarize pos vs neg distributions.
    key_scores = ("minus_llr", "abs_llr", "minus_entropy", "logp_ref",
                  "embed_l2_flat_last", "embed_cosine_flat_last", "embed_l2_mean_last")
    rows = []
    for ds_name, df in dfs.items():
        for subset, sub in df.groupby("subset"):
            pos = sub[sub["label"] == 1]
            neg = sub[sub["label"] == 0]
            if len(pos) < 5:
                continue
            for sc in key_scores:
                p_vals = pos[sc].values
                n_vals = neg[sc].values
                # Paired: per match_group, pos_score - neg_score
                pos_idx = pos.set_index("match_group")[sc]
                neg_idx = neg.set_index("match_group")[sc]
                common = pos_idx.index.intersection(neg_idx.index)
                diff = (pos_idx.loc[common] - neg_idx.loc[common]).values
                rows.append({
                    "dataset": ds_name, "subset": subset, "score": sc,
                    "n_pairs": len(common),
                    "pos_mean": float(p_vals.mean()),
                    "pos_std": float(p_vals.std()),
                    "neg_mean": float(n_vals.mean()),
                    "neg_std": float(n_vals.std()),
                    "mean_diff_pos_neg": float(diff.mean()),
                    "std_diff_pos_neg": float(diff.std()),
                    "cohens_d_paired": float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0,
                    "frac_pos_gt_neg": float((diff > 0).mean()),
                })
    summary = pd.DataFrame(rows)
    summary.to_parquet("scratch/iter2/eqtl_null_summary.parquet", index=False)

    print("\n=== Paired difference stats per (dataset, subset, score) — exp58-animals @ win=256 ===")
    print("Cohen's d (paired) = mean_diff / std_diff. |d| > 0.2 = small effect; |d| > 0.5 = medium; |d| > 0.8 = large.")
    print()

    # For each (dataset, score), find the distal subset row.
    for sc in key_scores:
        print(f"\n--- {sc} on `distal` subset ---")
        print(summary[(summary["subset"] == "distal") & (summary["score"] == sc)][
            ["dataset", "n_pairs", "pos_mean", "neg_mean", "mean_diff_pos_neg",
             "std_diff_pos_neg", "cohens_d_paired", "frac_pos_gt_neg"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Cross-subset: for eqtl, which subset has the most signal?
    print("\n=== eqtl: Cohen's d_paired across subsets, for each key score ===")
    eqtl_df = summary[summary["dataset"] == "eqtl"]
    pivot = eqtl_df.pivot_table(index="subset", columns="score", values="cohens_d_paired")
    print(pivot.to_string(float_format=lambda x: f"{x:+.3f}"))

    # Same for mendelian for contrast.
    print("\n=== mendelian: Cohen's d_paired across subsets, for each key score ===")
    m_df = summary[summary["dataset"] == "mendelian_traits"]
    pivot = m_df.pivot_table(index="subset", columns="score", values="cohens_d_paired")
    print(pivot.to_string(float_format=lambda x: f"{x:+.3f}"))

    # Summary verdict.
    print("\n=== Verdict ===")
    eqtl_distal = summary[(summary["dataset"] == "eqtl") & (summary["subset"] == "distal")]
    mendelian_distal = summary[(summary["dataset"] == "mendelian_traits") & (summary["subset"] == "distal")]
    print(f"eqtl distal max |Cohen's d|: {eqtl_distal['cohens_d_paired'].abs().max():.4f}")
    print(f"mendelian distal max |Cohen's d|: {mendelian_distal['cohens_d_paired'].abs().max():.4f}")
    print(f"eqtl distal max frac(pos > neg): {eqtl_distal['frac_pos_gt_neg'].max():.4f}")
    print(f"mendelian distal max frac(pos > neg): {mendelian_distal['frac_pos_gt_neg'].max():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
