"""Iter 4 analysis: FWD vs RC vs AVG strand handling on exp55-mammals × mendelian.

For each of the 30 raw iter-1 scores, build 3 versions:
  - FWD: original forward-strand score (from iter-1 parquet)
  - RC:  reverse-complement-strand score (from iter-4 RC parquet)
  - AVG: simple per-variant mean of FWD + RC

Apply sign convention (SCORE_DIRECTIONS from iter-1) AFTER averaging.
Same answer either way since the average is linear and signs are ±1.

Reports:
  - PairwiseAccuracy + closed-form one-sided sign-test q (BH within 3 modes × 30 scores × 8 subsets = 720 tests) per (mode, score, subset)
  - Variability per (score, subset): Pearson, Spearman, RMSE between FWD and RC raw values across variants
  - Paired McNemar AVG vs FWD, AVG vs RC, FWD vs RC, on the same matched pairs per (score, subset)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from bolinas.evals.metrics import paired_score_comparison, pairwise_accuracy
from bolinas.zeroshot_vep.scores import (
    SCORE_DIRECTIONS,
    SCORE_NAMES,
    apply_score_directions,
)


def bh_adjust(p):
    p = np.asarray(p, dtype=np.float64); mask = ~np.isnan(p); out = np.full_like(p, np.nan)
    if mask.sum() == 0: return out
    v = p[mask]; o = np.argsort(v); n = len(v); r = np.arange(1, n + 1)
    a = v[o] * n / r; a = np.minimum.accumulate(a[::-1])[::-1]; a = np.minimum(a, 1.0)
    out[mask] = a[np.argsort(o)]
    return out


def main() -> int:
    out_dir = Path("scratch/iter4")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load iter-1 forward + iter-4 RC parquets, merge by variant key.
    print("[iter4] loading FWD (iter-1) and RC (iter-4) parquets")
    fwd_raw = pd.read_parquet("scratch/iter1/scores/exp55-mammals__win256__mendelian_traits.parquet")
    rc = pd.read_parquet("scratch/iter4/iter4_rc_exp55-mammals__win256__mendelian_traits.parquet")
    key = ["chrom", "pos", "ref", "alt"]
    # Align: keep only meta + raw scores from both, rename RC scores with suffix.
    rc_scores = rc[key + SCORE_NAMES].rename(columns={c: f"{c}__rc" for c in SCORE_NAMES})
    merged = fwd_raw.merge(rc_scores, on=key, how="inner")
    assert len(merged) == len(fwd_raw) == len(rc), "FWD/RC row counts differ"
    print(f"[iter4] {len(merged)} variants merged")

    # Build the 3 mode dataframes of SIGNED scores.
    # Strategy:
    #   - signed_fwd[score] = signed FWD value
    #   - signed_rc[score]  = signed RC value
    #   - signed_avg[score] = (signed_fwd + signed_rc) / 2
    signed_fwd = apply_score_directions(merged[SCORE_NAMES])
    rc_raw_renamed = merged[[f"{c}__rc" for c in SCORE_NAMES]].rename(columns=lambda c: c.replace("__rc", ""))
    signed_rc = apply_score_directions(rc_raw_renamed)
    signed_avg = (signed_fwd + signed_rc) / 2.0

    final_names = list(SCORE_DIRECTIONS.keys())  # the 30 final names

    # ---- (A) Solo PairwiseAccuracy per (mode, score, subset) ----
    records = []
    for mode_name, signed in [("FWD", signed_fwd), ("RC", signed_rc), ("AVG", signed_avg)]:
        for subset, sub_idx in merged.groupby("subset").groups.items():
            sub = merged.loc[sub_idx]
            for score in final_names:
                res = pairwise_accuracy(
                    sub["label"], signed.loc[sub_idx, score], sub["match_group"],
                    alternative="greater",
                )
                records.append({
                    "mode": mode_name, "score": score, "subset": subset,
                    **res,
                })
    solo = pd.DataFrame(records)
    solo["q_value"] = bh_adjust(solo["p_value"].values)
    solo.to_parquet(out_dir / "iter4_solo_metrics.parquet", index=False)

    # ---- (B) Variability between FWD and RC per (score, subset) ----
    var_records = []
    for subset, sub_idx in merged.groupby("subset").groups.items():
        for score in final_names:
            f = signed_fwd.loc[sub_idx, score].values
            r = signed_rc.loc[sub_idx, score].values
            if len(f) < 3 or np.std(f) == 0 or np.std(r) == 0:
                pearson_r = np.nan; spearman_r = np.nan
            else:
                pearson_r = float(pearsonr(f, r)[0])
                spearman_r = float(spearmanr(f, r)[0])
            rmse = float(np.sqrt(np.mean((f - r) ** 2)))
            # Relative RMSE: RMSE / max(stdev(f), stdev(r)) — dimensionless.
            denom = max(float(np.std(f)), float(np.std(r)), 1e-12)
            rmse_rel = rmse / denom
            var_records.append({
                "subset": subset, "score": score,
                "n_variants": len(f),
                "pearson_r": pearson_r,
                "spearman_r": spearman_r,
                "rmse": rmse,
                "rmse_relative_to_stdev": rmse_rel,
            })
    variability = pd.DataFrame(var_records)
    variability.to_parquet(out_dir / "iter4_fwd_rc_variability.parquet", index=False)

    # ---- (C) Paired McNemar: AVG vs FWD, AVG vs RC, FWD vs RC ----
    paired_records = []
    for subset, sub_idx in merged.groupby("subset").groups.items():
        sub = merged.loc[sub_idx]
        for score in final_names:
            f = signed_fwd.loc[sub_idx, score]
            r = signed_rc.loc[sub_idx, score]
            a = signed_avg.loc[sub_idx, score]
            for (cmp_name, sa, sb) in [
                ("AVG_vs_FWD", a, f),
                ("AVG_vs_RC",  a, r),
                ("FWD_vs_RC",  f, r),
            ]:
                res = paired_score_comparison(
                    label=sub["label"], score_a=sa, score_b=sb,
                    match_group=sub["match_group"], alternative="two-sided",
                )
                paired_records.append({
                    "subset": subset, "score": score, "comparison": cmp_name,
                    **res,
                })
    paired = pd.DataFrame(paired_records)
    paired["q_value"] = bh_adjust(paired["p_value"].values)
    paired.to_parquet(out_dir / "iter4_paired_strand_modes.parquet", index=False)

    # ---- Reports ----
    print("\n=== Best (mode, score) per subset, by PairwiseAccuracy (q<0.05 only) ===")
    best = solo[solo["q_value"] < 0.05].sort_values("value", ascending=False).groupby("subset").head(3)
    print(best.sort_values(["subset", "value"], ascending=[True, False])[
        ["subset", "mode", "score", "value", "n_pairs", "p_value", "q_value"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Per-subset summary: best AVG, best FWD, best RC.
    print("\n=== Mode-by-subset best score ===")
    rows = []
    for subset in sorted(solo["subset"].unique()):
        for mode in ("FWD", "RC", "AVG"):
            s = solo[(solo["subset"] == subset) & (solo["mode"] == mode)]
            top = s.sort_values("value", ascending=False).iloc[0]
            rows.append({
                "subset": subset, "mode": mode, "best_score": top["score"],
                "value": top["value"], "q": top["q_value"], "n_pairs": int(top["n_pairs"]),
            })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Variability: how correlated are FWD and RC?
    print("\n=== FWD vs RC variability — median Pearson / Spearman across the 30 scores, per subset ===")
    var_summary = variability.groupby("subset").agg(
        median_pearson=("pearson_r", "median"),
        median_spearman=("spearman_r", "median"),
        median_rmse_rel=("rmse_relative_to_stdev", "median"),
        n_variants=("n_variants", "first"),
    ).reset_index()
    print(var_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Scores with the LEAST agreement between FWD and RC (lowest Pearson).
    print("\n=== Most strand-asymmetric (score, subset) pairs (lowest Pearson r FWD vs RC) ===")
    print(variability.sort_values("pearson_r").head(15)[
        ["subset", "score", "n_variants", "pearson_r", "spearman_r", "rmse_relative_to_stdev"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Most strand-symmetric (highest Pearson r FWD vs RC) ===")
    print(variability.sort_values("pearson_r", ascending=False).head(10)[
        ["subset", "score", "n_variants", "pearson_r", "spearman_r", "rmse_relative_to_stdev"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Paired: AVG vs FWD/RC — does averaging help?
    print("\n=== Paired McNemar summary: AVG vs FWD, AVG vs RC, FWD vs RC (counts of q<0.05 wins) ===")
    paired_summary = paired.groupby("comparison").apply(
        lambda g: pd.Series({
            "n_cells": len(g),
            "n_a_wins_sig": int(((g["q_value"] < 0.05) & (g["value"] > 0.5)).sum()),
            "n_b_wins_sig": int(((g["q_value"] < 0.05) & (g["value"] < 0.5)).sum()),
            "mean_value": float(g["value"].mean()),
            "median_value": float(g["value"].median()),
        }), include_groups=False
    ).reset_index()
    print(paired_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
