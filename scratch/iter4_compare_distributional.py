"""Distributional + non-scale-invariant FWD-vs-RC diagnostics, per (model, subset).

Goal: distinguish two failure modes that both produce low FWD↔RC Pearson:
  (A) Model is uninformative on this subset → LLR concentrated near 0 in both
      strands → Pearson dominated by noise, low MSE (small spread).
  (B) Model is informative but strands disagree on magnitudes → LLR has
      meaningful spread, low Pearson, high MSE between strands.

For each (model, subset) we report:
  - Distributional shape of LLR (FWD and RC separately):
    n, mean, std, median |LLR|, skew, excess kurtosis, fraction |LLR| < 0.5
  - Non-scale-invariant strand-disagreement metric:
    MSE(FWD - RC), RMSE, MSE / Var(FWD) (≈ 2*(1-r) for matched-variance draws)
  - Pearson for reference

Also marks each row as "target" or "non-target" based on each model's intended
training-domain subsets, so we can see whether the (A)-vs-(B) pattern lines
up with target-domain expertise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, pearsonr, skew

from bolinas.zeroshot_vep.scores import SCORE_NAMES, apply_score_directions


MODEL_SPECS = [
    ("exp55-mammals", 256),
    ("exp58-mammals", 256),
    ("exp59-mammals", 256),
    ("exp136-proj_v30", 255),
]

# Per the iter-3 home-subset definitions and zeroshot_vep plan:
TARGET_SUBSETS: dict[str, set[str]] = {
    "exp55-mammals": {"tss_proximal", "5_prime_UTR_variant"},
    "exp58-mammals": {"missense_variant", "synonymous_variant", "splicing"},
    "exp59-mammals": {"3_prime_UTR_variant"},
    "exp136-proj_v30": {"distal"},
}


def load_pair(model: str, window: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fp = Path(f"scratch/iter1/scores/{model}__win{window}__mendelian_traits.parquet")
    rp = Path(f"scratch/iter4/iter4_rc_{model}__win{window}__mendelian_traits.parquet")
    fwd = pd.read_parquet(fp)
    rc = pd.read_parquet(rp)
    key = ["chrom", "pos", "ref", "alt"]
    rc_renamed = rc[key + SCORE_NAMES].rename(columns={c: f"{c}__rc" for c in SCORE_NAMES})
    m = fwd.merge(rc_renamed, on=key)
    f_signed = apply_score_directions(m[SCORE_NAMES])
    rc_only = m[[f"{c}__rc" for c in SCORE_NAMES]].rename(columns=lambda c: c.replace("__rc", ""))
    r_signed = apply_score_directions(rc_only)
    return m, f_signed, r_signed


def dist_stats(arr: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_median_abs": float(np.median(np.abs(arr))),
        f"{prefix}_skew": float(skew(arr)),
        f"{prefix}_excess_kurtosis": float(kurtosis(arr)),
        f"{prefix}_frac_small": float(np.mean(np.abs(arr) < 0.5)),
    }


def main(score: str = "llr") -> int:
    rows = []
    for model, window in MODEL_SPECS:
        try:
            m, f, r = load_pair(model, window)
        except FileNotFoundError as e:
            print(f"[skip] {model}: {e}", file=sys.stderr)
            continue
        # Use raw LLR (sign-direction-applied via apply_score_directions:
        # SCORE_DIRECTIONS['llr'] = +1, so signed_llr == raw_llr).
        for subset, sub_idx in m.groupby("subset").groups.items():
            f_arr = f.loc[sub_idx, score].values.astype(np.float64)
            r_arr = r.loc[sub_idx, score].values.astype(np.float64)
            n = len(f_arr)
            row = {
                "model": model, "subset": subset, "n": n,
                "is_target": subset in TARGET_SUBSETS.get(model, set()),
                **dist_stats(f_arr, "fwd"),
                **dist_stats(r_arr, "rc"),
                "mse": float(np.mean((f_arr - r_arr) ** 2)),
                "rmse": float(np.sqrt(np.mean((f_arr - r_arr) ** 2))),
                "mse_over_var_fwd": float(np.mean((f_arr - r_arr) ** 2) / max(np.var(f_arr), 1e-12)),
                "pearson": float(pearsonr(f_arr, r_arr)[0]) if np.std(f_arr) > 0 and np.std(r_arr) > 0 else np.nan,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(Path("scratch/iter4/iter4_fwd_rc_distributional.parquet"), index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)

    print(f"=== Distributional FWD-vs-RC stats per (model, subset), score={score} ===\n")
    print("Columns:")
    print("  fwd_std, rc_std   — spread of LLR per strand. Low std = uninformative.")
    print("  *_frac_small      — fraction of variants with |LLR| < 0.5 (noise threshold).")
    print("  *_median_abs      — median |LLR|. Low = uninformative.")
    print("  mse, rmse         — strand disagreement (non-scale-invariant).")
    print("  mse_over_var_fwd  — relative to FWD variance. 0 = identical, ~2 = uncorrelated noise.")
    print("  pearson           — for reference (scale-invariant, can be misleading near noise).")
    print()

    # Compact table per model
    for model, _ in MODEL_SPECS:
        sub = df[df["model"] == model].copy()
        sub = sub.sort_values(["is_target", "subset"], ascending=[False, True])
        cols = ["subset", "is_target", "n",
                "fwd_std", "fwd_median_abs", "fwd_frac_small",
                "rc_std", "rc_median_abs", "rc_frac_small",
                "rmse", "mse_over_var_fwd", "pearson"]
        print(f"--- {model} ---")
        print(sub[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print()

    # Cross-model summary on TARGET subsets specifically
    print("=== Summary: target-subset rows only (each row = (model, target_subset)) ===")
    tgt = df[df["is_target"]].sort_values(["model", "subset"])
    cols = ["model", "subset", "n", "fwd_std", "fwd_median_abs", "rc_std", "rc_median_abs",
            "rmse", "mse_over_var_fwd", "pearson"]
    print(tgt[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== Summary: non-target rows averaged per model (showing collapse pattern) ===")
    nt = df[~df["is_target"]].groupby("model").agg(
        n_subsets=("subset", "count"),
        fwd_std_median=("fwd_std", "median"),
        fwd_frac_small_median=("fwd_frac_small", "median"),
        rc_std_median=("rc_std", "median"),
        rc_frac_small_median=("rc_frac_small", "median"),
        rmse_median=("rmse", "median"),
        mse_over_var_median=("mse_over_var_fwd", "median"),
        pearson_median=("pearson", "median"),
    ).reset_index()
    print(nt.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="llr", help="Score column name to analyze (default: llr)")
    args = ap.parse_args()
    sys.exit(main(args.score))
