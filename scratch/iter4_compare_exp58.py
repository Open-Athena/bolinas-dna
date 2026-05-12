"""Quick analysis: joint-LLR Pearson FWD vs RC for exp58-mammals × mendelian.

Goal: compare exp58's joint-LLR strand symmetry to exp55's (computed in iter-4).

If both models have RC augmentation:
- Joint-LLR FWD↔RC Pearson is the right diagnostic for RC equivariance.
- Per-position logit symmetry (the bug-check) is NOT — different autoregressive
  factorizations give different per-position conditionals even with perfect
  joint-level RC equivariance.

Inputs (read from S3 or local):
- scratch/iter1/scores/exp58-mammals__win256__mendelian_traits.parquet  (FWD)
- exp58 RC parquet (downloaded from cluster to scratch/iter4/)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from bolinas.zeroshot_vep.scores import SCORE_DIRECTIONS, SCORE_NAMES, apply_score_directions


def compute_pooled_pearson(model: str, fwd_path: Path, rc_path: Path) -> pd.DataFrame:
    fwd = pd.read_parquet(fwd_path)
    rc = pd.read_parquet(rc_path)
    key = ["chrom", "pos", "ref", "alt"]
    rc_renamed = rc[key + SCORE_NAMES].rename(columns={c: f"{c}__rc" for c in SCORE_NAMES})
    m = fwd.merge(rc_renamed, on=key, how="inner")
    assert len(m) == len(fwd) == len(rc), f"merge mismatch for {model}"

    signed_fwd = apply_score_directions(m[SCORE_NAMES])
    rc_only = m[[f"{c}__rc" for c in SCORE_NAMES]].rename(columns=lambda c: c.replace("__rc", ""))
    signed_rc = apply_score_directions(rc_only)

    rows = []
    for col in signed_fwd.columns:
        f, r = signed_fwd[col].values, signed_rc[col].values
        pr = float(pearsonr(f, r)[0]) if np.std(f) > 0 and np.std(r) > 0 else np.nan
        sr = float(spearmanr(f, r)[0]) if np.std(f) > 0 and np.std(r) > 0 else np.nan
        rows.append({"model": model, "score": col, "pearson": pr, "spearman": sr})
    return pd.DataFrame(rows)


def main() -> int:
    out_dir = Path("scratch/iter4")

    # exp55 already computed from iter-4
    exp55 = compute_pooled_pearson(
        "exp55-mammals",
        Path("scratch/iter1/scores/exp55-mammals__win256__mendelian_traits.parquet"),
        Path("scratch/iter4/iter4_rc_exp55-mammals__win256__mendelian_traits.parquet"),
    )
    exp58 = compute_pooled_pearson(
        "exp58-mammals",
        Path("scratch/iter1/scores/exp58-mammals__win256__mendelian_traits.parquet"),
        Path("scratch/iter4/iter4_rc_exp58-mammals__win256__mendelian_traits.parquet"),
    )
    df = pd.concat([exp55, exp58])
    df.to_parquet(out_dir / "iter4_joint_llr_pearson_exp55_vs_exp58.parquet", index=False)

    # Pivot to side-by-side comparison
    print("=== Joint-score FWD↔RC pooled Pearson: exp55 vs exp58 (sorted by exp58 - exp55) ===\n")
    pivot = df.pivot(index="score", columns="model", values="pearson").reset_index()
    pivot["delta"] = pivot["exp58-mammals"] - pivot["exp55-mammals"]
    pivot = pivot.sort_values("exp58-mammals", ascending=False)
    print(pivot.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Focus: LLR family + a few diagnostic embeds
    print("\n=== Focus: LLR family ===")
    fam = ["llr", "minus_llr", "abs_llr", "logp_ref", "minus_logp_alt", "minus_entropy"]
    print(pivot[pivot["score"].isin(fam)].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
