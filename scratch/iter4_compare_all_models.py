"""Joint-LLR Pearson(FWD, RC) across all 4 mendelian-evaluated models.

Models:
- exp55-mammals (promoter, 256 bp, no BOS)
- exp58-mammals (CDS, 256 bp, no BOS)
- exp59-mammals (3'UTR, 256 bp, no BOS)
- exp136-proj_v30 (enhancer, 255 bp + BOS)

For each model, compute the pooled Pearson(llr_fwd, llr_rc) across all 9820
mendelian variants. If the user's expectation "trained with RC augmentation
→ joint-LLR should be ~symmetric" is correct, we'd see Pearson near 1.
Lower Pearson means either (a) RC augmentation wasn't applied / was weak,
or (b) there's a subtle bug that affects different models differently.

Reports global Pearson per (model, score) AND per-subset for LLR family.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from bolinas.zeroshot_vep.scores import SCORE_NAMES, apply_score_directions


MODEL_SPECS = [
    ("exp55-mammals", 256),
    ("exp58-mammals", 256),
    ("exp59-mammals", 256),
    ("exp136-proj_v30", 255),
]


def fwd_path(model: str, window: int) -> Path:
    return Path(f"scratch/iter1/scores/{model}__win{window}__mendelian_traits.parquet")


def rc_path(model: str, window: int) -> Path:
    return Path(f"scratch/iter4/iter4_rc_{model}__win{window}__mendelian_traits.parquet")


def compute_pooled(model: str, window: int) -> pd.DataFrame:
    fp, rp = fwd_path(model, window), rc_path(model, window)
    if not rp.exists():
        print(f"[skip] {model}: {rp} not found yet", file=sys.stderr)
        return pd.DataFrame()
    fwd = pd.read_parquet(fp)
    rc = pd.read_parquet(rp)
    key = ["chrom", "pos", "ref", "alt"]
    rc_renamed = rc[key + SCORE_NAMES].rename(columns={c: f"{c}__rc" for c in SCORE_NAMES})
    m = fwd.merge(rc_renamed, on=key)
    f_signed = apply_score_directions(m[SCORE_NAMES])
    rc_only = m[[f"{c}__rc" for c in SCORE_NAMES]].rename(columns=lambda c: c.replace("__rc", ""))
    r_signed = apply_score_directions(rc_only)
    rows = []
    for col in f_signed.columns:
        f, r = f_signed[col].values, r_signed[col].values
        pr = float(pearsonr(f, r)[0]) if np.std(f) > 0 and np.std(r) > 0 else np.nan
        sr = float(spearmanr(f, r)[0]) if np.std(f) > 0 and np.std(r) > 0 else np.nan
        rows.append({"model": model, "window": window, "score": col, "pearson": pr, "spearman": sr})
    return pd.DataFrame(rows)


def compute_per_subset_llr(model: str, window: int) -> pd.DataFrame:
    fp, rp = fwd_path(model, window), rc_path(model, window)
    if not rp.exists():
        return pd.DataFrame()
    fwd = pd.read_parquet(fp)
    rc = pd.read_parquet(rp)
    key = ["chrom", "pos", "ref", "alt"]
    rc_renamed = rc[key + SCORE_NAMES].rename(columns={c: f"{c}__rc" for c in SCORE_NAMES})
    m = fwd.merge(rc_renamed, on=key)
    f_signed = apply_score_directions(m[SCORE_NAMES])
    rc_only = m[[f"{c}__rc" for c in SCORE_NAMES]].rename(columns=lambda c: c.replace("__rc", ""))
    r_signed = apply_score_directions(rc_only)
    rows = []
    for sub, sub_idx in m.groupby("subset").groups.items():
        for score in ("minus_llr", "minus_entropy", "logp_ref"):
            f = f_signed.loc[sub_idx, score].values
            r = r_signed.loc[sub_idx, score].values
            if len(f) < 3 or np.std(f) == 0 or np.std(r) == 0:
                pr = np.nan
            else:
                pr = float(pearsonr(f, r)[0])
            rows.append({"model": model, "subset": sub, "score": score, "n": len(f), "pearson": pr})
    return pd.DataFrame(rows)


def main() -> int:
    out_dir = Path("scratch/iter4")

    # Pooled per-score table
    pooled_pieces = [compute_pooled(m, w) for m, w in MODEL_SPECS]
    pooled = pd.concat([p for p in pooled_pieces if len(p) > 0])
    if len(pooled) == 0:
        print("No RC parquets yet. Run scout first.", file=sys.stderr)
        return 1
    pooled.to_parquet(out_dir / "iter4_joint_llr_pearson_all_models.parquet", index=False)

    pivot = pooled.pivot_table(index="score", columns="model", values="pearson").reset_index()
    cols_order = ["score"] + [m for m, _ in MODEL_SPECS if m in pivot.columns]
    pivot = pivot[cols_order]
    print("=== Joint-score FWD↔RC pooled Pearson across all 4 models (sorted by exp55) ===\n")
    if "exp55-mammals" in pivot.columns:
        pivot = pivot.sort_values("exp55-mammals", ascending=False)
    print(pivot.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Focus: LLR family
    print("\n=== LLR family pooled Pearson, all models ===")
    fam = ["llr", "minus_llr", "abs_llr", "logp_ref", "minus_logp_alt", "minus_entropy"]
    print(pivot[pivot["score"].isin(fam)].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Per-subset for minus_llr
    print("\n=== Per-subset Pearson(llr_fwd, llr_rc), all models ===")
    per_sub_pieces = [compute_per_subset_llr(m, w) for m, w in MODEL_SPECS]
    per_sub = pd.concat([p for p in per_sub_pieces if len(p) > 0])
    per_sub.to_parquet(out_dir / "iter4_joint_llr_pearson_per_subset_all_models.parquet", index=False)

    sub_pivot = per_sub[per_sub["score"] == "minus_llr"].pivot_table(
        index="subset", columns="model", values="pearson"
    ).reset_index()
    cols = ["subset"] + [m for m, _ in MODEL_SPECS if m in sub_pivot.columns]
    sub_pivot = sub_pivot[cols]
    print(sub_pivot.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
