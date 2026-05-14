"""3-way paired comparison: LLR-default vs `down_jsd_mean` vs `mean_rank(LLR, L2)`.

`mean_rank(LLR, L2)` is reproduced from the iter-5 ensemble: within-subset
average rank of LLR-default and `embed_l2_flat_last` on FWD+RC AVG signed
scores. Higher rank → more pathogenic.

Paired McNemar (two-sided) at Global level. BH within dataset (3 pairwise
tests each).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bolinas.evals.metrics import paired_score_comparison, pairwise_accuracy

import importlib.util
_spec = importlib.util.spec_from_file_location("iter6", "scratch/zeroshot_vep_iter6_nucdep_analyze.py")
_a = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_a)

CANDIDATES = ["LLR", "down_jsd_mean", "mean_rank"]


def _bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64); mask = ~np.isnan(p); out = np.full_like(p, np.nan)
    if mask.sum() == 0: return out
    v = p[mask]; order = np.argsort(v); n = len(v); ranks = np.arange(1, n + 1)
    q_sorted = v[order] * n / ranks
    q_monotonic = np.minimum.accumulate(q_sorted[::-1])[::-1]
    out_mask = np.empty(n, dtype=np.float64); out_mask[order] = np.minimum(q_monotonic, 1.0)
    out[mask] = out_mask
    return out


def _fmt_p(p):
    return "—" if np.isnan(p) else (f"{p:.1e}" if p < 1e-4 else f"{p:.4f}")


def _build_with_meanrank(ds: str) -> pd.DataFrame:
    """Combined baseline (LLR, L2) + jsd_mean + within-subset mean_rank(LLR, L2)."""
    df = _a._build_combined(ds)
    llr_name = _a.LLR_BY_DATASET[ds]
    g = df.groupby("subset", sort=False)
    r_llr = g[llr_name].rank(method="average")
    r_emb = g[_a.EMBED].rank(method="average")
    df["mean_rank"] = (r_llr + r_emb) / 2.0
    # Pick out the three candidates with stable names: LLR (renamed to a constant
    # column), down_jsd_mean (already present), mean_rank.
    df["LLR"] = df[llr_name].values
    return df


def main() -> int:
    out_dir = Path("scratch/iter6")
    rows_global = []
    rows_paired = []
    persubset_rows = []

    for ds in _a.DATASETS:
        df = _build_with_meanrank(ds)
        # Global PA per candidate
        for cand in CANDIDATES:
            r = pairwise_accuracy(df["label"], df[cand], df["match_group"], alternative="greater")
            rows_global.append({"dataset": ds, "candidate": cand,
                                "PA": r["value"], "SE": r["se"], "n_pairs": r["n_pairs"]})
        # 3 pairwise paired tests
        pairs = [
            ("down_jsd_mean", "LLR"),
            ("mean_rank", "LLR"),
            ("mean_rank", "down_jsd_mean"),
        ]
        for a, b in pairs:
            r = paired_score_comparison(
                df["label"], df[a], df[b], df["match_group"], alternative="two-sided"
            )
            rows_paired.append({
                "dataset": ds, "A": a, "B": b,
                "PA_A": pairwise_accuracy(df["label"], df[a], df["match_group"], alternative="greater")["value"],
                "PA_B": pairwise_accuracy(df["label"], df[b], df["match_group"], alternative="greater")["value"],
                "delta": pairwise_accuracy(df["label"], df[a], df["match_group"], alternative="greater")["value"]
                       - pairwise_accuracy(df["label"], df[b], df["match_group"], alternative="greater")["value"],
                "n_pairs": r["n_pairs"],
                "n_A_wins": r["n_a_wins"], "n_B_wins": r["n_b_wins"],
                "n_concordant": r["n_concordant"], "n_half": r["n_half"],
                "p_value": r["p_value"],
            })
        # Per-subset PA
        for subset, sub in df.groupby("subset", sort=False):
            row = {"dataset": ds, "subset": subset, "n_pairs": sub["match_group"].nunique()}
            for cand in CANDIDATES:
                row[f"PA_{cand}"] = pairwise_accuracy(
                    sub["label"], sub[cand], sub["match_group"], alternative="greater"
                )["value"]
            persubset_rows.append(row)

    glob = pd.DataFrame(rows_global)
    paired = pd.DataFrame(rows_paired)
    paired["q_value"] = np.nan
    for ds in _a.DATASETS:
        idx = paired["dataset"] == ds
        paired.loc[idx, "q_value"] = _bh_adjust(paired.loc[idx, "p_value"].values)
    paired.to_parquet(out_dir / "iter6_3way_paired.parquet", index=False)
    persubset = pd.DataFrame(persubset_rows)
    persubset.to_parquet(out_dir / "iter6_3way_persubset.parquet", index=False)

    lines = [
        "🤖 **Iter-6 — 3-way: LLR-default vs `down_jsd_mean` vs `mean_rank(LLR, L2)` ensemble.**",
        "",
        "All FWD+RC AVG, exp166-p1B. `mean_rank` = within-subset average rank of LLR-default (`minus_llr` for mendelian, `abs_llr` for complex/eqtl) and `embed_l2_flat_last` — the iter-5 standard ensemble. Paired McNemar (two-sided) at Global level, BH within dataset (3 tests each).",
        "",
        "## Global PA",
        "",
        "| dataset | LLR | down_jsd_mean | mean_rank(LLR, L2) |",
        "|---|---:|---:|---:|",
    ]
    for ds in _a.DATASETS:
        sub = glob[glob["dataset"] == ds].set_index("candidate")
        lines.append(
            f"| {ds} | {sub.loc['LLR','PA']:.4f} ± {sub.loc['LLR','SE']:.4f} | "
            f"{sub.loc['down_jsd_mean','PA']:.4f} ± {sub.loc['down_jsd_mean','SE']:.4f} | "
            f"{sub.loc['mean_rank','PA']:.4f} ± {sub.loc['mean_rank','SE']:.4f} |"
        )
    lines.append("")
    lines.append("## Pairwise paired tests (Global level)")
    lines.append("")
    lines.append("Δ = PA(A) − PA(B); positive favors A.")
    lines.append("")
    for ds in _a.DATASETS:
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| A vs B | PA(A) | PA(B) | Δ | n_pairs | n_A>B | n_B>A | p | q |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in paired[paired["dataset"] == ds].iterrows():
            sig = " ★" if r["q_value"] < 0.05 else ""
            lines.append(
                f"| `{r['A']}` vs `{r['B']}` | {r['PA_A']:.4f} | {r['PA_B']:.4f} | "
                f"{r['delta']:+.4f} | {int(r['n_pairs'])} | "
                f"{int(r['n_A_wins'])} | {int(r['n_B_wins'])} | "
                f"{_fmt_p(r['p_value'])} | {_fmt_p(r['q_value'])}{sig} |"
            )
        lines.append("")
    lines.append("★ = q < 0.05.")
    lines.append("")
    lines.append("## Per-subset PA")
    lines.append("")
    for ds in _a.DATASETS:
        sub = persubset[persubset["dataset"] == ds].sort_values("n_pairs", ascending=False)
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| subset | n_pairs | LLR | down_jsd_mean | mean_rank |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['subset']} | {int(r['n_pairs'])} | "
                f"{r['PA_LLR']:.4f} | {r['PA_down_jsd_mean']:.4f} | {r['PA_mean_rank']:.4f} |"
            )
        lines.append("")

    text = "\n".join(lines)
    (out_dir / "iter6_3way_comment.md").write_text(text)
    print(f"[iter6] wrote {out_dir / 'iter6_3way_comment.md'}")
    print("\n=== Global PA ===")
    print(glob.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print("\n=== Paired ===")
    print(paired[["dataset","A","B","PA_A","PA_B","delta","n_A_wins","n_B_wins","p_value","q_value"]].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
