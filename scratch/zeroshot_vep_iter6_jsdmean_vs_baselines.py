"""Pin `down_jsd_mean` (the systematic best from iter-6) and compare to the
LLR-default + embed_l2_flat_last baselines on all 3 datasets.

Paired McNemar (Global level) + Spearman correlation. BH within dataset (2
tests each).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bolinas.evals.metrics import paired_score_comparison, pairwise_accuracy

import importlib.util
_spec = importlib.util.spec_from_file_location("iter6", "scratch/zeroshot_vep_iter6_nucdep_analyze.py")
_a = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_a)

CANDIDATE = "down_jsd_mean"


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


def main() -> int:
    out_dir = Path("scratch/iter6")
    rows = []
    spearman_rows = []
    persubset_rows = []

    for ds in _a.DATASETS:
        df = _a._build_combined(ds)
        llr_name = _a.LLR_BY_DATASET[ds]
        for baseline, baseline_pretty in [(llr_name, f"`{llr_name}`"),
                                           (_a.EMBED, f"`{_a.EMBED}`")]:
            r = paired_score_comparison(
                df["label"], df[CANDIDATE], df[baseline], df["match_group"],
                alternative="two-sided",
            )
            rows.append({
                "dataset": ds, "candidate": CANDIDATE, "baseline": baseline,
                "PA_cand": _a._pa(df, CANDIDATE)["value"],
                "PA_base": _a._pa(df, baseline)["value"],
                "delta": _a._pa(df, CANDIDATE)["value"] - _a._pa(df, baseline)["value"],
                "n_pairs": r["n_pairs"], "n_cand_wins": r["n_a_wins"],
                "n_base_wins": r["n_b_wins"],
                "n_concordant": r["n_concordant"], "n_half": r["n_half"],
                "p_value": r["p_value"],
            })
        # Spearman
        for baseline in (llr_name, _a.EMBED):
            rho_global = float(spearmanr(df[CANDIDATE], df[baseline]).statistic)
            rhos_within = []
            for subset, sub in df.groupby("subset", sort=False):
                if len(sub) < 5: continue
                rho = spearmanr(sub[CANDIDATE], sub[baseline]).statistic
                if not np.isnan(rho): rhos_within.append(rho)
            spearman_rows.append({
                "dataset": ds, "baseline": baseline,
                "spearman_global": rho_global,
                "spearman_within": float(np.mean(rhos_within)) if rhos_within else float("nan"),
            })
        # Per-subset PA for the candidate vs the two baselines
        for subset, sub in df.groupby("subset", sort=False):
            persubset_rows.append({
                "dataset": ds, "subset": subset,
                "n_pairs": sub["match_group"].nunique(),
                f"PA_{CANDIDATE}": _a._pa(sub, CANDIDATE)["value"],
                f"PA_{llr_name}":  _a._pa(sub, llr_name)["value"],
                f"PA_{_a.EMBED}":  _a._pa(sub, _a.EMBED)["value"],
            })

    paired = pd.DataFrame(rows)
    paired["q_value"] = np.nan
    for ds in _a.DATASETS:
        idx = paired["dataset"] == ds
        paired.loc[idx, "q_value"] = _bh_adjust(paired.loc[idx, "p_value"].values)
    paired.to_parquet(out_dir / "iter6_jsdmean_vs_baselines.parquet", index=False)

    sp = pd.DataFrame(spearman_rows)
    sp.to_parquet(out_dir / "iter6_jsdmean_spearman.parquet", index=False)

    ps = pd.DataFrame(persubset_rows)

    # Render
    lines = [
        f"🤖 **Iter-6 — fix `{CANDIDATE}` (systematic best from previous comment) and compare to LLR / L2 baselines.**",
        "",
        "Paired McNemar (two-sided) at Global level, BH within dataset (2 tests each). Δ = PA(`down_jsd_mean`) − PA(baseline).",
        "",
        "## Global paired tests",
        "",
        "| dataset | baseline | PA(`down_jsd_mean`) | PA(baseline) | Δ | n_pairs | n_cand>base | n_base>cand | p | q |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in paired.iterrows():
        b_pretty = "`embed_l2_flat_last`" if r["baseline"] == _a.EMBED else f"`{r['baseline']}`"
        sig = " ★" if r["q_value"] < 0.05 else ""
        lines.append(
            f"| {r['dataset']} | {b_pretty} | {r['PA_cand']:.4f} | {r['PA_base']:.4f} | "
            f"{r['delta']:+.4f} | {int(r['n_pairs'])} | {int(r['n_cand_wins'])} | {int(r['n_base_wins'])} | "
            f"{_fmt_p(r['p_value'])} | {_fmt_p(r['q_value'])}{sig} |"
        )
    lines.append("")
    lines.append("★ = q < 0.05.")
    lines.append("")
    lines.append("## Spearman correlation (within-subset mean)")
    lines.append("")
    lines.append("| dataset | `down_jsd_mean` × LLR | `down_jsd_mean` × L2 |")
    lines.append("|---|---:|---:|")
    for ds in _a.DATASETS:
        llr_name = _a.LLR_BY_DATASET[ds]
        rho_llr = sp[(sp["dataset"] == ds) & (sp["baseline"] == llr_name)]["spearman_within"].iloc[0]
        rho_l2 = sp[(sp["dataset"] == ds) & (sp["baseline"] == _a.EMBED)]["spearman_within"].iloc[0]
        lines.append(f"| {ds} | {rho_llr:.3f} | {rho_l2:.3f} |")
    lines.append("")
    lines.append("## Per-subset PA")
    lines.append("")
    for ds in _a.DATASETS:
        llr_name = _a.LLR_BY_DATASET[ds]
        sub = ps[ps["dataset"] == ds].sort_values("n_pairs", ascending=False)
        lines.append(f"### {ds}")
        lines.append("")
        lines.append(f"| subset | n_pairs | `down_jsd_mean` | `{llr_name}` | `embed_l2_flat_last` |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['subset']} | {int(r['n_pairs'])} | "
                f"{r[f'PA_{CANDIDATE}']:.4f} | {r[f'PA_{llr_name}']:.4f} | {r[f'PA_{_a.EMBED}']:.4f} |"
            )
        lines.append("")

    text = "\n".join(lines)
    (out_dir / "iter6_jsdmean_vs_baselines_comment.md").write_text(text)
    print(f"[iter6] wrote {out_dir / 'iter6_jsdmean_vs_baselines_comment.md'}")
    print("\n=== Global paired ===")
    print(paired.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
