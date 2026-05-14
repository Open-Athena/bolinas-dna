"""Iter-6 systematic comparison of the 8 nuc-dep scores.

Two questions:

Q-systematic-a) For each metric in {jsd, l1, l2, linf}, is the `_mean` aggregation
   systematically better than `_max`? 4 paired McNemar tests per dataset (one
   per metric), BH within dataset.

Q-systematic-b) Among the 8 nuc-dep scores, is there a single one that
   "systematically" wins — i.e., not significantly worse than the best in
   ALL 3 datasets? For each dataset, paired McNemar of each score vs the
   per-dataset Global-PA-best score (7 tests per dataset), BH within dataset.
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
DATASETS = _a.DATASETS
NUCDEP_SCORES = _a.NUCDEP_SCORES
N_MIN_MACRO = _a.N_MIN_MACRO


def _bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    mask = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if mask.sum() == 0:
        return out
    v = p[mask]
    order = np.argsort(v)
    n = len(v)
    ranks = np.arange(1, n + 1)
    q_sorted = v[order] * n / ranks
    q_monotonic = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_clipped = np.minimum(q_monotonic, 1.0)
    out_mask = np.empty(n, dtype=np.float64)
    out_mask[order] = q_clipped
    out[mask] = out_mask
    return out


def _build(ds: str) -> pd.DataFrame:
    """Load FWD+RC nuc-dep parquets and return signed AVG (sign=+1 for all 8)."""
    fwd = pd.read_parquet(f"scratch/iter6/iter6_nucdep_fwd_exp166-p1B__win255__{ds}.parquet")
    rc = pd.read_parquet(f"scratch/iter6/iter6_nucdep_rc_exp166-p1B__win255__{ds}.parquet")
    KEY = ["chrom", "pos", "ref", "alt"]
    rc_renamed = rc[KEY + NUCDEP_SCORES].rename(columns={c: f"{c}__rc" for c in NUCDEP_SCORES})
    df = fwd.merge(rc_renamed, on=KEY, how="inner")
    assert len(df) == len(fwd) == len(rc)
    avg = {c: (df[c].values + df[f"{c}__rc"].values) / 2.0 for c in NUCDEP_SCORES}
    out = pd.DataFrame(avg)
    out["subset"] = df["subset"].values
    out["label"] = df["label"].values
    out["match_group"] = df["match_group"].values
    return out


def _pa(df, col):
    return pairwise_accuracy(df["label"], df[col], df["match_group"], alternative="greater")["value"]


def _paired_global(df, a, b):
    return paired_score_comparison(df["label"], df[a], df[b], df["match_group"], alternative="two-sided")


def _fmt_p(p):
    return "—" if np.isnan(p) else (f"{p:.1e}" if p < 1e-4 else f"{p:.4f}")


def main() -> int:
    out_dir = Path("scratch/iter6")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Q-a: mean vs max paired test per metric, per dataset -----
    meanmax_rows = []
    for ds in DATASETS:
        df = _build(ds)
        for metric in ("jsd", "l1", "l2", "linf"):
            a = f"down_{metric}_mean"
            b = f"down_{metric}_max"
            r = _paired_global(df, a, b)
            meanmax_rows.append({
                "dataset": ds, "metric": metric,
                "mean_PA": _pa(df, a), "max_PA": _pa(df, b),
                "delta": _pa(df, a) - _pa(df, b),
                "n_pairs": r["n_pairs"],
                "n_mean_wins": r["n_a_wins"],
                "n_max_wins": r["n_b_wins"],
                "p_value": r["p_value"],
            })
    mm = pd.DataFrame(meanmax_rows)
    mm["q_value"] = np.nan
    for ds in DATASETS:
        idx = mm["dataset"] == ds
        mm.loc[idx, "q_value"] = _bh_adjust(mm.loc[idx, "p_value"].values)
    mm.to_parquet(out_dir / "iter6_systematic_meanvsmax.parquet", index=False)

    # ----- Q-b: each score vs per-dataset best (by Global PA) -----
    vs_best_rows = []
    best_per_ds: dict[str, str] = {}
    for ds in DATASETS:
        df = _build(ds)
        pas = {c: _pa(df, c) for c in NUCDEP_SCORES}
        best = max(pas.items(), key=lambda kv: kv[1])[0]
        best_per_ds[ds] = best
        for c in NUCDEP_SCORES:
            if c == best:
                continue
            r = _paired_global(df, c, best)
            vs_best_rows.append({
                "dataset": ds, "candidate": c, "best": best,
                "PA_cand": _pa(df, c), "PA_best": pas[best],
                "delta": _pa(df, c) - pas[best],
                "n_pairs": r["n_pairs"],
                "n_cand_wins": r["n_a_wins"],
                "n_best_wins": r["n_b_wins"],
                "p_value": r["p_value"],
            })
    vb = pd.DataFrame(vs_best_rows)
    vb["q_value"] = np.nan
    for ds in DATASETS:
        idx = vb["dataset"] == ds
        vb.loc[idx, "q_value"] = _bh_adjust(vb.loc[idx, "p_value"].values)
    vb.to_parquet(out_dir / "iter6_systematic_vsbest.parquet", index=False)

    _write_comment(mm, vb, best_per_ds, out_dir)
    return 0


def _write_comment(mm: pd.DataFrame, vb: pd.DataFrame,
                   best_per_ds: dict, out_dir: Path) -> None:
    lines = [
        "🤖 **Q (systematic) — is any specific nuc-dep approach systematically better?**",
        "",
        "Two paired-McNemar analyses on the iter-6 exp166-p1B FWD+RC AVG nuc-dep scores:",
        "",
        "## (a) `_mean` vs `_max` aggregation — paired test per metric per dataset",
        "",
        "For each metric (`jsd`/`l1`/`l2`/`linf`), test whether the `_mean` aggregation across positions differs from `_max`. Δ = PA(mean) − PA(max); BH within dataset (4 tests each).",
        "",
        "| dataset | metric | mean PA | max PA | Δ | n_pairs | n_mean>max | n_max>mean | p | q |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in mm.iterrows():
        sig = " ★" if r["q_value"] < 0.05 else ""
        lines.append(
            f"| {r['dataset']} | `{r['metric']}` | {r['mean_PA']:.4f} | {r['max_PA']:.4f} | "
            f"{r['delta']:+.4f} | {int(r['n_pairs'])} | {int(r['n_mean_wins'])} | {int(r['n_max_wins'])} | "
            f"{_fmt_p(r['p_value'])} | {_fmt_p(r['q_value'])}{sig} |"
        )
    lines.append("")
    lines.append("★ = q < 0.05.")
    lines.append("")

    # Per-dataset summary: mean wins systematically?
    pos_count_per_metric = {}
    for metric in ("jsd", "l1", "l2", "linf"):
        rows = mm[mm["metric"] == metric]
        pos = (rows["delta"] > 0).sum()
        sig = (rows["q_value"] < 0.05).sum()
        pos_count_per_metric[metric] = (pos, sig, len(rows))
    lines.append("**Summary (a)**: across all 12 (metric × dataset) cells, " +
                 f"mean wins in {(mm['delta'] > 0).sum()}/12 cells; "
                 f"{(mm['q_value'] < 0.05).sum()} are significant after BH.")
    lines.append("")

    # ----- (b) -----
    lines.append("## (b) Each score vs the per-dataset Global-PA-best nuc-dep score")
    lines.append("")
    lines.append("Per-dataset best (by Global PA on FWD+RC AVG):")
    for ds in DATASETS:
        lines.append(f"- **{ds}**: `{best_per_ds[ds]}`")
    lines.append("")
    lines.append("Paired McNemar of each other nuc-dep vs the per-dataset best; BH within dataset (7 tests each). Δ = PA(candidate) − PA(best).")
    lines.append("")
    for ds in DATASETS:
        lines.append(f"### {ds}  (best = `{best_per_ds[ds]}`)")
        lines.append("")
        sub = vb[vb["dataset"] == ds].sort_values("delta", ascending=False)
        lines.append("| candidate | PA(cand) | Δ vs best | n_pairs | n_cand>best | n_best>cand | p | q |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            sig = " ★" if r["q_value"] < 0.05 else ""
            lines.append(
                f"| `{r['candidate']}` | {r['PA_cand']:.4f} | {r['delta']:+.4f} | "
                f"{int(r['n_pairs'])} | {int(r['n_cand_wins'])} | {int(r['n_best_wins'])} | "
                f"{_fmt_p(r['p_value'])} | {_fmt_p(r['q_value'])}{sig} |"
            )
        lines.append("")

    # Cross-dataset summary: which scores are "tied with best" in every dataset
    # (i.e., never significantly worse)?
    not_worse = {}
    for c in NUCDEP_SCORES:
        ds_status = []
        for ds in DATASETS:
            if c == best_per_ds[ds]:
                ds_status.append("BEST")
            else:
                row = vb[(vb["dataset"] == ds) & (vb["candidate"] == c)]
                if row.empty: continue
                if row["q_value"].iloc[0] >= 0.05:
                    ds_status.append("tied")
                else:
                    ds_status.append("WORSE")
        not_worse[c] = ds_status
    lines.append("**Cross-dataset summary**: status per nuc-dep score in each dataset (BEST = per-dataset Global winner; tied = not significantly worse than best at q<0.05; WORSE = significantly worse).")
    lines.append("")
    lines.append("| score | " + " | ".join(DATASETS) + " |")
    lines.append("|---|" + "---|" * len(DATASETS))
    for c in NUCDEP_SCORES:
        statuses = []
        for i, _ in enumerate(DATASETS):
            s = not_worse[c][i]
            statuses.append(s)
        lines.append(f"| `{c}` | " + " | ".join(statuses) + " |")
    lines.append("")
    lines.append("A score is **systematically a safe choice** if it's BEST or tied in all 3 datasets — i.e., never significantly worse than the per-dataset winner.")

    text = "\n".join(lines)
    (out_dir / "iter6_systematic_comment.md").write_text(text)
    print(f"[iter6] wrote {out_dir / 'iter6_systematic_comment.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
