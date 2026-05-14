"""Iter-5 per-subset paired McNemar tests for Q1 and Q2.

Same comparisons as ``zeroshot_vep_iter5_paired_tests.py`` but stratified by
``subset`` instead of pooled at Global. This gives a heat-map of where the
differences live within each dataset.

Q1: 3 comparisons (L2 vs LLR, best-ens vs LLR, best-ens vs L2) per
    (dataset, subset). BH within (dataset, comparison) — same-comparison
    across subsets.

Q2: 6 comparisons (each non-mean_rank ensemble vs mean_rank) per
    (dataset, subset). BH within (dataset, candidate).

Best ensemble per dataset is the same as iter5_paired_tests.py picks (by
Global PA): geomean_rank / max_rank / rrf_k60.

Subsets included: only those with n_pairs >= N_MIN_FOR_TEST so the paired
test has at least nominal power. Default 5 (well below leaderboard's 30 —
this is just to skip degenerate per-pair cases like complex_traits splicing
n_pairs=1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bolinas.evals.metrics import paired_score_comparison, pairwise_accuracy

# Reuse score-table builder from the global paired-test script.
import importlib.util
_spec = importlib.util.spec_from_file_location("iter5_pt", "scratch/zeroshot_vep_iter5_paired_tests.py")
_pt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_pt)
DATASETS = _pt.DATASETS
LLR_BY_DATASET = _pt.LLR_BY_DATASET
EMBED = _pt.EMBED
ENSEMBLES = _pt.ENSEMBLES
ENSEMBLES_VS_MEAN = _pt.ENSEMBLES_VS_MEAN
_bh = _pt._bh_adjust

N_MIN_FOR_TEST: int = 5


def _paired(sub: pd.DataFrame, col_a: str, col_b: str) -> dict:
    return paired_score_comparison(
        label=sub["label"], score_a=sub[col_a], score_b=sub[col_b],
        match_group=sub["match_group"], alternative="two-sided",
    )


def _pa(sub: pd.DataFrame, col: str) -> float:
    return pairwise_accuracy(sub["label"], sub[col], sub["match_group"],
                              alternative="greater")["value"]


def _best_ens_by_global(ds: str) -> str:
    scored = _pt._build_score_table(ds)
    pas = {c: _pa(scored, c) for c in ENSEMBLES}
    return max(pas.items(), key=lambda kv: kv[1])[0]


def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "—"
    return f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"


def main() -> int:
    out_dir = Path("scratch/iter5")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------- Q1 per-subset ---------
    q1_rows = []
    best_ens_per_dataset: dict[str, str] = {}
    for ds in DATASETS:
        scored = _pt._build_score_table(ds)
        llr_name = LLR_BY_DATASET[ds]
        best_ens = _best_ens_by_global(ds)
        best_ens_per_dataset[ds] = best_ens
        comparisons = [
            ("L2 vs LLR", EMBED, llr_name),
            (f"{best_ens} vs LLR", best_ens, llr_name),
            (f"{best_ens} vs L2", best_ens, EMBED),
        ]
        for subset, sub in scored.groupby("subset", sort=False):
            n_variants = len(sub)
            n_pairs = sub["match_group"].nunique()
            if n_pairs < N_MIN_FOR_TEST:
                continue
            for cmp_name, col_a, col_b in comparisons:
                r = _paired(sub, col_a, col_b)
                q1_rows.append({
                    "dataset": ds, "subset": subset, "n_pairs": n_pairs,
                    "comparison": cmp_name,
                    "PA_A": _pa(sub, col_a),
                    "PA_B": _pa(sub, col_b),
                    "delta": _pa(sub, col_a) - _pa(sub, col_b),
                    "n_a_wins": r["n_a_wins"], "n_b_wins": r["n_b_wins"],
                    "p_value": r["p_value"],
                })
    q1 = pd.DataFrame(q1_rows)
    # BH within (dataset, comparison) — same comparison across subsets is the
    # natural family.
    q1["q_value"] = np.nan
    for (ds, cmp_), g in q1.groupby(["dataset", "comparison"]):
        q1.loc[g.index, "q_value"] = _bh(g["p_value"].values)
    q1.to_parquet(out_dir / "iter5_q1_per_subset_paired.parquet", index=False)

    # --------- Q2 per-subset ---------
    q2_rows = []
    for ds in DATASETS:
        scored = _pt._build_score_table(ds)
        for subset, sub in scored.groupby("subset", sort=False):
            n_pairs = sub["match_group"].nunique()
            if n_pairs < N_MIN_FOR_TEST:
                continue
            pa_base = _pa(sub, "mean_rank")
            for cand in ENSEMBLES_VS_MEAN:
                r = _paired(sub, cand, "mean_rank")
                q2_rows.append({
                    "dataset": ds, "subset": subset, "n_pairs": n_pairs,
                    "candidate": cand,
                    "PA_cand": _pa(sub, cand),
                    "PA_base": pa_base,
                    "delta": _pa(sub, cand) - pa_base,
                    "n_cand_wins": r["n_a_wins"], "n_base_wins": r["n_b_wins"],
                    "p_value": r["p_value"],
                })
    q2 = pd.DataFrame(q2_rows)
    # BH within (dataset, candidate)
    q2["q_value"] = np.nan
    for (ds, cand), g in q2.groupby(["dataset", "candidate"]):
        q2.loc[g.index, "q_value"] = _bh(g["p_value"].values)
    q2.to_parquet(out_dir / "iter5_q2_per_subset_paired.parquet", index=False)

    # --------- Render comments ---------
    _write_q1_comment(q1, best_ens_per_dataset, out_dir)
    _write_q2_comment(q2, out_dir)

    return 0


def _write_q1_comment(q1: pd.DataFrame, best_ens: dict, out_dir: Path) -> None:
    lines = [
        "🤖 **Q1 per-subset McNemar — where do the LLR / L2 / ensemble differences live within each dataset?**",
        "",
        f"Per-subset paired sign test (two-sided), restricted to subsets with n_pairs ≥ {N_MIN_FOR_TEST}. "
        "BH-adjusted q within (dataset, comparison) — i.e. across the qualifying subsets per dataset, per comparison-pair. "
        "Δ = PA(A) − PA(B); positive favors A.",
        "",
    ]
    for ds in DATASETS:
        lines.append(f"### {ds}  (best ensemble = `{best_ens[ds]}`)")
        lines.append("")
        sub_df = q1[q1["dataset"] == ds].copy()
        # Explicit column order for natural reading: base-vs-base, then ensemble vs each base.
        be = best_ens[ds]
        comparisons = ["L2 vs LLR", f"{be} vs LLR", f"{be} vs L2"]
        # Sort subsets by n_pairs descending (biggest = most power → readable first).
        subset_order = (sub_df.drop_duplicates("subset")
                              .sort_values("n_pairs", ascending=False)["subset"]
                              .tolist())
        # Wide-format: rows = subset, cols = each comparison's Δ and p (compact)
        lines.append("| subset | n_pairs | " +
                     " | ".join(f"{c} Δ | p | q" for c in comparisons) + " |")
        sep = "|---|---:|"
        for _ in comparisons:
            sep += "---:|---:|---:|"
        lines.append(sep)
        for subset in subset_order:
            row_vals = [subset, str(int(sub_df[sub_df["subset"]==subset]["n_pairs"].iloc[0]))]
            for cmp_ in comparisons:
                cell = sub_df[(sub_df["subset"]==subset) & (sub_df["comparison"]==cmp_)]
                if cell.empty:
                    row_vals.extend(["—", "—", "—"])
                    continue
                d = cell["delta"].iloc[0]
                p = cell["p_value"].iloc[0]
                q = cell["q_value"].iloc[0]
                sig = " ★" if q < 0.05 else ""
                row_vals.extend([f"{d:+.4f}", _fmt_p(p), f"{_fmt_p(q)}{sig}"])
            lines.append("| " + " | ".join(row_vals) + " |")
        lines.append("")
    lines.append("★ = q < 0.05 (BH within dataset × comparison-pair).")

    text = "\n".join(lines)
    (out_dir / "iter5_q1_per_subset_comment.md").write_text(text)
    print(f"[iter5] wrote {out_dir / 'iter5_q1_per_subset_comment.md'}")


def _write_q2_comment(q2: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "🤖 **Q2 per-subset McNemar — where do the ensembles diverge from `mean_rank` within each dataset?**",
        "",
        f"Per-subset paired sign test (two-sided), restricted to subsets with n_pairs ≥ {N_MIN_FOR_TEST}. "
        "BH-adjusted q within (dataset, candidate) — across subsets per candidate-ensemble. "
        "Δ = PA(candidate) − PA(mean_rank); positive = candidate better.",
        "",
    ]
    for ds in DATASETS:
        lines.append(f"### {ds}")
        lines.append("")
        sub_df = q2[q2["dataset"] == ds].copy()
        cands = ENSEMBLES_VS_MEAN
        subset_order = (sub_df.drop_duplicates("subset")
                              .sort_values("n_pairs", ascending=False)["subset"]
                              .tolist())
        lines.append("| subset | n_pairs | " +
                     " | ".join(f"`{c}` Δ (p / q)" for c in cands) + " |")
        sep = "|---|---:|" + "---:|" * len(cands)
        lines.append(sep)
        for subset in subset_order:
            row_vals = [subset, str(int(sub_df[sub_df["subset"]==subset]["n_pairs"].iloc[0]))]
            for cand in cands:
                cell = sub_df[(sub_df["subset"]==subset) & (sub_df["candidate"]==cand)]
                if cell.empty:
                    row_vals.append("—")
                    continue
                d = cell["delta"].iloc[0]
                p = cell["p_value"].iloc[0]
                q = cell["q_value"].iloc[0]
                sig = " ★" if q < 0.05 else ""
                row_vals.append(f"{d:+.4f} ({_fmt_p(p)} / {_fmt_p(q)}){sig}")
            lines.append("| " + " | ".join(row_vals) + " |")
        lines.append("")
    lines.append("★ = q < 0.05 (BH within dataset × candidate).")

    text = "\n".join(lines)
    (out_dir / "iter5_q2_per_subset_comment.md").write_text(text)
    print(f"[iter5] wrote {out_dir / 'iter5_q2_per_subset_comment.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
