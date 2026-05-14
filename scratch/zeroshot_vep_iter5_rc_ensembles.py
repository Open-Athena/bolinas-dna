"""Iter-5 follow-up to #175: exp166-p1B FWD+RC AVG — LLR + embed_l2_flat_last + ensembles.

For each of 3 datasets (mendelian_traits, complex_traits, eqtl):
  - Load FWD + RC per-variant score parquets (RC for complex/eqtl comes from
    iter-5 GPU run; mendelian RC is from iter-4)
  - Sign-normalize both (apply_score_directions) and average elementwise
  - Pull the two "base" scores: minus_llr (mendelian) / abs_llr (complex/eqtl),
    and embed_l2_flat_last
  - Rank within (dataset, subset); compute 7 ensembles of the two base scores:
    mean_rank, min_rank, max_rank, geomean_rank (≡ mean log rank, n=2),
    harmonic_rank, rrf_k60, zscore_mean (non-rank baseline)
  - Compute PairwiseAccuracy per subset, plus Global (PA over all match-pairs
    concatenated, no filter) and Macro Avg (unweighted mean of per-subset PAs
    over subsets with n_pairs ≥ 30) — canonical leaderboard convention from
    `bolinas.evals.metrics.compute_pairwise_metrics` on main (PR #178).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# n_pairs threshold for inclusion in Macro Avg. Leaderboard convention is 30.
N_MIN_MACRO: int = 30

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bolinas.evals.metrics import pairwise_accuracy
from bolinas.zeroshot_vep.scores import SCORE_NAMES, apply_score_directions


DATASETS = ["mendelian_traits", "complex_traits", "eqtl"]

LLR_BY_DATASET = {
    "mendelian_traits": "minus_llr",
    "complex_traits": "abs_llr",
    "eqtl": "abs_llr",
}
EMBED = "embed_l2_flat_last"

KEY = ["chrom", "pos", "ref", "alt"]


def _rc_path(ds: str) -> str:
    # Mendelian RC was computed in iter-4; complex/eqtl in iter-5.
    if ds == "mendelian_traits":
        return f"scratch/iter4/iter4_rc_exp166-p1B__win255__{ds}.parquet"
    return f"scratch/iter5/iter5_rc_exp166-p1B__win255__{ds}.parquet"


def _fwd_path(ds: str) -> str:
    return f"scratch/iter4/iter4_fwd_exp166-p1B__win255__{ds}.parquet"


def _build_signed_avg(ds: str) -> pd.DataFrame:
    """Load FWD + RC for one dataset, return DataFrame with signed AVG scores +
    variant metadata (subset, label, match_group)."""
    fwd = pd.read_parquet(_fwd_path(ds))
    rc = pd.read_parquet(_rc_path(ds))
    assert len(fwd) == len(rc), f"{ds}: FWD/RC row counts differ ({len(fwd)} vs {len(rc)})"

    rc_renamed = rc[KEY + list(SCORE_NAMES)].rename(columns={c: f"{c}__rc" for c in SCORE_NAMES})
    df = fwd.merge(rc_renamed, on=KEY, how="inner")
    assert len(df) == len(fwd), f"{ds}: merge dropped rows ({len(df)} vs {len(fwd)})"

    # match_group must not span subsets
    span = df.groupby("match_group")["subset"].nunique().max()
    assert span == 1, f"{ds}: match_group spans subsets (max nunique={span})"

    signed_fwd = apply_score_directions(df[SCORE_NAMES])
    rc_raw = df[[f"{c}__rc" for c in SCORE_NAMES]].rename(columns=lambda c: c.replace("__rc", ""))
    signed_rc = apply_score_directions(rc_raw)
    signed_avg = (signed_fwd + signed_rc) / 2.0

    meta = df[["subset", "label", "match_group"]].reset_index(drop=True)
    signed_avg = signed_avg.reset_index(drop=True)
    return pd.concat([meta, signed_avg], axis=1)


def _build_score_table(avg_df: pd.DataFrame, ds: str) -> pd.DataFrame:
    """Build the 9-column score table per variant (subset/label/match_group + 9 scores).

    Two base columns kept under their canonical names; ranks are computed
    within subset; 7 ensembles derived from the 2 base ranks.
    """
    llr_name = LLR_BY_DATASET[ds]

    out = pd.DataFrame({
        "subset": avg_df["subset"].values,
        "label": avg_df["label"].values,
        "match_group": avg_df["match_group"].values,
        llr_name: avg_df[llr_name].values,
        EMBED: avg_df[EMBED].values,
    })

    # Ranks within subset; method='average' handles ties.
    g = out.groupby("subset", sort=False)
    r_llr = g[llr_name].rank(method="average")
    r_emb = g[EMBED].rank(method="average")

    out["mean_rank"] = (r_llr + r_emb) / 2.0
    out["min_rank"] = np.minimum(r_llr.values, r_emb.values)
    out["max_rank"] = np.maximum(r_llr.values, r_emb.values)
    out["geomean_rank"] = np.sqrt(r_llr.values * r_emb.values)  # ≡ mean(log rank) for n=2 (monotone)
    out["harmonic_rank"] = 2.0 / (1.0 / r_llr.values + 1.0 / r_emb.values)
    # RRF: pandas .rank() returns rank=1 for the lowest score; standard RRF assumes
    # rank=1 is the BEST item. Flip to rank-from-top (n+1-r) so positives — which
    # tend to have higher signed scores → higher r → smaller rank-from-top — receive
    # a larger RRF score and the PA direction stays "higher = more pathogenic".
    n = g[llr_name].transform("count").values
    rft_llr = n + 1.0 - r_llr.values
    rft_emb = n + 1.0 - r_emb.values
    out["rrf_k60"] = 1.0 / (60.0 + rft_llr) + 1.0 / (60.0 + rft_emb)

    # Non-rank baseline: z-score per subset, then mean.
    def _z(s: pd.Series) -> np.ndarray:
        mean = g[s.name].transform("mean")
        std = g[s.name].transform("std").replace(0, 1.0)
        return ((s - mean) / std).values

    z_llr = _z(out[llr_name])
    z_emb = _z(out[EMBED])
    out["zscore_mean"] = (z_llr + z_emb) / 2.0

    return out


def _score_columns(ds: str) -> list[str]:
    llr_name = LLR_BY_DATASET[ds]
    return [llr_name, EMBED, "mean_rank", "min_rank", "max_rank",
            "geomean_rank", "harmonic_rank", "rrf_k60", "zscore_mean"]


def _compute_metrics(scored: pd.DataFrame, ds: str) -> pd.DataFrame:
    """Per-subset + Global + Macro Avg metrics matching the leaderboard convention.

    - Per-subset PA: one PA per (dataset, subset, score), no filter.
    - Global PA: single PA over all match-pairs concatenated (no filter).
    - Macro Avg PA: unweighted mean of per-subset PAs over subsets with
      ``n_pairs >= N_MIN_MACRO``. SE is ``sqrt(Σ SE_s²) / K`` (SE of unweighted
      mean of K independent estimates), matching ``compute_pairwise_metrics``
      on main (PR #178).
    """
    records = []
    cols = _score_columns(ds)

    # Per-subset PA
    per_subset_rows: dict[str, list[dict]] = {c: [] for c in cols}
    for subset, sub in scored.groupby("subset", sort=False):
        for col in cols:
            r = pairwise_accuracy(
                sub["label"], sub[col], sub["match_group"], alternative="greater"
            )
            row = {"dataset": ds, "agg": "per_subset", "subset": subset,
                   "score": col, **r}
            records.append(row)
            per_subset_rows[col].append(row)

    # Global — single PA over all match-pairs concatenated (no n filter).
    for col in cols:
        r = pairwise_accuracy(
            scored["label"], scored[col], scored["match_group"], alternative="greater"
        )
        records.append({"dataset": ds, "agg": "global", "subset": "_global_",
                        "score": col, **r})

    # Macro Avg — mean of per-subset PA over subsets with n_pairs >= N_MIN_MACRO.
    for col in cols:
        qualifying = [r for r in per_subset_rows[col] if r["n_pairs"] >= N_MIN_MACRO]
        assert qualifying, f"{ds} × {col}: no subsets with n_pairs >= {N_MIN_MACRO}"
        k = len(qualifying)
        value = float(np.mean([r["value"] for r in qualifying]))
        se = float(np.sqrt(sum(r["se"] ** 2 for r in qualifying)) / k)
        records.append({
            "dataset": ds, "agg": "macro_avg", "subset": "_macro_avg_",
            "score": col,
            "value": value, "se": se,
            "n_pairs": k,
            "n_ties": int(sum(r["n_ties"] for r in qualifying)),
            "p_value": float("nan"),
        })

    return pd.DataFrame(records)


def _global_table(metrics: pd.DataFrame, ds: str) -> pd.DataFrame:
    """Build the 9-row × 2-col (Global, Macro Avg) table for one dataset."""
    cols = _score_columns(ds)
    rows = []
    glb = metrics.query("dataset == @ds and agg == 'global'").set_index("score")
    macro = metrics.query("dataset == @ds and agg == 'macro_avg'").set_index("score")
    for c in cols:
        rows.append({
            "score": c,
            "global": glb.loc[c, "value"],
            "global_se": glb.loc[c, "se"],
            "macro": macro.loc[c, "value"],
            "macro_se": macro.loc[c, "se"],
            "macro_k": macro.loc[c, "n_pairs"],
        })
    return pd.DataFrame(rows)


def _per_subset_table(metrics: pd.DataFrame, ds: str) -> pd.DataFrame:
    """Wide per-subset × score table."""
    cols = _score_columns(ds)
    sub = (metrics.query("dataset == @ds and agg == 'per_subset'")
                  .pivot(index="subset", columns="score", values="value")
                  .reindex(columns=cols))
    return sub


def _format_pretty_name(score: str, ds: str) -> str:
    if score == LLR_BY_DATASET[ds]:
        return f"`{score}`"
    return f"`{score}`"


def _format_global_md(table: pd.DataFrame, ds: str) -> str:
    """Two-col Global / Macro Avg markdown table with SE in parens."""
    macro_k = int(table["macro_k"].iloc[0])
    lines = []
    lines.append(f"| score | Global | Macro Avg ({macro_k} subsets, n≥{N_MIN_MACRO}) |")
    lines.append("|---|---:|---:|")
    for _, row in table.iterrows():
        lines.append(
            f"| {_format_pretty_name(row['score'], ds)} | "
            f"{row['global']:.4f} ± {row['global_se']:.4f} | "
            f"{row['macro']:.4f} ± {row['macro_se']:.4f} |"
        )
    return "\n".join(lines)


def _format_subset_md(sub_table: pd.DataFrame, ds: str) -> str:
    cols = list(sub_table.columns)
    header = "| subset | " + " | ".join(cols) + " |"
    sep = "|---|" + "---:|" * len(cols)
    lines = [header, sep]
    for subset, row in sub_table.iterrows():
        vals = " | ".join(f"{v:.4f}" if pd.notna(v) else "—" for v in row)
        lines.append(f"| {subset} | {vals} |")
    return "\n".join(lines)


def _render_dataset_comment(metrics: pd.DataFrame, ds: str) -> str:
    g = _global_table(metrics, ds)
    sub = _per_subset_table(metrics, ds)
    llr_name = LLR_BY_DATASET[ds]
    macro_k = int(g["macro_k"].iloc[0])

    # Identify best ensemble (max of global + macro_avg)
    ens_cols = ["mean_rank", "min_rank", "max_rank", "geomean_rank",
                "harmonic_rank", "rrf_k60", "zscore_mean"]
    ens_rows = g[g["score"].isin(ens_cols)]
    best_global = ens_rows.loc[ens_rows["global"].idxmax()]
    best_macro = ens_rows.loc[ens_rows["macro"].idxmax()]
    base_global = g[g["score"] == llr_name].iloc[0]["global"]
    base_macro = g[g["score"] == llr_name].iloc[0]["macro"]
    emb_global = g[g["score"] == EMBED].iloc[0]["global"]
    emb_macro = g[g["score"] == EMBED].iloc[0]["macro"]

    lines = []
    lines.append(f"🤖 **exp166-p1B (FWD+RC AVG) — {ds}.** LLR-family default vs `embed_l2_flat_last` vs 7 ensembles of the two. Global + Macro Avg PairwiseAccuracy (leaderboard convention: Macro Avg = unweighted mean of per-subset PAs over subsets with n_pairs ≥ {N_MIN_MACRO}).")
    lines.append("")
    lines.append(f"## Global + Macro Avg ({macro_k} subsets qualify)")
    lines.append("")
    lines.append(_format_global_md(g, ds))
    lines.append("")
    lines.append(f"- Best ensemble (Global): `{best_global['score']}` = **{best_global['global']:.4f}** vs `{llr_name}` {base_global:.4f} (Δ {best_global['global']-base_global:+.4f}), vs `{EMBED}` {emb_global:.4f} (Δ {best_global['global']-emb_global:+.4f}).")
    lines.append(f"- Best ensemble (Macro Avg): `{best_macro['score']}` = **{best_macro['macro']:.4f}** vs `{llr_name}` {base_macro:.4f} (Δ {best_macro['macro']-base_macro:+.4f}), vs `{EMBED}` {emb_macro:.4f} (Δ {best_macro['macro']-emb_macro:+.4f}).")
    lines.append("")
    lines.append("## Per-subset PA")
    lines.append("")
    lines.append(_format_subset_md(sub, ds))
    lines.append("")
    return "\n".join(lines)


def _render_summary_comment(metrics: pd.DataFrame) -> str:
    """Cross-dataset summary: 1 row per (dataset × score) → Global, Macro Avg."""
    lines = ["🤖 **Iter-5 summary — exp166-p1B FWD+RC AVG, 3 datasets × 9 scores.**\n"]
    lines.append("Columns: Global / Macro Avg PairwiseAccuracy per dataset (canonical leaderboard convention: Macro Avg over subsets with n_pairs ≥ 30). **Bold** marks the best ensemble per (dataset, metric) cell.\n")

    score_order = ["LLR-default", "embed_l2_flat_last",
                   "mean_rank", "min_rank", "max_rank", "geomean_rank",
                   "harmonic_rank", "rrf_k60", "zscore_mean"]

    per_ds_tables = {ds: _global_table(metrics, ds).set_index("score") for ds in DATASETS}
    ens_cols = ["mean_rank", "min_rank", "max_rank", "geomean_rank",
                "harmonic_rank", "rrf_k60", "zscore_mean"]
    best_global = {ds: per_ds_tables[ds].loc[ens_cols]["global"].idxmax() for ds in DATASETS}
    best_macro = {ds: per_ds_tables[ds].loc[ens_cols]["macro"].idxmax() for ds in DATASETS}

    macro_ks = {ds: int(per_ds_tables[ds]["macro_k"].iloc[0]) for ds in DATASETS}
    header_parts = []
    for ds in DATASETS:
        header_parts.append(f"{ds} Global | {ds} Macro Avg (k={macro_ks[ds]})")
    header = "| score | " + " | ".join(header_parts) + " |"
    sep = "|---|" + "---:|---:|" * len(DATASETS)
    lines.append(header)
    lines.append(sep)

    for s in score_order:
        row = [s]
        for ds in DATASETS:
            llr_name = LLR_BY_DATASET[ds]
            actual_score = llr_name if s == "LLR-default" else s
            try:
                glb = per_ds_tables[ds].loc[actual_score, "global"]
                macro = per_ds_tables[ds].loc[actual_score, "macro"]
            except KeyError:
                row.extend(["—", "—"])
                continue
            glb_s = f"**{glb:.4f}**" if (s in ens_cols and s == best_global[ds]) else f"{glb:.4f}"
            macro_s = f"**{macro:.4f}**" if (s in ens_cols and s == best_macro[ds]) else f"{macro:.4f}"
            row.extend([glb_s, macro_s])
        if s == "LLR-default":
            row[0] = "`minus_llr` / `abs_llr`*"
        elif s == EMBED:
            row[0] = f"`{EMBED}`"
        else:
            row[0] = f"`{s}`"
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("*`minus_llr` for mendelian; `abs_llr` for complex_traits & eqtl.*")
    lines.append("")
    lines.append("**Conventions**:")
    lines.append(f"- **Global** = PA across ALL match-pairs concatenated (no n filter). **Macro Avg** = unweighted mean of per-subset PAs over subsets with n_pairs ≥ {N_MIN_MACRO} — matches `bolinas.evals.metrics.compute_pairwise_metrics` on main (PR #178).")
    lines.append("- Ranks computed within `(dataset, subset)` (matched pairs are subset-local).")
    lines.append("- `mean_rank` ≡ iter-2's `rk_minus_llr_plus_l2flat_last` formula (sum is monotone-equivalent to mean).")
    lines.append("- `rrf_k60` uses rank-from-top (`n+1-r`) so standard \"rank 1 = best\" RRF semantics apply; PA direction reproduced via `PA(x).value + PA(-x).value == 1` cross-check.")
    return "\n".join(lines)


def main() -> int:
    out_dir = Path("scratch/iter5")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    pretty_tables = {}

    for ds in DATASETS:
        print(f"\n[iter5] === {ds} ===", flush=True)
        avg_df = _build_signed_avg(ds)
        print(f"[iter5] {ds}: {len(avg_df):,} variants after FWD+RC merge", flush=True)
        scored = _build_score_table(avg_df, ds)
        metrics = _compute_metrics(scored, ds)
        all_metrics.append(metrics)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_parquet(out_dir / "iter5_metrics_exp166_p1B_rcavg.parquet", index=False)
    print(f"\n[iter5] wrote {out_dir / 'iter5_metrics_exp166_p1B_rcavg.parquet'}", flush=True)

    # Local report (kept for debug — full per-dataset + per-subset)
    md_path = out_dir / "iter5_report.md"
    md_lines = ["# Iter-5: exp166-p1B FWD+RC AVG — LLR + embed_l2_flat_last + ensembles\n"]
    for ds in DATASETS:
        md_lines.append(f"## {ds}\n")
        md_lines.append("### Global (pooled, macro)\n")
        md_lines.append(_format_global_md(_global_table(metrics, ds), ds))
        md_lines.append("\n\n### Per-subset PA\n")
        md_lines.append(_format_subset_md(_per_subset_table(metrics, ds), ds))
        md_lines.append("\n")
    md = "\n".join(md_lines)
    md_path.write_text(md)
    print(f"[iter5] wrote {md_path}", flush=True)

    # GitHub-ready per-dataset comments (one per dataset) + summary
    for ds in DATASETS:
        comment = _render_dataset_comment(metrics, ds)
        cpath = out_dir / f"iter5_comment_{ds}.md"
        cpath.write_text(comment)
        print(f"[iter5] wrote {cpath}", flush=True)

    summary = _render_summary_comment(metrics)
    spath = out_dir / "iter5_comment_summary.md"
    spath.write_text(summary)
    print(f"[iter5] wrote {spath}", flush=True)

    # Stdout summary
    for ds in DATASETS:
        print(f"\n=== {ds} (global) ===")
        print(_format_global_md(_global_table(metrics, ds), ds))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
