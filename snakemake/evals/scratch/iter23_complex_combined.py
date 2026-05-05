"""Iter 23 — combined complex_traits production design (apply all wins).

Combines:
- Mendelian iter 22's tss_dist split: edges [0, 50, 100, 200, 500, 1000]
- Mendelian iter 22's splice filter: drop splicing w/ exon_dist > 30
- iter 14's MAF bins: [0, 0.005, 0.02, 0.05, 0.2, 0.5]
- iter 19A's refined ld_score bins: [0, 1, 5, 20, 50, 100, 500, 1e6]

Two configs:
  D. Combined with ld_score
  E. Combined without ld_score (drops ld_score from continuous + categorical)

Plus reference: iter 14 baseline (no refinements).

For each, full per-subset leakage with significance.
"""
import math

import polars as pl
from scipy.stats import norm

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"

TSS_BIN_EDGES_REFINED = [0, 50, 100, 200, 500, 1000]
TSS_BIN_EDGES_OLD = [0, 50, 200, 500, 1000]   # iter 14
EXON_BIN_EDGES = [0, 5, 20, 30]
MAF_BIN_EDGES = [0.0, 0.005, 0.02, 0.05, 0.2, 0.5]
LD_BIN_EDGES_REFINED = [0.0, 1.0, 5.0, 20.0, 50.0, 100.0, 500.0, 1e6]
LD_BIN_EDGES_OLD = [0.0, 1.0, 5.0, 20.0, 1e6]   # iter 14


def add_bin_open(V, feature, edges, col):
    expr = pl.lit("b0")
    n = len(edges) - 1
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        cond = (
            (pl.col(feature) > lo) & (pl.col(feature) <= hi)
            if i < n - 1
            else ((pl.col(feature) >= lo) & (pl.col(feature) <= hi))
        )
        expr = pl.when(cond).then(pl.lit(f"b{i}")).otherwise(expr)
    return V.with_columns(**{col: expr})


def add_bin_oor(V, feature, edges, col):
    n = len(edges) - 1
    expr = pl.lit("OOR")
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        cond = (
            (pl.col(feature) >= lo) & (pl.col(feature) < hi)
            if i < n - 1
            else ((pl.col(feature) >= lo) & (pl.col(feature) <= hi))
        )
        expr = pl.when(cond).then(pl.lit(f"b{i}")).otherwise(expr)
    return V.with_columns(**{col: expr})


def pacc_with_significance(V, feat):
    pos = V.filter(pl.col("label")).select(["match_group", feat]).rename({feat: "p"})
    neg = V.filter(~pl.col("label")).select(["match_group", feat]).rename({feat: "n"})
    paired = pos.join(neg, on="match_group", how="inner")
    N = paired.height
    if N == 0:
        return float("nan"), float("nan"), 0, float("nan")
    diff = (paired["p"] - paired["n"]).to_numpy()
    n_wins = int((diff > 0).sum())
    n_ties = int((diff == 0).sum())
    n_losses = int((diff < 0).sum())
    B = n_wins + n_losses
    pacc = (n_wins + 0.5 * n_ties) / N
    if B == 0:
        return pacc, 0.0, N, 1.0
    se = 0.5 * math.sqrt(B) / N
    z = (pacc - 0.5) / se
    p = 2.0 * norm.sf(abs(z))
    return pacc, se, N, p


def sig(p):
    if math.isnan(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def fmt(pacc, se, p):
    return f"{pacc:.3f}±{se:.3f} (p={p:.2g}{sig(p)})"


def report_full_leakage(V_m, features):
    print(f"  {'subset':<35s}  {'n':>5s}  " + "  ".join(f"{f:>22s}" for f in features))
    for subset in sorted(V_m["consequence_group"].unique().to_list()):
        sub = V_m.filter(pl.col("consequence_group") == subset)
        n_pos = sub.filter(pl.col("label")).height
        if n_pos == 0:
            continue
        parts = []
        for f in features:
            pacc, se, _, p = pacc_with_significance(sub, f)
            parts.append(fmt(pacc, se, p))
        print(f"  {subset:<35s}  {n_pos:>5d}  " + "  ".join(parts))


print("Loading complex_traits dataset_all...", flush=True)
V_full = pl.read_parquet(
    f"{BASE}/complex_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist", "MAF", "ld_score"],
)
n_pos_input = V_full.filter(pl.col("label")).height
print(f"  {V_full.height} rows; {n_pos_input} positives in dataset_all", flush=True)


def build_with_bins(V_in, tss_edges, ld_edges):
    V = add_bin_open(V_in, "MAF", MAF_BIN_EDGES, "MAF_bin")
    V = add_bin_open(V, "ld_score", ld_edges, "ld_score_bin")
    V = add_bin_oor(V, "tss_dist", tss_edges, "_tss_full")
    V = add_bin_oor(V, "exon_dist", EXON_BIN_EDGES, "_exon_full")
    return V.with_columns(
        pl.when(pl.col("consequence_group") == "tss_proximal")
        .then(pl.col("_tss_full")).otherwise(pl.lit("NA")).alias("tss_dist_bin"),
        pl.when(pl.col("consequence_group") == "splicing")
        .then(pl.col("_exon_full")).otherwise(pl.lit("NA")).alias("exon_dist_bin"),
    )


def run(name, V_in, tss_edges, ld_edges, include_ld_score):
    V = build_with_bins(V_in, tss_edges, ld_edges)
    continuous = ["tss_dist", "exon_dist", "MAF"]
    categorical = ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
                   "MAF_bin", "tss_dist_bin", "exon_dist_bin"]
    if include_ld_score:
        continuous.append("ld_score")
        categorical.append("ld_score_bin")
    V_m = match_features(
        V.filter(pl.col("label")),
        V.filter(~pl.col("label")),
        continuous,
        categorical,
        k=1,
    )
    pos = V_m.filter(pl.col("label"))
    n_total = pos.height
    print(f"\n========== {name} ==========")
    print(f"  total kept: {n_total}/{n_pos_input}  ({100*n_total/n_pos_input:.0f}%)", flush=True)
    print(f"  features used: continuous={continuous}")
    print(f"                 categorical={categorical}")
    print(f"\n  full leakage:")
    report_full_leakage(V_m, ["tss_dist", "exon_dist", "MAF", "ld_score"])
    return V_m


# ---- C. iter 14 baseline (for reference) ----
V_m_C = run("C. iter 14 baseline (no refinements, no filter)",
             V_full, TSS_BIN_EDGES_OLD, LD_BIN_EDGES_OLD, include_ld_score=True)


# ---- Pre-filter: drop splice with exon_dist > 30 ----
V_filtered = V_full.filter(
    ~((pl.col("consequence_group") == "splicing") & (pl.col("exon_dist") > 30))
)
n_dropped = V_full.height - V_filtered.height
n_pos_dropped = n_pos_input - V_filtered.filter(pl.col("label")).height
print(f"\n[Pre-filter] dropped {n_dropped} variants ({n_pos_dropped} positives) with consequence_group=='splicing' AND exon_dist > 30", flush=True)


# ---- D. Refined design WITH ld_score ----
V_m_D = run("D. Combined refined (refined tss + filter + refined ld_score)",
             V_filtered, TSS_BIN_EDGES_REFINED, LD_BIN_EDGES_REFINED, include_ld_score=True)


# ---- E. Refined design WITHOUT ld_score ----
V_m_E = run("E. Combined refined, NO ld_score (refined tss + filter, drop ld_score)",
             V_filtered, TSS_BIN_EDGES_REFINED, LD_BIN_EDGES_REFINED, include_ld_score=False)


# ---- Side-by-side per-subset retention ----
print(f"\n\n========== Per-subset retention (C/D/E) ==========")
print(f"  {'subset':<35s}  {'C(iter14)':>9s}  {'D(refined)':>10s}  {'E(no ld)':>9s}")
for subset in sorted(set(V_m_C["consequence_group"].unique().to_list() +
                          V_m_D["consequence_group"].unique().to_list() +
                          V_m_E["consequence_group"].unique().to_list())):
    nC = V_m_C.filter((pl.col("consequence_group") == subset) & pl.col("label")).height
    nD = V_m_D.filter((pl.col("consequence_group") == subset) & pl.col("label")).height
    nE = V_m_E.filter((pl.col("consequence_group") == subset) & pl.col("label")).height
    print(f"  {subset:<35s}  {nC:>9d}  {nD:>10d}  {nE:>9d}")
