"""Iter 21 — split b1=(50, 200] of mendelian tss_dist_bin to fix the residual leak.

Iter 20 showed the entire tss_proximal/tss_dist leak (overall pacc=0.363, p=0.003)
is concentrated in b1 (50, 200]:
  n_pos=63, pacc=0.214, p=5e-6, Δ_mean=-40.9 bp

Split b1 at 100bp. Try multiple refinements:
  iter11 baseline: [0, 50, 200, 500, 1000]                  (4 bins, b1 wide)
  refined-1:       [0, 50, 100, 200, 500, 1000]             (5 bins, b1 split at 100)
  refined-2:       [0, 50, 100, 150, 200, 500, 1000]        (6 bins, 50bp granularity in (50,200])
"""
import math

import polars as pl
from scipy.stats import norm

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"
EXON_BIN_EDGES = [0, 5, 20, 30]

TSS_BIN_SCHEMES = {
    "iter11 baseline":   [0, 50, 200, 500, 1000],
    "refined-1 (split b1 at 100)":   [0, 50, 100, 200, 500, 1000],
    "refined-2 (50bp grid in b1)":   [0, 50, 100, 150, 200, 500, 1000],
}


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


print("Loading mendelian dataset_all...", flush=True)
V_base = pl.read_parquet(
    f"{BASE}/mendelian_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist"],
)
V_base = add_bin_oor(V_base, "exon_dist", EXON_BIN_EDGES, "_exon_full")
V_base = V_base.with_columns(
    pl.when(pl.col("consequence_group") == "splicing")
    .then(pl.col("_exon_full")).otherwise(pl.lit("NA")).alias("exon_dist_bin"),
)


def run(name, tss_edges):
    print(f"\n========== {name} | edges={tss_edges} ==========", flush=True)
    V = add_bin_oor(V_base, "tss_dist", tss_edges, "_tss_full")
    V = V.with_columns(
        pl.when(pl.col("consequence_group") == "tss_proximal")
        .then(pl.col("_tss_full")).otherwise(pl.lit("NA")).alias("tss_dist_bin"),
    )
    V_m = match_features(
        V.filter(pl.col("label")),
        V.filter(~pl.col("label")),
        ["tss_dist", "exon_dist"],
        ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
         "tss_dist_bin", "exon_dist_bin"],
        k=1,
    )
    pos = V_m.filter(pl.col("label"))
    sub = V_m.filter(pl.col("consequence_group") == "tss_proximal")
    n_total = pos.height
    n_tss = sub.filter(pl.col("label")).height

    print(f"  total kept: {n_total}/9767  ({100*n_total/9767:.0f}%)", flush=True)
    print(f"  tss_proximal kept: {n_tss}", flush=True)

    # Overall tss_proximal/tss_dist
    pacc, se, _, p = pacc_with_significance(sub, "tss_dist")
    print(f"  tss_proximal tss_dist: {fmt(pacc, se, p)}")

    # Per-bin breakdown
    n_bins = len(tss_edges) - 1
    print(f"  per-bin breakdown:")
    for i in range(n_bins):
        lbl = f"b{i}"
        sub_b = sub.filter(pl.col("tss_dist_bin") == lbl)
        n_pos_b = sub_b.filter(pl.col("label")).height
        if n_pos_b == 0:
            continue
        pacc_b, se_b, _, p_b = pacc_with_significance(sub_b, "tss_dist")
        rng = f"({tss_edges[i]:.0f}, {tss_edges[i+1]:.0f}]"
        print(f"    {lbl} {rng:<14s} n={n_pos_b:>3d}  {fmt(pacc_b, se_b, p_b)}")


for name, edges in TSS_BIN_SCHEMES.items():
    run(name, edges)
