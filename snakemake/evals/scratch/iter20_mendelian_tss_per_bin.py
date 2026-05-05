"""Iter 20 — per-bin breakdown of tss_proximal/tss_dist leak in mendelian.

Iter 15 showed mendelian iter 11 has one significant residual leak:
  tss_proximal/tss_dist: pacc=0.363±0.046, p=0.003**

Question: which bin(s) of tss_dist drive this? Same diagnostic as iter 18 for
complex, but on the mendelian dataset and zoomed into the tss_proximal subset.

Bins: [0, 50, 200, 500, 1000] (4 bins, b0-b3, OOR fallback).
"""
import math

import polars as pl
from scipy.stats import norm

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"
TSS_BIN_EDGES = [0, 50, 200, 500, 1000]
EXON_BIN_EDGES = [0, 5, 20, 30]


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
        return float("nan"), float("nan"), 0, float("nan"), float("nan"), float("nan")
    diff = (paired["p"] - paired["n"]).to_numpy()
    n_wins = int((diff > 0).sum())
    n_ties = int((diff == 0).sum())
    n_losses = int((diff < 0).sum())
    B = n_wins + n_losses
    pacc = (n_wins + 0.5 * n_ties) / N
    mean_p = float(paired["p"].mean())
    mean_n = float(paired["n"].mean())
    if B == 0:
        return pacc, 0.0, N, 1.0, mean_p, mean_n
    se = 0.5 * math.sqrt(B) / N
    z = (pacc - 0.5) / se
    p = 2.0 * norm.sf(abs(z))
    return pacc, se, N, p, mean_p, mean_n


def sig(p):
    if math.isnan(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def bin_label(edges, i):
    lo, hi = edges[i], edges[i + 1]
    return f"({lo:.0f}, {hi:.0f}]"


print("Loading mendelian dataset_all...", flush=True)
V = pl.read_parquet(
    f"{BASE}/mendelian_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist"],
)
print(f"  {V.height} rows; {V.filter(pl.col('label')).height} positives", flush=True)

V = add_bin_oor(V, "tss_dist", TSS_BIN_EDGES, "_tss_full")
V = add_bin_oor(V, "exon_dist", EXON_BIN_EDGES, "_exon_full")
V = V.with_columns(
    pl.when(pl.col("consequence_group") == "tss_proximal")
    .then(pl.col("_tss_full")).otherwise(pl.lit("NA")).alias("tss_dist_bin"),
    pl.when(pl.col("consequence_group") == "splicing")
    .then(pl.col("_exon_full")).otherwise(pl.lit("NA")).alias("exon_dist_bin"),
)

print("\nRunning iter-11 matching...", flush=True)
V_m = match_features(
    V.filter(pl.col("label")),
    V.filter(~pl.col("label")),
    ["tss_dist", "exon_dist"],
    ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
     "tss_dist_bin", "exon_dist_bin"],
    k=1,
)
print(f"  total positives kept: {V_m.filter(pl.col('label')).height}", flush=True)

# Per-bin breakdown for tss_proximal / tss_dist
sub = V_m.filter(pl.col("consequence_group") == "tss_proximal")
n_total = sub.filter(pl.col("label")).height
print(f"\n{'='*100}")
print(f"  tss_proximal — tss_dist broken down by tss_dist_bin (mendelian iter 11)")
print(f"{'='*100}")
print(f"  {'bin':<6s}  {'range':<14s}  {'n_pos':>5s}  {'pacc±SE':<14s}  {'p':<10s}  {'mean_pos':>9s}  {'mean_neg':>9s}  {'Δ_mean':>9s}")
print("  " + "-" * 96)

n_bins = len(TSS_BIN_EDGES) - 1
bin_labels = [f"b{i}" for i in range(n_bins)]

overall = pacc_with_significance(sub, "tss_dist")

for i, lbl in enumerate(bin_labels):
    sub_b = sub.filter(pl.col("tss_dist_bin") == lbl)
    n_pos = sub_b.filter(pl.col("label")).height
    if n_pos == 0:
        continue
    pacc, se, N, p, mean_p, mean_n = pacc_with_significance(sub_b, "tss_dist")
    rng = bin_label(TSS_BIN_EDGES, i)
    delta = mean_p - mean_n
    print(f"  {lbl:<6s}  {rng:<14s}  {n_pos:>5d}  "
          f"{pacc:.3f}±{se:.3f}  {p:<7.3g}{sig(p):<3s}  "
          f"{mean_p:>9.1f}  {mean_n:>9.1f}  {delta:>+9.1f}")

print("  " + "-" * 96)
overall_pacc, overall_se, _, overall_p, overall_mp, overall_mn = overall
print(f"  {'TOTAL':<6s}  {'':<14s}  {n_total:>5d}  "
      f"{overall_pacc:.3f}±{overall_se:.3f}  {overall_p:<7.3g}{sig(overall_p):<3s}  "
      f"{overall_mp:>9.1f}  {overall_mn:>9.1f}  {overall_mp-overall_mn:>+9.1f}")
