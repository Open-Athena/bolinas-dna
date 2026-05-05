"""Iter 24 — MAF Pareto sweep for complex_traits (no ld_score variant).

Decision so far for complex:
- Drop ld_score (user: "ok to only match MAF for now").
- Apply mendelian-style refinements:
    tss_dist_bin: [0, 50, 100, 200, 500, 1000]
    exon_dist_bin: [0, 5, 20, 30]
    pre-filter: drop splicing w/ exon_dist > 30

Open question: which MAF bin scheme balances retention vs. missense MAF leak.
Iter 18 showed missense MAF leak concentrates in the (0, 0.005] bin (rare-vs-edge).
Iter 23 E (5 MAF bins) gave: 1586 retained, missense MAF=0.678 (p=2.4e-7 ***).

Sweep MAF bin schemes:
  3 bins   [0, 0.01, 0.05, 0.5]
  5 bins   [0, 0.005, 0.02, 0.05, 0.2, 0.5]                          (iter 14 default)
  7 bins   [0, 0.0015, 0.003, 0.005, 0.02, 0.05, 0.2, 0.5]            (split low end)
  10 bins  [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
  15 bins  [0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.5]
  20 bins  (very fine in low-MAF range)

For each: retention + per-(subset, MAF) significance.
"""
import math

import polars as pl
from scipy.stats import norm

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"
TSS_BIN_EDGES = [0, 50, 100, 200, 500, 1000]
EXON_BIN_EDGES = [0, 5, 20, 30]

MAF_SCHEMES = {
    "3 bins":  [0.0, 0.01, 0.05, 0.5],
    "5 bins (iter14 default)":  [0.0, 0.005, 0.02, 0.05, 0.2, 0.5],
    "7 bins (split low)":  [0.0, 0.0015, 0.003, 0.005, 0.02, 0.05, 0.2, 0.5],
    "10 bins": [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
    "15 bins": [0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.5],
    "20 bins (very fine low)": [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004,
                                  0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.5],
}


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


print("Loading complex_traits dataset_all...", flush=True)
V_full = pl.read_parquet(
    f"{BASE}/complex_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist", "MAF", "ld_score"],
)
n_pos_input = V_full.filter(pl.col("label")).height
print(f"  {V_full.height} rows; {n_pos_input} positives", flush=True)

# Apply pre-filter (mendelian-style splice filter)
V_filtered = V_full.filter(
    ~((pl.col("consequence_group") == "splicing") & (pl.col("exon_dist") > 30))
)
n_dropped = V_full.height - V_filtered.height
n_pos_dropped = n_pos_input - V_filtered.filter(pl.col("label")).height
print(f"[Pre-filter] dropped {n_dropped} variants ({n_pos_dropped} positives)", flush=True)

# Pre-build distance bins (always-on, mendelian-style)
V_filtered = add_bin_oor(V_filtered, "tss_dist", TSS_BIN_EDGES, "_tss_full")
V_filtered = add_bin_oor(V_filtered, "exon_dist", EXON_BIN_EDGES, "_exon_full")
V_filtered = V_filtered.with_columns(
    pl.when(pl.col("consequence_group") == "tss_proximal")
    .then(pl.col("_tss_full")).otherwise(pl.lit("NA")).alias("tss_dist_bin"),
    pl.when(pl.col("consequence_group") == "splicing")
    .then(pl.col("_exon_full")).otherwise(pl.lit("NA")).alias("exon_dist_bin"),
)


def run_scheme(name, maf_edges):
    V = add_bin_open(V_filtered, "MAF", maf_edges, "MAF_bin")
    V_m = match_features(
        V.filter(pl.col("label")),
        V.filter(~pl.col("label")),
        ["tss_dist", "exon_dist", "MAF"],   # NO ld_score
        ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
         "MAF_bin", "tss_dist_bin", "exon_dist_bin"],   # NO ld_score_bin
        k=1,
    )
    n_total = V_m.filter(pl.col("label")).height

    print(f"\n========== {name} | edges={maf_edges} ==========")
    print(f"  total kept: {n_total}/{n_pos_input}  ({100*n_total/n_pos_input:.0f}%)", flush=True)

    print(f"  per-subset MAF + tss_dist leakage:")
    print(f"  {'subset':<35s}  {'n':>5s}  {'MAF':<28s}  {'tss_dist':<28s}  {'exon_dist':<28s}")
    for subset in ["distal", "missense_variant", "non_coding_transcript_exon_variant",
                   "tss_proximal", "5_prime_UTR_variant", "3_prime_UTR_variant"]:
        sub = V_m.filter(pl.col("consequence_group") == subset)
        n_pos = sub.filter(pl.col("label")).height
        if n_pos == 0:
            continue
        parts = []
        for f in ["MAF", "tss_dist", "exon_dist"]:
            pacc, se, _, p = pacc_with_significance(sub, f)
            parts.append(fmt(pacc, se, p))
        print(f"  {subset:<35s}  {n_pos:>5d}  " + "  ".join(parts), flush=True)


for name, edges in MAF_SCHEMES.items():
    run_scheme(name, edges)
