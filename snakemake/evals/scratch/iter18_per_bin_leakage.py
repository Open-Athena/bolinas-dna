"""Iter 18 — per-bin leakage breakdown for complex_traits.

iter 15 / 16 / 17 showed residual MAF and ld_score leaks in distal, missense,
ncRNA. Question: WHICH bins drive the leak — low MAF, high MAF? low ld_score?

Uses iter-14 design (MAF + ld_score bins + continuous on all 4 features).
For each problematic (subset, feature):
  - Number of positives per bin
  - Within each bin: pacc, mean(pos), mean(neg)

Reveals whether the leak is concentrated in specific bin(s) and whether it's
asymmetric (e.g., positives systematically lower MAF in the high-MAF bin).
"""
import math

import polars as pl
from scipy.stats import norm

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"

TSS_BIN_EDGES = [0, 50, 200, 500, 1000]
EXON_BIN_EDGES = [0, 5, 20, 30]
MAF_BIN_EDGES = [0.0, 0.005, 0.02, 0.05, 0.2, 0.5]
LD_BIN_EDGES = [0.0, 1.0, 5.0, 20.0, 1e6]


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


def edges_to_label(edges, i):
    """Render bin label as (lo, hi]."""
    lo, hi = edges[i], edges[i + 1]
    if hi >= 1e5:
        return f"({lo:.3g}, inf]"
    return f"({lo:.3g}, {hi:.3g}]"


print("Loading complex_traits dataset_all...", flush=True)
V = pl.read_parquet(
    f"{BASE}/complex_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist", "MAF", "ld_score"],
)
print(f"  {V.height} rows; {V.filter(pl.col('label')).height} positives", flush=True)

# Build all bins
V = add_bin_open(V, "MAF", MAF_BIN_EDGES, "MAF_bin")
V = add_bin_open(V, "ld_score", LD_BIN_EDGES, "ld_score_bin")
V = add_bin_oor(V, "tss_dist", TSS_BIN_EDGES, "_tss_full")
V = add_bin_oor(V, "exon_dist", EXON_BIN_EDGES, "_exon_full")
V = V.with_columns(
    pl.when(pl.col("consequence_group") == "tss_proximal")
    .then(pl.col("_tss_full")).otherwise(pl.lit("NA")).alias("tss_dist_bin"),
    pl.when(pl.col("consequence_group") == "splicing")
    .then(pl.col("_exon_full")).otherwise(pl.lit("NA")).alias("exon_dist_bin"),
)

# Run iter-14 matching
print("\nRunning iter-14 matching (MAF + ld_score bins + continuous on all 4)...", flush=True)
V_m = match_features(
    V.filter(pl.col("label")),
    V.filter(~pl.col("label")),
    ["tss_dist", "exon_dist", "MAF", "ld_score"],
    ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
     "MAF_bin", "ld_score_bin", "tss_dist_bin", "exon_dist_bin"],
    k=1,
)
print(f"  total positives kept: {V_m.filter(pl.col('label')).height}", flush=True)


# Per-bin breakdown for problematic subsets
FOCUS_SUBSETS = ["distal", "missense_variant", "non_coding_transcript_exon_variant"]


def per_bin_report(subset, feature, bin_col, edges):
    print(f"\n{'='*80}")
    print(f"  {subset} — {feature} broken down by {bin_col}")
    print(f"{'='*80}")
    sub = V_m.filter(pl.col("consequence_group") == subset)
    n_total = sub.filter(pl.col("label")).height

    # Per-bin counts
    bin_counts = (
        sub.filter(pl.col("label"))
        .group_by(bin_col)
        .agg(pl.len().alias("n_pos"))
        .sort(bin_col)
    )

    print(f"  {'bin':<20s}  {'range':<18s}  {'n_pos':>5s}  {'pacc±SE':<14s}  {'p':<8s}  {'mean_pos':>8s}  {'mean_neg':>8s}  {'Δ_mean':>8s}")
    print("  " + "-" * 96)

    # Iterate over actual bin labels in the data, in order b0, b1, ..., OOR
    n_bins = len(edges) - 1
    bin_labels = [f"b{i}" for i in range(n_bins)]
    if "OOR" in sub[bin_col].unique().to_list():
        bin_labels.append("OOR")

    overall_pacc, overall_se, overall_N, overall_p, overall_mp, overall_mn = pacc_with_significance(sub, feature)

    for i, lbl in enumerate(bin_labels):
        sub_b = sub.filter(pl.col(bin_col) == lbl)
        n_pos = sub_b.filter(pl.col("label")).height
        if n_pos == 0:
            continue
        pacc, se, N, p, mean_p, mean_n = pacc_with_significance(sub_b, feature)
        rng = edges_to_label(edges, i) if lbl != "OOR" else "OOR"
        delta = mean_p - mean_n
        print(f"  {lbl:<20s}  {rng:<18s}  {n_pos:>5d}  "
              f"{pacc:.3f}±{se:.3f}  {p:<7.2g}{sig(p):<2s}  "
              f"{mean_p:>8.4g}  {mean_n:>8.4g}  {delta:>+8.4g}")
    print("  " + "-" * 96)
    print(f"  {'OVERALL':<20s}  {'':<18s}  {n_total:>5d}  "
          f"{overall_pacc:.3f}±{overall_se:.3f}  {overall_p:<7.2g}{sig(overall_p):<2s}  "
          f"{overall_mp:>8.4g}  {overall_mn:>8.4g}  {overall_mp-overall_mn:>+8.4g}")


# MAF breakdown
for subset in FOCUS_SUBSETS:
    per_bin_report(subset, "MAF", "MAF_bin", MAF_BIN_EDGES)

# ld_score breakdown
for subset in FOCUS_SUBSETS:
    per_bin_report(subset, "ld_score", "ld_score_bin", LD_BIN_EDGES)
