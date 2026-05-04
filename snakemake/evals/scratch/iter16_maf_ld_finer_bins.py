"""Iter 16 — sweep finer MAF and ld_score bin schemes for complex_traits.

Iter 15 showed iter 14 still has statistically significant MAF/ld_score leaks
in distal (N=1148) + missense (N=152) + ncRNA (N=63). Goal here: see if finer
bins close the leaks at acceptable retention cost.

Design held: iter 14 (subset-conditional tss/exon bins, continuous on all 4
features). Only MAF_bin and ld_score_bin schemes vary.

Reports per scheme:
  total_retention,
  per (subset, feature): pacc ± SE, p-value
focused on distal / missense / ncRNA where there's enough N to detect leaks.
"""
import math

import polars as pl
from scipy.stats import norm

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"

TSS_BIN_EDGES = [0, 50, 200, 500, 1000]
EXON_BIN_EDGES = [0, 5, 20, 30]

# MAF bin schemes (current iter-14 default = 5 bins)
MAF_SCHEMES = {
    "3 bins":           [0.0, 0.01, 0.05, 0.5],
    "5 bins (iter14)":  [0.0, 0.005, 0.02, 0.05, 0.2, 0.5],
    "7 bins":           [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.2, 0.5],
    "10 bins":          [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
    "15 bins":          [0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.5],
}

# ld_score bin schemes (current iter-14 default = 4 bins)
LD_SCHEMES = {
    "3 bins":           [0.0, 2.0, 10.0, 1e6],
    "4 bins (iter14)":  [0.0, 1.0, 5.0, 20.0, 1e6],
    "6 bins":           [0.0, 0.5, 1.5, 3.0, 7.0, 30.0, 1e6],
    "8 bins":           [0.0, 0.25, 0.75, 1.5, 3.0, 5.0, 10.0, 30.0, 1e6],
    "10 bins":          [0.0, 0.1, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0, 15.0, 30.0, 1e6],
}


def add_bin_open(V: pl.DataFrame, feature: str, edges: list[float], col: str) -> pl.DataFrame:
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


def add_bin_oor(V: pl.DataFrame, feature: str, edges: list[float], col: str) -> pl.DataFrame:
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


def pacc_with_significance(V: pl.DataFrame, feat: str) -> tuple[float, float, int, float]:
    """Return (pacc, se, n_pairs, p_value)."""
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


def sig_marker(p: float) -> str:
    if math.isnan(p):
        return "    "
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "** "
    if p < 0.05:
        return "*  "
    return "   "


def fmt(pacc: float, se: float, p: float) -> str:
    return f"{pacc:.3f}±{se:.3f} (p={p:.2g}{sig_marker(p)})"


def run_scheme(V: pl.DataFrame, maf_edges, ld_edges) -> dict:
    Vb = add_bin_open(V, "MAF", maf_edges, "MAF_bin")
    Vb = add_bin_open(Vb, "ld_score", ld_edges, "ld_score_bin")
    Vb = add_bin_oor(Vb, "tss_dist", TSS_BIN_EDGES, "_tss_full")
    Vb = add_bin_oor(Vb, "exon_dist", EXON_BIN_EDGES, "_exon_full")
    Vb = Vb.with_columns(
        pl.when(pl.col("consequence_group") == "tss_proximal")
        .then(pl.col("_tss_full")).otherwise(pl.lit("NA")).alias("tss_dist_bin"),
        pl.when(pl.col("consequence_group") == "splicing")
        .then(pl.col("_exon_full")).otherwise(pl.lit("NA")).alias("exon_dist_bin"),
    )
    V_m = match_features(
        Vb.filter(pl.col("label")),
        Vb.filter(~pl.col("label")),
        ["tss_dist", "exon_dist", "MAF", "ld_score"],
        ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
         "MAF_bin", "ld_score_bin", "tss_dist_bin", "exon_dist_bin"],
        k=1,
    )
    out = {"total": V_m.filter(pl.col("label")).height}
    for sub_name in ["distal", "missense_variant", "non_coding_transcript_exon_variant", "tss_proximal"]:
        sub = V_m.filter(pl.col("consequence_group") == sub_name)
        n = sub.filter(pl.col("label")).height
        out[f"{sub_name}_n"] = n
        for f in ["MAF", "ld_score", "tss_dist", "exon_dist"]:
            pacc, se, _, p = pacc_with_significance(sub, f)
            out[f"{sub_name}_{f}"] = (pacc, se, p)
    return out


print("Loading complex_traits dataset_all...", flush=True)
V = pl.read_parquet(
    f"{BASE}/complex_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist", "MAF", "ld_score"],
)
n_total = V.filter(pl.col("label")).height
print(f"  {V.height} rows; {n_total} positives", flush=True)


# ============================================================================
# MAF sweep (ld_score = iter-14 default 4 bins)
# ============================================================================
print(f"\n========== MAF sweep (ld_score = iter-14 default '4 bins') ==========")
default_ld = LD_SCHEMES["4 bins (iter14)"]
print(f"{'scheme':<20s}  {'total':>5s}  {'distal_n':>8s}  {'distal MAF':<28s}  {'distal ld_score':<28s}  {'missense MAF':<28s}  {'missense ld_score':<28s}")
for name, maf_edges in MAF_SCHEMES.items():
    r = run_scheme(V, maf_edges, default_ld)
    p_d = r["distal_MAF"]; l_d = r["distal_ld_score"]
    p_m = r["missense_variant_MAF"]; l_m = r["missense_variant_ld_score"]
    print(f"{name:<20s}  {r['total']:>5d}  {r['distal_n']:>8d}  "
          f"{fmt(*p_d):<28s}  {fmt(*l_d):<28s}  {fmt(*p_m):<28s}  {fmt(*l_m):<28s}", flush=True)


# ============================================================================
# ld_score sweep (MAF = iter-14 default 5 bins)
# ============================================================================
print(f"\n========== ld_score sweep (MAF = iter-14 default '5 bins') ==========")
default_maf = MAF_SCHEMES["5 bins (iter14)"]
print(f"{'scheme':<20s}  {'total':>5s}  {'distal_n':>8s}  {'distal MAF':<28s}  {'distal ld_score':<28s}  {'missense MAF':<28s}  {'missense ld_score':<28s}")
for name, ld_edges in LD_SCHEMES.items():
    r = run_scheme(V, default_maf, ld_edges)
    p_d = r["distal_MAF"]; l_d = r["distal_ld_score"]
    p_m = r["missense_variant_MAF"]; l_m = r["missense_variant_ld_score"]
    print(f"{name:<20s}  {r['total']:>5d}  {r['distal_n']:>8d}  "
          f"{fmt(*p_d):<28s}  {fmt(*l_d):<28s}  {fmt(*p_m):<28s}  {fmt(*l_m):<28s}", flush=True)


# ============================================================================
# Best joint: pick a refined MAF + refined ld_score
# ============================================================================
print(f"\n========== Joint: MAF=10 bins + ld_score=8 bins ==========")
r = run_scheme(V, MAF_SCHEMES["10 bins"], LD_SCHEMES["8 bins"])
print(f"  total: {r['total']}/{n_total} = {100*r['total']/n_total:.0f}%")
for sub in ["distal", "missense_variant", "non_coding_transcript_exon_variant", "tss_proximal"]:
    n = r[f"{sub}_n"]
    if n == 0:
        continue
    print(f"  {sub:<35s} n={n:>4d}")
    for f in ["MAF", "ld_score", "tss_dist", "exon_dist"]:
        pacc, se, p = r[f"{sub}_{f}"]
        print(f"    {f:<10s} {fmt(pacc, se, p)}")
