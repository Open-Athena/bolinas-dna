"""Iter 17 — disentangle MAF vs ld_score matching for complex_traits.

iter 14 / 16 showed both MAF and ld_score residual leaks, with finer bins
trading off across them. Question: is the issue jointly matching both, or
does each leak independently?

Four configurations on complex_traits, each with iter-11 tss/exon bins +
continuous on those features:

  A. Match neither MAF nor ld_score (production-ish baseline)
  B. Match MAF only      (continuous + bin)
  C. Match ld_score only (continuous + bin)
  D. Match both          (iter 14)

Reports per (subset, feature) pacc ± SE + significance.
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
    if math.isnan(p): return "    "
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "   "


def fmt(pacc, se, p):
    return f"{pacc:.3f}±{se:.3f} (p={p:.2g}{sig(p)})"


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

# Pre-build all bins (cheap)
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

# Common categorical (always-on): chrom, consequence_final, gene IDs, tss/exon dist bins
COMMON_CAT = ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
              "tss_dist_bin", "exon_dist_bin"]

CONFIGS = {
    "A. neither (only tss/exon)": {
        "continuous": ["tss_dist", "exon_dist"],
        "categorical": COMMON_CAT,
    },
    "B. MAF only": {
        "continuous": ["tss_dist", "exon_dist", "MAF"],
        "categorical": COMMON_CAT + ["MAF_bin"],
    },
    "C. ld_score only": {
        "continuous": ["tss_dist", "exon_dist", "ld_score"],
        "categorical": COMMON_CAT + ["ld_score_bin"],
    },
    "D. both (iter14)": {
        "continuous": ["tss_dist", "exon_dist", "MAF", "ld_score"],
        "categorical": COMMON_CAT + ["MAF_bin", "ld_score_bin"],
    },
}


REPORT_FEATURES = ["MAF", "ld_score", "tss_dist", "exon_dist"]
FOCUS_SUBSETS = ["distal", "missense_variant", "non_coding_transcript_exon_variant", "tss_proximal"]


def run_config(name, cfg):
    print(f"\n========== {name} ==========", flush=True)
    print(f"  continuous: {cfg['continuous']}")
    print(f"  categorical: {cfg['categorical']}")
    V_m = match_features(
        V.filter(pl.col("label")),
        V.filter(~pl.col("label")),
        cfg["continuous"],
        cfg["categorical"],
        k=1,
    )
    pos = V_m.filter(pl.col("label"))
    print(f"  total positives kept: {pos.height}/{n_total} = {100*pos.height/n_total:.0f}%", flush=True)
    print(f"\n  {'subset':<35s}  {'n':>4s}  " + "  ".join(f"{f:>22s}" for f in REPORT_FEATURES))
    for subset in FOCUS_SUBSETS:
        sub = V_m.filter(pl.col("consequence_group") == subset)
        n = sub.filter(pl.col("label")).height
        if n == 0:
            continue
        parts = []
        for f in REPORT_FEATURES:
            pacc, se, _, p = pacc_with_significance(sub, f)
            parts.append(fmt(pacc, se, p))
        print(f"  {subset:<35s}  {n:>4d}  " + "  ".join(parts), flush=True)


for name, cfg in CONFIGS.items():
    run_config(name, cfg)
