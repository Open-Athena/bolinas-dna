"""Iter 15 — pacc with closed-form significance tests.

For each (subset, feature), report:
  pacc ± SE  [p=...]   with significance flags

Closed form:
  N = #pairs (one per match_group, since k=1)
  B = #non-tie pairs
  pacc = (#wins + 0.5·#ties) / N
  Under H0 (no leak, P(pos>neg) = P(pos<neg) = (1-q)/2):
    SE(pacc) = 0.5 · sqrt(B) / N
    z = (pacc - 0.5) / SE
    p = 2 · (1 - Phi(|z|))    (two-sided)

Equivalent to a sign test on non-tied pairs.

Runs the recommended designs:
- Mendelian: iter 11 (subset-conditional tss/exon bins + continuous)
- Complex: iter 14 (iter 11 bins + MAF/ld_score bins + continuous on all 4)
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


def pacc_with_significance(V: pl.DataFrame, feat: str) -> tuple[float, float, int, int, float]:
    """Return (pacc, se, n_pairs, n_ties, p_value) for feature `feat`."""
    pos = V.filter(pl.col("label")).select(["match_group", feat]).rename({feat: "p"})
    neg = V.filter(~pl.col("label")).select(["match_group", feat]).rename({feat: "n"})
    paired = pos.join(neg, on="match_group", how="inner")
    N = paired.height
    if N == 0:
        return float("nan"), float("nan"), 0, 0, float("nan")
    diff = (paired["p"] - paired["n"]).to_numpy()
    n_wins = int((diff > 0).sum())
    n_ties = int((diff == 0).sum())
    n_losses = int((diff < 0).sum())
    B = n_wins + n_losses        # non-ties
    pacc = (n_wins + 0.5 * n_ties) / N
    if B == 0:
        # All ties; pacc=0.5 trivially, no power.
        return pacc, 0.0, N, n_ties, 1.0
    se = 0.5 * math.sqrt(B) / N
    z = (pacc - 0.5) / se
    p = 2.0 * norm.sf(abs(z))
    return pacc, se, N, n_ties, p


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


def report_sig(V_m: pl.DataFrame, features: list[str], title: str) -> None:
    print(f"\n========== {title} ==========")
    print(f"  {'subset':<35s}  {'n':>4s}  " + "  ".join(f"{f:>22s}" for f in features))
    for subset in sorted(V_m["consequence_group"].unique().to_list()):
        sub = V_m.filter(pl.col("consequence_group") == subset)
        n_pos = sub.filter(pl.col("label")).height
        if n_pos == 0:
            continue
        parts = []
        for f in features:
            pacc, se, N, ties, p = pacc_with_significance(sub, f)
            mark = sig_marker(p)
            parts.append(f"{pacc:.3f}±{se:.3f} p={p:.3g}{mark}")
        print(f"  {subset:<35s}  {n_pos:>4d}  " + "  ".join(parts))
    print(f"  Significance: * p<0.05, ** p<0.01, *** p<0.001 (two-sided sign test on non-tied pairs)")


# ============================================================================
# MENDELIAN: iter 11 design
# ============================================================================
print("Loading mendelian dataset_all...", flush=True)
V_m_data = pl.read_parquet(
    f"{BASE}/mendelian_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist"],
)
V_m_data = add_bin_oor(V_m_data, "tss_dist", TSS_BIN_EDGES, "_tss_full")
V_m_data = add_bin_oor(V_m_data, "exon_dist", EXON_BIN_EDGES, "_exon_full")
V_m_data = V_m_data.with_columns(
    pl.when(pl.col("consequence_group") == "tss_proximal")
    .then(pl.col("_tss_full"))
    .otherwise(pl.lit("NA"))
    .alias("tss_dist_bin"),
    pl.when(pl.col("consequence_group") == "splicing")
    .then(pl.col("_exon_full"))
    .otherwise(pl.lit("NA"))
    .alias("exon_dist_bin"),
)
V_m_matched = match_features(
    V_m_data.filter(pl.col("label")),
    V_m_data.filter(~pl.col("label")),
    ["tss_dist", "exon_dist"],
    ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
     "tss_dist_bin", "exon_dist_bin"],
    k=1,
)
n_pos = V_m_matched.filter(pl.col("label")).height
print(f"Mendelian (iter 11): {n_pos}/9767 positives kept", flush=True)
report_sig(V_m_matched, ["tss_dist", "exon_dist"],
           "MENDELIAN iter 11 — pacc ± SE, p-value (H0: pos and neg same distribution)")

# ============================================================================
# COMPLEX: iter 14 design
# ============================================================================
print("\n\nLoading complex_traits dataset_all...", flush=True)
V_c_data = pl.read_parquet(
    f"{BASE}/complex_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist", "MAF", "ld_score"],
)
V_c_data = add_bin_open(V_c_data, "MAF", MAF_BIN_EDGES, "MAF_bin")
V_c_data = add_bin_open(V_c_data, "ld_score", LD_BIN_EDGES, "ld_score_bin")
V_c_data = add_bin_oor(V_c_data, "tss_dist", TSS_BIN_EDGES, "_tss_full")
V_c_data = add_bin_oor(V_c_data, "exon_dist", EXON_BIN_EDGES, "_exon_full")
V_c_data = V_c_data.with_columns(
    pl.when(pl.col("consequence_group") == "tss_proximal")
    .then(pl.col("_tss_full"))
    .otherwise(pl.lit("NA"))
    .alias("tss_dist_bin"),
    pl.when(pl.col("consequence_group") == "splicing")
    .then(pl.col("_exon_full"))
    .otherwise(pl.lit("NA"))
    .alias("exon_dist_bin"),
)
V_c_matched = match_features(
    V_c_data.filter(pl.col("label")),
    V_c_data.filter(~pl.col("label")),
    ["tss_dist", "exon_dist", "MAF", "ld_score"],
    ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
     "MAF_bin", "ld_score_bin", "tss_dist_bin", "exon_dist_bin"],
    k=1,
)
n_pos_c = V_c_matched.filter(pl.col("label")).height
print(f"Complex (iter 14): {n_pos_c}/2165 positives kept", flush=True)
report_sig(V_c_matched, ["tss_dist", "exon_dist", "MAF", "ld_score"],
           "COMPLEX iter 14 — pacc ± SE, p-value (H0: pos and neg same distribution)")
