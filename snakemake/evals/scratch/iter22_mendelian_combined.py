"""Iter 22 — combined mendelian production design.

Two refinements on iter 11:
1. Split tss_dist b1=(50, 200] at 100 → edges [0, 50, 100, 200, 500, 1000].
   Closes the residual tss_proximal/tss_dist leak (iter 20 / iter 21).
2. Filter out variants with consequence_group == "splicing" AND exon_dist > 30
   BEFORE matching. Drops the 8 splicing positives at non-coding-transcript
   splice sites (their PC-only exon_dist is misleading); removes the OOR
   special case from exon_dist_bin.

Reports:
- Total retention vs iter 11
- Per-subset retention vs iter 11
- Full leakage table with significance for both designs
"""
import math

import polars as pl
from scipy.stats import norm

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"
TSS_BIN_EDGES_OLD = [0, 50, 200, 500, 1000]                  # iter 11
TSS_BIN_EDGES_NEW = [0, 50, 100, 200, 500, 1000]             # iter 22 (b1 split at 100)
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


def report_full_leakage(V_m, label):
    print(f"\n  {label} — full leakage by subset:")
    print(f"  {'subset':<35s}  {'n':>5s}  {'tss_dist':<28s}  {'exon_dist':<28s}")
    for subset in sorted(V_m["consequence_group"].unique().to_list()):
        sub = V_m.filter(pl.col("consequence_group") == subset)
        n_pos = sub.filter(pl.col("label")).height
        if n_pos == 0:
            continue
        parts = []
        for f in ["tss_dist", "exon_dist"]:
            pacc, se, _, p = pacc_with_significance(sub, f)
            parts.append(fmt(pacc, se, p))
        print(f"  {subset:<35s}  {n_pos:>5d}  " + "  ".join(parts))


print("Loading mendelian dataset_all...", flush=True)
V_full = pl.read_parquet(
    f"{BASE}/mendelian_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist"],
)
n_pos_input = V_full.filter(pl.col("label")).height
print(f"  {V_full.height} rows; {n_pos_input} positives in dataset_all", flush=True)


def run(name, tss_edges, V_in):
    V = add_bin_oor(V_in, "tss_dist", tss_edges, "_tss_full")
    V = add_bin_oor(V, "exon_dist", EXON_BIN_EDGES, "_exon_full")
    V = V.with_columns(
        pl.when(pl.col("consequence_group") == "tss_proximal")
        .then(pl.col("_tss_full")).otherwise(pl.lit("NA")).alias("tss_dist_bin"),
        pl.when(pl.col("consequence_group") == "splicing")
        .then(pl.col("_exon_full")).otherwise(pl.lit("NA")).alias("exon_dist_bin"),
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
    n_total = pos.height
    print(f"\n========== {name} ==========")
    print(f"  total kept: {n_total}/{n_pos_input}", flush=True)
    return V_m


# ---- A. iter 11 baseline (no filter, old tss bins) ----
V_m_iter11 = run("A. iter 11 baseline (old tss bins, no splice filter)",
                  TSS_BIN_EDGES_OLD, V_full)
report_full_leakage(V_m_iter11, "iter 11 baseline")

# ---- B. iter 22 final: refined tss + splice filter ----
# Drop splicing variants with exon_dist > 30 BEFORE matching
V_filtered = V_full.filter(
    ~((pl.col("consequence_group") == "splicing") & (pl.col("exon_dist") > 30))
)
n_dropped = V_full.height - V_filtered.height
n_pos_dropped = V_full.filter(pl.col("label")).height - V_filtered.filter(pl.col("label")).height
print(f"\nFiltered out {n_dropped} variants ({n_pos_dropped} positives) with consequence_group=='splicing' AND exon_dist > 30")

V_m_iter22 = run("B. iter 22 final (refined tss bins + splice filter)",
                  TSS_BIN_EDGES_NEW, V_filtered)
report_full_leakage(V_m_iter22, "iter 22 final")


# ---- Side-by-side per-subset ----
print(f"\n\n========== Per-subset retention comparison ==========")
print(f"  {'subset':<35s}  {'iter11':>6s}  {'iter22':>6s}  {'Δ':>5s}")
for subset in sorted(V_m_iter11["consequence_group"].unique().to_list()):
    n11 = V_m_iter11.filter((pl.col("consequence_group") == subset) & pl.col("label")).height
    n22 = V_m_iter22.filter((pl.col("consequence_group") == subset) & pl.col("label")).height
    print(f"  {subset:<35s}  {n11:>6d}  {n22:>6d}  {n22-n11:>+5d}")
