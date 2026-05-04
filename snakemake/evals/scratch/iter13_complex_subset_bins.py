"""Iter 13 — apply iter-11 design (subset-conditional tss/exon bins) to complex_traits.

Builds on iter 7's complex_traits design (MAF 5 bins + ld_score 4 bins) by ADDING
iter 11's subset-conditional bins:
- tss_dist_bin for consequence_group == tss_proximal only (else "NA"),
  edges [0, 50, 200, 500, 1000], OOR fallback.
- exon_dist_bin for consequence_group == splicing only (else "NA"),
  edges [0, 5, 20, 30], OOR fallback.

Continuous matching kept on [tss_dist, exon_dist] (same as iter 7 — MAF/ld_score
are binned-only, since the production-style continuous wasn't tight enough).

Goal: see if iter 11's design transfers to complex without breaking the
MAF/ld_score leak fix; check retention + per-subset leakage including OOR sizes.
"""
import polars as pl
from sklearn.metrics import roc_auc_score

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"

# Iter 7's best for complex
MAF_BIN_EDGES = [0.0, 0.005, 0.02, 0.05, 0.2, 0.5]    # 5 bins
LD_BIN_EDGES = [0.0, 1.0, 5.0, 20.0, 1e6]              # 4 bins

# Iter 11's design for tss/exon (capped at the proximal threshold; OOR fallback)
TSS_BIN_EDGES = [0, 50, 200, 500, 1000]                # 4 bins
EXON_BIN_EDGES = [0, 5, 20, 30]                        # 3 bins


def add_bin_open(V: pl.DataFrame, feature: str, edges: list[float], col: str) -> pl.DataFrame:
    """Bin with default 'b0' fallback (used for MAF, ld_score where coverage is full)."""
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
    """Bin with 'OOR' fallback (used for tss_dist, exon_dist subset-conditionally)."""
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


def pairwise_acc(V: pl.DataFrame, feat: str) -> float:
    pos = V.filter(pl.col("label")).select(["match_group", feat]).rename({feat: "p"})
    neg = V.filter(~pl.col("label")).select(["match_group", feat]).rename({feat: "n"})
    paired = pos.join(neg, on="match_group", how="inner")
    if paired.height == 0:
        return float("nan")
    diff = (paired["p"] - paired["n"]).to_numpy()
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / paired.height)


def report_leakage(V_m: pl.DataFrame, features: list[str]) -> None:
    print(f"  {'subset':<40s} {'n':>5s}  " + "  ".join(f"{f}: AUROC/pacc" for f in features))
    for subset in sorted(V_m["consequence_group"].unique().to_list()):
        sub = V_m.filter(pl.col("consequence_group") == subset)
        n_pos = sub.filter(pl.col("label")).height
        if n_pos == 0:
            continue
        labels = sub["label"].cast(pl.Int8).to_numpy()
        parts = []
        for f in features:
            try:
                auroc = roc_auc_score(labels, sub[f].to_numpy())
            except Exception:
                auroc = float("nan")
            pacc = pairwise_acc(sub, f)
            parts.append(f"{auroc:.3f}/{pacc:.3f}")
        print(f"  {subset:<40s} {n_pos:>5d}  " + "      ".join(parts))


print("Loading complex_traits dataset_all...", flush=True)
V = pl.read_parquet(
    f"{BASE}/complex_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist", "MAF", "ld_score"],
)
print(f"  {V.height} rows; {V.filter(pl.col('label')).height} positives", flush=True)

# ---- Empirical check: in-subset variants exceeding caps? ----
print("\n========== Empirical check: in-subset variants exceeding caps? ==========")
tss_check = V.filter(
    (pl.col("consequence_group") == "tss_proximal") & (pl.col("tss_dist") > 1000)
).group_by("label").agg(pl.len().alias("n"))
print(f"  tss_proximal w/ tss_dist > 1000:")
print(f"    {tss_check if tss_check.height else 'NONE'}")
exon_check = V.filter(
    (pl.col("consequence_group") == "splicing") & (pl.col("exon_dist") > 30)
).group_by(["label", "consequence_final"]).agg(pl.len().alias("n")).sort(["label", "n"], descending=[True, True])
print(f"  splicing w/ exon_dist > 30:")
print(f"    {exon_check if exon_check.height else 'NONE'}")

# ---- Build all bins ----
V = add_bin_open(V, "MAF", MAF_BIN_EDGES, "MAF_bin")
V = add_bin_open(V, "ld_score", LD_BIN_EDGES, "ld_score_bin")
V = add_bin_oor(V, "tss_dist", TSS_BIN_EDGES, "_tss_full")
V = add_bin_oor(V, "exon_dist", EXON_BIN_EDGES, "_exon_full")
V = V.with_columns(
    pl.when(pl.col("consequence_group") == "tss_proximal")
    .then(pl.col("_tss_full"))
    .otherwise(pl.lit("NA"))
    .alias("tss_dist_bin"),
    pl.when(pl.col("consequence_group") == "splicing")
    .then(pl.col("_exon_full"))
    .otherwise(pl.lit("NA"))
    .alias("exon_dist_bin"),
)

# OOR / NA distribution among positives, by consequence_group
print("\nPositives' tss_dist_bin distribution:")
print(
    V.filter(pl.col("label"))
    .group_by(["consequence_group", "tss_dist_bin"])
    .agg(pl.len())
    .pivot(values="len", index="consequence_group", on="tss_dist_bin")
    .sort("consequence_group")
)
print("\nPositives' exon_dist_bin distribution:")
print(
    V.filter(pl.col("label"))
    .group_by(["consequence_group", "exon_dist_bin"])
    .agg(pl.len())
    .pivot(values="len", index="consequence_group", on="exon_dist_bin")
    .sort("consequence_group")
)

# ---- Match: iter 7 design + iter 11 bins ----
print("\n========== iter 13: iter 7 + iter 11 subset-conditional bins ==========")
V_m = match_features(
    V.filter(pl.col("label")),
    V.filter(~pl.col("label")),
    ["tss_dist", "exon_dist"],   # continuous (same as iter 7)
    ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
     "MAF_bin", "ld_score_bin", "tss_dist_bin", "exon_dist_bin"],
    k=1,
)
pos = V_m.filter(pl.col("label"))
n_total = V.filter(pl.col("label")).height
print(f"\nTotal positives kept: {pos.height}/{n_total} = {100*pos.height/n_total:.0f}%", flush=True)

print("\nPer-subset retention (vs iter 7 baseline / production):")
# Production = iter 7 baseline = 1520 from iter 6 (5 MAF + 4 ld_score)
# True production has 2066 positives total (no MAF/ld_score binning)
print(f"  {'subset':<40s}  {'iter13':>6s}")
for subset in sorted(pos["consequence_group"].unique().to_list()):
    sub_pos = pos.filter(pl.col("consequence_group") == subset).height
    print(f"  {subset:<40s}  {sub_pos:>6d}")

print("\nFull leakage table:")
report_leakage(V_m, ["tss_dist", "exon_dist", "MAF", "ld_score"])

# ---- Reference: iter 7 design WITHOUT iter 11 bins (for comparison) ----
print("\n\n========== iter 7 baseline (no tss/exon bins) for comparison ==========")
V_m_ref = match_features(
    V.filter(pl.col("label")),
    V.filter(~pl.col("label")),
    ["tss_dist", "exon_dist"],
    ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
     "MAF_bin", "ld_score_bin"],
    k=1,
)
pos_ref = V_m_ref.filter(pl.col("label"))
print(f"\nTotal positives kept: {pos_ref.height}/{n_total} = {100*pos_ref.height/n_total:.0f}%", flush=True)
print(f"  {'subset':<40s}  {'iter7_ref':>9s}")
for subset in sorted(pos_ref["consequence_group"].unique().to_list()):
    sub_pos = pos_ref.filter(pl.col("consequence_group") == subset).height
    print(f"  {subset:<40s}  {sub_pos:>9d}")
print("\nFull leakage table (iter 7 ref):")
report_leakage(V_m_ref, ["tss_dist", "exon_dist", "MAF", "ld_score"])
