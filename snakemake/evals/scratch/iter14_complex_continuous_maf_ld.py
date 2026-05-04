"""Iter 14 — complex_traits with discrete + continuous for MAF and ld_score.

Following the iter-11 pattern (bin AS WELL AS continuous) for MAF and ld_score:
- Continuous matching: [tss_dist, exon_dist, MAF, ld_score]  (production)
- Categorical bins (same as iter 13):
    - tss_dist_bin for tss_proximal, edges [0, 50, 200, 500, 1000], OOR
    - exon_dist_bin for splicing, edges [0, 5, 20, 30], OOR
    - MAF_bin edges [0.0, 0.005, 0.02, 0.05, 0.2, 0.5]
    - ld_score_bin edges [0.0, 1.0, 5.0, 20.0, 1e6]

iter 13 dropped MAF/ld_score from the continuous list (only kept them as bins).
This adds them back, mirroring iter 11's design choice for tss/exon.

Hypothesis: continuous + bins should give similar leak fix to bin-only, and
might recover some tss_proximal positives lost in iter 13 (since the gene-
matching can find a better neighbor when continuous distance is the tie-breaker
within a bin).
"""
import polars as pl
from sklearn.metrics import roc_auc_score

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"

MAF_BIN_EDGES = [0.0, 0.005, 0.02, 0.05, 0.2, 0.5]
LD_BIN_EDGES = [0.0, 1.0, 5.0, 20.0, 1e6]
TSS_BIN_EDGES = [0, 50, 200, 500, 1000]
EXON_BIN_EDGES = [0, 5, 20, 30]


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
n_total = V.filter(pl.col("label")).height
print(f"  {V.height} rows; {n_total} positives", flush=True)

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

# ---- Iter 14: bins + continuous on ALL 4 features ----
print("\n========== iter 14: bins + continuous on [tss_dist, exon_dist, MAF, ld_score] ==========")
V_m = match_features(
    V.filter(pl.col("label")),
    V.filter(~pl.col("label")),
    ["tss_dist", "exon_dist", "MAF", "ld_score"],   # ALL 4 continuous
    ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
     "MAF_bin", "ld_score_bin", "tss_dist_bin", "exon_dist_bin"],
    k=1,
)
pos = V_m.filter(pl.col("label"))
print(f"\nTotal positives kept: {pos.height}/{n_total} = {100*pos.height/n_total:.0f}%", flush=True)

print("\nPer-subset retention:")
print(f"  {'subset':<40s}  {'iter14':>6s}")
for subset in sorted(pos["consequence_group"].unique().to_list()):
    sub_pos = pos.filter(pl.col("consequence_group") == subset).height
    print(f"  {subset:<40s}  {sub_pos:>6d}")

print("\nFull leakage table:")
report_leakage(V_m, ["tss_dist", "exon_dist", "MAF", "ld_score"])

# ---- Iter 13 reference: bins ONLY (no MAF/ld_score continuous) ----
print("\n\n========== iter 13 reference (bins only, no MAF/ld_score continuous) ==========")
V_m_ref = match_features(
    V.filter(pl.col("label")),
    V.filter(~pl.col("label")),
    ["tss_dist", "exon_dist"],   # only 2 continuous
    ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
     "MAF_bin", "ld_score_bin", "tss_dist_bin", "exon_dist_bin"],
    k=1,
)
pos_ref = V_m_ref.filter(pl.col("label"))
print(f"\nTotal positives kept: {pos_ref.height}/{n_total} = {100*pos_ref.height/n_total:.0f}%", flush=True)
print(f"  {'subset':<40s}  {'iter13':>6s}")
for subset in sorted(pos_ref["consequence_group"].unique().to_list()):
    sub_pos = pos_ref.filter(pl.col("consequence_group") == subset).height
    print(f"  {subset:<40s}  {sub_pos:>6d}")
print("\nFull leakage table (iter 13):")
report_leakage(V_m_ref, ["tss_dist", "exon_dist", "MAF", "ld_score"])
