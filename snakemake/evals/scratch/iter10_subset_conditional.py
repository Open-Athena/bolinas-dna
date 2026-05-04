"""Iter 10 — subset-conditional binning for mendelian.

tss_dist_bin only varies for {distal, tss_proximal}; constant "NA" elsewhere.
exon_dist_bin only varies for splicing; constant "NA" elsewhere.

Goal: keep iter-8 variant A's leakage fix in tss_proximal and splicing while
not fragmenting missense / UTR / synonymous (where the binned feature is
biologically irrelevant).
"""
import polars as pl
from sklearn.metrics import roc_auc_score

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"
TSS_BIN_EDGES = [0, 50, 200, 500, 1000, 5000, 50_000, 500_000, 1e10]
EXON_BIN_EDGES = [0, 5, 20, 50, 200, 1000, 1e10]


def add_bin_label(V: pl.DataFrame, feature: str, edges: list[float], col: str) -> pl.DataFrame:
    expr = pl.lit(f"b{len(edges) - 2}")
    n = len(edges) - 1
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


print("Loading mendelian dataset_all...", flush=True)
V = pl.read_parquet(
    f"{BASE}/mendelian_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist"],
)
print(f"  {V.height} rows", flush=True)

# Compute the full bin labels first
V = add_bin_label(V, "tss_dist", TSS_BIN_EDGES, "_tss_dist_bin_full")
V = add_bin_label(V, "exon_dist", EXON_BIN_EDGES, "_exon_dist_bin_full")

# Subset-conditional bin: keep only for the relevant consequence_group, else "NA"
V = V.with_columns(
    pl.when(pl.col("consequence_group").is_in(["distal", "tss_proximal"]))
    .then(pl.col("_tss_dist_bin_full"))
    .otherwise(pl.lit("NA"))
    .alias("tss_dist_bin"),
    pl.when(pl.col("consequence_group") == "splicing")
    .then(pl.col("_exon_dist_bin_full"))
    .otherwise(pl.lit("NA"))
    .alias("exon_dist_bin"),
)

print("\n========== 10a: bins ADD to production matching (keep continuous) ==========", flush=True)
V_m = match_features(
    V.filter(pl.col("label")),
    V.filter(~pl.col("label")),
    ["tss_dist", "exon_dist"],  # keep production's continuous matching
    ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
     "tss_dist_bin", "exon_dist_bin"],
    k=1,
)
pos = V_m.filter(pl.col("label"))
print(f"Total positives kept: {pos.height}/9767 = {100*pos.height/9767:.0f}%", flush=True)

print("\nPer-subset retention (vs production / iter-8 unconditional):")
prod_counts = {
    "missense_variant": 8778, "splicing": 247, "tss_proximal": 233,
    "5_prime_UTR_variant": 152, "distal": 118, "non_coding_transcript_exon_variant": 99,
    "3_prime_UTR_variant": 85, "synonymous_variant": 54, "mature_miRNA_variant": 1,
}
iter8_counts = {
    "missense_variant": 6688, "splicing": 106, "tss_proximal": 98,
    "5_prime_UTR_variant": 108, "distal": 110, "non_coding_transcript_exon_variant": 49,
    "3_prime_UTR_variant": 80, "synonymous_variant": 43, "mature_miRNA_variant": 0,
}
print(f"  {'subset':<40s}  {'iter10':>6s}  {'iter8':>6s}  {'prod':>6s}")
for subset, prod_n in prod_counts.items():
    sub_pos = pos.filter(pl.col("consequence_group") == subset).height
    print(f"  {subset:<40s}  {sub_pos:>6d}  {iter8_counts.get(subset, 0):>6d}  {prod_n:>6d}")

print("\nFull leakage table:")
report_leakage(V_m, ["tss_dist", "exon_dist"])
