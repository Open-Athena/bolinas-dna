"""Iter 11 — simplified subset-conditional bins (drop unreachable outer bins).

Refines iter 10 by removing redundancy:
- tss_dist bins applied ONLY to tss_proximal (was: tss_proximal + distal).
  distal had no leak in iter 10 — binning it just fragmented w/o benefit.
- Edges [0, 50, 200, 500, 1000] — 4 bins, no "1000+" outer bin.
  consequence_final == tss_proximal is bounded by tss_proximal_dist=1000 by
  definition (set in trait_intervals.add_tss), so the outer bin is unreachable
  for any in-subset variant (pos or neg).

- exon_dist bins applied ONLY to splicing (same as iter 10).
- Edges [0, 5, 20, 30] — 3 bins, no "30+" outer bin.
  exon_proximal is bounded by exon_proximal_dist=30 by definition. VEP splice
  categories aren't strictly bounded — empirical check inline below.

Bin function uses 'OOR' as fallback (distinct from any real bin) so any
out-of-range variant is safely isolated rather than silently lumped.

Compare against iter 10 (with outer bins + tss_dist applied to distal too).
"""
import polars as pl
from sklearn.metrics import roc_auc_score

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"
TSS_BIN_EDGES = [0, 50, 200, 500, 1000]   # 4 bins; no "1000+" outer bin
EXON_BIN_EDGES = [0, 5, 20, 30]            # 3 bins; no "30+" outer bin


def add_bin_label(V: pl.DataFrame, feature: str, edges: list[float], col: str) -> pl.DataFrame:
    """Bin into len(edges)-1 buckets. Out-of-range values get 'OOR' (distinct from any real bin)."""
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


print("Loading mendelian dataset_all...", flush=True)
V = pl.read_parquet(
    f"{BASE}/mendelian_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist"],
)
print(f"  {V.height} rows", flush=True)

# ---- Empirical check: are the caps actually never exceeded in-subset? ----
print("\n========== Empirical check: in-subset variants exceeding caps? ==========")
tss_check = V.filter(
    (pl.col("consequence_group") == "tss_proximal") & (pl.col("tss_dist") > 1000)
).group_by("label").agg(pl.len().alias("n"))
print(f"  tss_proximal variants with tss_dist > 1000:")
print(f"    {tss_check if tss_check.height else 'NONE — outer bin truly unreachable'}")

exon_check = V.filter(
    (pl.col("consequence_group") == "splicing") & (pl.col("exon_dist") > 30)
).group_by(["label", "consequence_final"]).agg(pl.len().alias("n")).sort(["label", "n"], descending=[True, True])
print(f"  splicing variants with exon_dist > 30:")
print(f"    {exon_check if exon_check.height else 'NONE — outer bin truly unreachable'}")

# ---- Subset-conditional bin labels ----
V = add_bin_label(V, "tss_dist", TSS_BIN_EDGES, "_tss_full")
V = add_bin_label(V, "exon_dist", EXON_BIN_EDGES, "_exon_full")
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

# Bin distribution among positives, by consequence_group
print(f"\nPositives' tss_dist_bin distribution by subset:")
print(
    V.filter(pl.col("label"))
    .group_by(["consequence_group", "tss_dist_bin"])
    .agg(pl.len())
    .pivot(values="len", index="consequence_group", on="tss_dist_bin")
    .sort("consequence_group")
)
print(f"\nPositives' exon_dist_bin distribution by subset:")
print(
    V.filter(pl.col("label"))
    .group_by(["consequence_group", "exon_dist_bin"])
    .agg(pl.len())
    .pivot(values="len", index="consequence_group", on="exon_dist_bin")
    .sort("consequence_group")
)

V_m = match_features(
    V.filter(pl.col("label")),
    V.filter(~pl.col("label")),
    ["tss_dist", "exon_dist"],  # keep continuous matching
    ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
     "tss_dist_bin", "exon_dist_bin"],
    k=1,
)
pos = V_m.filter(pl.col("label"))
print(f"\nTotal positives kept: {pos.height}/9767 = {100*pos.height/9767:.0f}%", flush=True)

print("\nPer-subset retention (vs iter 10 / production):")
prod_counts = {
    "missense_variant": 8778, "splicing": 247, "tss_proximal": 233,
    "5_prime_UTR_variant": 152, "distal": 118, "non_coding_transcript_exon_variant": 99,
    "3_prime_UTR_variant": 85, "synonymous_variant": 54, "mature_miRNA_variant": 1,
}
iter10_counts = {
    "missense_variant": 8778, "splicing": 163, "tss_proximal": 117,
    "5_prime_UTR_variant": 152, "distal": 116, "non_coding_transcript_exon_variant": 99,
    "3_prime_UTR_variant": 85, "synonymous_variant": 54, "mature_miRNA_variant": 1,
}
print(f"  {'subset':<40s}  {'iter11':>6s}  {'iter10':>6s}  {'prod':>6s}")
for subset, prod_n in prod_counts.items():
    sub_pos = pos.filter(pl.col("consequence_group") == subset).height
    print(f"  {subset:<40s}  {sub_pos:>6d}  {iter10_counts.get(subset, 0):>6d}  {prod_n:>6d}")

print("\nFull leakage table:")
report_leakage(V_m, ["tss_dist", "exon_dist"])
