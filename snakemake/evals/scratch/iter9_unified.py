"""Iter 9 — unified rule: bin every matched continuous feature in BOTH datasets.

Mendelian: tss_dist_bin + exon_dist_bin (already studied in iter 8 var A).
Complex: MAF_bin + ld_score_bin + tss_dist_bin + exon_dist_bin (unified version).

Goal: confirm the unified rule's retention on complex_traits doesn't collapse,
and produce the full leakage table for both datasets at the same matching
philosophy: "no continuous matching, bin everything".
"""
import polars as pl
from sklearn.metrics import roc_auc_score

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"
TSS_BIN_EDGES = [0, 50, 200, 500, 1000, 5000, 50_000, 500_000, 1e10]
EXON_BIN_EDGES = [0, 5, 20, 50, 200, 1000, 1e10]
MAF_BIN_EDGES = [0.0, 0.005, 0.02, 0.05, 0.2, 0.5]
LD_BIN_EDGES = [0.0, 1.0, 5.0, 20.0, 1e6]


def add_bin(V: pl.DataFrame, feature: str, edges: list[float], col: str) -> pl.DataFrame:
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


GENE_CATS = ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id"]


# ============= MENDELIAN: bin tss_dist + exon_dist =============
print("============= MENDELIAN: unified rule (tss_dist_bin + exon_dist_bin) =============", flush=True)
V_m = pl.read_parquet(
    f"{BASE}/mendelian_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist"],
)
print(f"  {V_m.height} rows", flush=True)
V_m = add_bin(V_m, "tss_dist", TSS_BIN_EDGES, "tss_dist_bin")
V_m = add_bin(V_m, "exon_dist", EXON_BIN_EDGES, "exon_dist_bin")

V_mat = match_features(
    V_m.filter(pl.col("label")),
    V_m.filter(~pl.col("label")),
    [],  # no continuous
    GENE_CATS + ["tss_dist_bin", "exon_dist_bin"],
    k=1,
)
pos_m = V_mat.filter(pl.col("label"))
print(f"\n  total positives kept: {pos_m.height}/9767 = {100*pos_m.height/9767:.0f}%", flush=True)
print("  per-subset positive counts vs production:")
prod_counts_m = {
    "missense_variant": 8778, "splicing": 247, "tss_proximal": 233,
    "5_prime_UTR_variant": 152, "distal": 118, "non_coding_transcript_exon_variant": 99,
    "3_prime_UTR_variant": 85, "synonymous_variant": 54, "mature_miRNA_variant": 1,
}
for subset, prod_n in prod_counts_m.items():
    sub_pos = pos_m.filter(pl.col("consequence_group") == subset).height
    pct = 100 * sub_pos / prod_n if prod_n else 0
    print(f"    {subset:<40s}  {sub_pos:>5d} / {prod_n:<5d}  ({pct:.0f}%)")
print("\n  full leakage table:")
report_leakage(V_mat, ["tss_dist", "exon_dist"])


# ============= COMPLEX: bin all four =============
print("\n\n============= COMPLEX: unified rule (MAF_bin + ld_score_bin + tss_dist_bin + exon_dist_bin) =============", flush=True)
V_c = pl.read_parquet(
    f"{BASE}/complex_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist", "MAF", "ld_score"],
)
print(f"  {V_c.height} rows", flush=True)
V_c = add_bin(V_c, "tss_dist", TSS_BIN_EDGES, "tss_dist_bin")
V_c = add_bin(V_c, "exon_dist", EXON_BIN_EDGES, "exon_dist_bin")
V_c = add_bin(V_c, "MAF", MAF_BIN_EDGES, "MAF_bin")
V_c = add_bin(V_c, "ld_score", LD_BIN_EDGES, "ld_score_bin")

V_mat_c = match_features(
    V_c.filter(pl.col("label")),
    V_c.filter(~pl.col("label")),
    [],  # no continuous
    GENE_CATS + ["tss_dist_bin", "exon_dist_bin", "MAF_bin", "ld_score_bin"],
    k=1,
)
pos_c = V_mat_c.filter(pl.col("label"))
print(f"\n  total positives kept: {pos_c.height}/2066 = {100*pos_c.height/2066:.0f}%", flush=True)
print("  per-subset positive counts vs production:")
prod_counts_c = {
    "distal": 1267, "missense_variant": 413, "tss_proximal": 147,
    "non_coding_transcript_exon_variant": 73, "3_prime_UTR_variant": 69,
    "5_prime_UTR_variant": 45, "synonymous_variant": 33, "splicing": 19,
}
for subset, prod_n in prod_counts_c.items():
    sub_pos = pos_c.filter(pl.col("consequence_group") == subset).height
    pct = 100 * sub_pos / prod_n if prod_n else 0
    print(f"    {subset:<40s}  {sub_pos:>5d} / {prod_n:<5d}  ({pct:.0f}%)")
print("\n  full leakage table:")
report_leakage(V_mat_c, ["tss_dist", "exon_dist", "MAF", "ld_score"])
