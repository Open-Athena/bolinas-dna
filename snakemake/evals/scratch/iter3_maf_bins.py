"""Iter 3 — Direction B: bin MAF for complex_traits and add as a categorical
matching feature.

Bin edges chosen to be tighter at the low end (where rare-variant
differences matter most biologically) and coarser at the upper end.
"""
import polars as pl
from sklearn.metrics import roc_auc_score

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"
DATASET_ALL = f"{BASE}/complex_traits/dataset_all.parquet"

# Tighter at the low end. MAF is in [0, 0.5].
MAF_BIN_EDGES = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
# Equivalent: 8 bins. Bin i = (edge[i], edge[i+1]].
N_BINS = len(MAF_BIN_EDGES) - 1


def add_maf_bin(V: pl.DataFrame) -> pl.DataFrame:
    """Add MAF_bin column (string, e.g. 'b3' for the 3rd bin)."""
    expr = pl.lit("b0")
    for i in range(N_BINS):
        lo, hi = MAF_BIN_EDGES[i], MAF_BIN_EDGES[i + 1]
        # last bin includes the upper edge
        cond = (
            (pl.col("MAF") > lo) & (pl.col("MAF") <= hi)
            if i < N_BINS - 1
            else ((pl.col("MAF") >= lo) & (pl.col("MAF") <= hi))
        )
        expr = pl.when(cond).then(pl.lit(f"b{i}")).otherwise(expr)
    return V.with_columns(MAF_bin=expr)


def pairwise_acc(V: pl.DataFrame, feat: str) -> float:
    pos = V.filter(pl.col("label")).select(["match_group", feat]).rename({feat: "p"})
    neg = V.filter(~pl.col("label")).select(["match_group", feat]).rename({feat: "n"})
    paired = pos.join(neg, on="match_group", how="inner")
    if paired.height == 0:
        return float("nan")
    diff = (paired["p"] - paired["n"]).to_numpy()
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / paired.height)


print(f"========== complex_traits (MAF binning, edges={MAF_BIN_EDGES}) ==========", flush=True)
cols = [
    "chrom", "pos", "ref", "alt", "label",
    "consequence_final", "consequence_group",
    "tss_closest_gene_id", "exon_closest_gene_id",
    "tss_dist", "exon_dist", "MAF", "ld_score",
]
V_all = pl.read_parquet(DATASET_ALL, columns=cols)
print(f"  read dataset_all: {V_all.height} rows", flush=True)
V_binned = add_maf_bin(V_all)
print(f"  MAF bin distribution:", flush=True)
print(V_binned.group_by("MAF_bin").agg(pl.len()).sort("MAF_bin"))

V_m = match_features(
    V_binned.filter(pl.col("label")),
    V_binned.filter(~pl.col("label")),
    # drop MAF from continuous, keep tss/exon_dist + ld_score; add MAF_bin to categorical
    ["tss_dist", "exon_dist", "ld_score"],
    [
        "chrom",
        "consequence_final",
        "tss_closest_gene_id",
        "exon_closest_gene_id",
        "MAF_bin",
    ],
    k=1,
)
pos = V_m.filter(pl.col("label"))
print(f"\n  total positives kept: {pos.height} (production: 2066)", flush=True)
print(f"  by subset:")
for subset in sorted(V_m["consequence_group"].unique().to_list()):
    sub = V_m.filter(pl.col("consequence_group") == subset)
    n_pos = sub.filter(pl.col("label")).height
    if n_pos == 0:
        continue
    labels = sub["label"].cast(pl.Int8).to_numpy()
    parts = []
    for f in ["tss_dist", "exon_dist", "MAF", "ld_score"]:
        try:
            auroc = roc_auc_score(labels, sub[f].to_numpy())
        except Exception:
            auroc = float("nan")
        pacc = pairwise_acc(sub, f)
        parts.append(f"{f}: {auroc:.3f}/{pacc:.3f}")
    print(f"    {subset:<40s}  n={n_pos:5d}  " + "  ".join(parts))
