"""Iter 6 — Direction B: sweep MAF bin schemes for complex_traits.

Iter 3 used 8 bins (edges [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
which was strict — kept only 1520/2066 (74%) and crashed splicing to n=2.
Try wider/coarser bin schemes and ld_score binning too.
"""
import polars as pl
from sklearn.metrics import roc_auc_score

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"
DATASET_ALL = f"{BASE}/complex_traits/dataset_all.parquet"

# Bin scheme variants
SCHEMES = {
    "iter3 (8 bins, fine low end)": {
        "MAF": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    },
    "5 bins (coarser)": {
        "MAF": [0.0, 0.005, 0.02, 0.05, 0.2, 0.5],
    },
    "4 bins (very coarse)": {
        "MAF": [0.0, 0.01, 0.05, 0.2, 0.5],
    },
    "3 bins": {
        "MAF": [0.0, 0.01, 0.1, 0.5],
    },
    "MAF 5 bins + ld_score 4 bins": {
        "MAF": [0.0, 0.005, 0.02, 0.05, 0.2, 0.5],
        "ld_score": [0.0, 1.0, 5.0, 20.0, 1e6],  # ld_score is heavy-tailed
    },
}


def add_bin(V: pl.DataFrame, feature: str, edges: list[float], col: str) -> pl.DataFrame:
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


def pairwise_acc(V: pl.DataFrame, feat: str) -> float:
    pos = V.filter(pl.col("label")).select(["match_group", feat]).rename({feat: "p"})
    neg = V.filter(~pl.col("label")).select(["match_group", feat]).rename({feat: "n"})
    paired = pos.join(neg, on="match_group", how="inner")
    if paired.height == 0:
        return float("nan")
    diff = (paired["p"] - paired["n"]).to_numpy()
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / paired.height)


print("Loading dataset_all...", flush=True)
V_all = pl.read_parquet(
    DATASET_ALL,
    columns=["chrom", "pos", "ref", "alt", "label", "consequence_final",
             "consequence_group", "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist", "MAF", "ld_score"],
)
print(f"  {V_all.height} rows\n", flush=True)


for scheme_name, bins in SCHEMES.items():
    print(f"========== {scheme_name} ==========", flush=True)
    V = V_all
    bin_cols = []
    for feat, edges in bins.items():
        col = f"{feat}_bin"
        V = add_bin(V, feat, edges, col)
        bin_cols.append(col)
    # Drop binned features from continuous matching, keep tss/exon_dist + remaining ones
    cont = ["tss_dist", "exon_dist"]
    if "MAF" not in bins:
        cont.append("MAF")
    if "ld_score" not in bins:
        cont.append("ld_score")
    cat = [
        "chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
        *bin_cols,
    ]
    V_m = match_features(
        V.filter(pl.col("label")),
        V.filter(~pl.col("label")),
        cont, cat, k=1,
    )
    pos = V_m.filter(pl.col("label"))
    print(f"  matched: {pos.height}/2066 = {100*pos.height/2066:.0f}%", flush=True)
    print(f"  {'subset':<40s} {'n':>5s}  MAF AUROC/pacc   ld_score AUROC/pacc")
    for subset in sorted(V_m["consequence_group"].unique().to_list()):
        sub = V_m.filter(pl.col("consequence_group") == subset)
        n_pos = sub.filter(pl.col("label")).height
        if n_pos == 0:
            continue
        labels = sub["label"].cast(pl.Int8).to_numpy()
        parts = []
        for f in ["MAF", "ld_score"]:
            try:
                auroc = roc_auc_score(labels, sub[f].to_numpy())
            except Exception:
                auroc = float("nan")
            pacc = pairwise_acc(sub, f)
            parts.append(f"{auroc:.3f}/{pacc:.3f}")
        print(f"  {subset:<40s} {n_pos:>5d}  " + "      ".join(parts))
    print()
