"""Iter 8 — Direction B: bin tss_dist + exon_dist as categoricals for mendelian.

Don't change tss_proximal_dist (keep at 1000); let the bin-categorical
constrain the matching so tss_proximal positives at, say, 50bp pair with
negatives at 0-100bp instead of being free to pair anywhere in 0-1000bp.

Same idea for exon_dist (addresses splice_region/donor_region leak).
"""
import polars as pl
import yaml
from sklearn.metrics import roc_auc_score

from bolinas.evals.matching import match_features
from bolinas.evals.trait_intervals import build_dataset


BASE = "s3://oa-bolinas/snakemake/evals/results"

# Tighter at the low end (where biology cares); coarser as we go far.
TSS_BIN_EDGES = [0, 50, 200, 500, 1000, 5000, 50_000, 500_000, 1e10]
EXON_BIN_EDGES = [0, 5, 20, 50, 200, 1000, 1e10]


def add_bin(V: pl.DataFrame, feature: str, edges: list[float], col: str) -> pl.DataFrame:
    expr = pl.lit(f"b{len(edges) - 2}")  # default to last bin (catches >= last edge)
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


# Read mendelian dataset_all from S3 (production AF>0.001 build, tss_prox=1000)
print("Loading mendelian dataset_all...", flush=True)
V = pl.read_parquet(
    f"{BASE}/mendelian_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label",
             "consequence_final", "consequence_group",
             "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist"],
)
print(f"  {V.height} rows", flush=True)
V = add_bin(V, "tss_dist", TSS_BIN_EDGES, "tss_dist_bin")
V = add_bin(V, "exon_dist", EXON_BIN_EDGES, "exon_dist_bin")
print(f"\nBin distributions (positives only):")
print(V.filter(pl.col("label")).group_by("tss_dist_bin").agg(pl.len()).sort("tss_dist_bin"))
print(V.filter(pl.col("label")).group_by("exon_dist_bin").agg(pl.len()).sort("exon_dist_bin"))

# Three variants:
# A: bin both, drop both from continuous
# B: bin tss_dist only, keep exon_dist continuous
# C: bin exon_dist only, keep tss_dist continuous

VARIANTS = {
    "A: bin both": dict(cont=[], cats=["tss_dist_bin", "exon_dist_bin"]),
    "B: bin tss_dist only": dict(cont=["exon_dist"], cats=["tss_dist_bin"]),
    "C: bin exon_dist only": dict(cont=["tss_dist"], cats=["exon_dist_bin"]),
}

for name, cfg in VARIANTS.items():
    print(f"\n========== {name} ==========", flush=True)
    base_cat = ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id"]
    cont = cfg["cont"]
    cats = base_cat + cfg["cats"]
    V_m = match_features(
        V.filter(pl.col("label")),
        V.filter(~pl.col("label")),
        cont, cats, k=1,
    )
    pos = V_m.filter(pl.col("label"))
    print(f"  total positives kept: {pos.height}/9767 = {100*pos.height/9767:.0f}%", flush=True)
    report_leakage(V_m, ["tss_dist", "exon_dist"])
