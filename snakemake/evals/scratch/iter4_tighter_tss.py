"""Iter 4 — cross-cutting: tighter tss_proximal_dist for mendelian.

Currently tss_proximal_dist=1000bp. The window-edge artifact in
mendelian's tss_proximal subset (positives near 1kb edge, negatives at TSS)
suggests a tighter window would give a more uniform distance distribution
inside the bucket. Try 100bp.

Re-runs build_dataset (mendelian positives + gnomAD common, AF>0.001) with
the new threshold, then matches.
"""
import polars as pl
import yaml
from sklearn.metrics import roc_auc_score

from bolinas.evals.matching import match_features
from bolinas.evals.trait_intervals import build_dataset


BASE = "s3://oa-bolinas/snakemake/evals/results"


def pairwise_acc(V: pl.DataFrame, feat: str) -> float:
    pos = V.filter(pl.col("label")).select(["match_group", feat]).rename({feat: "p"})
    neg = V.filter(~pl.col("label")).select(["match_group", feat]).rename({feat: "n"})
    paired = pos.join(neg, on="match_group", how="inner")
    if paired.height == 0:
        return float("nan")
    diff = (paired["p"] - paired["n"]).to_numpy()
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / paired.height)


with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

print("Loading inputs...", flush=True)
positives = (
    pl.read_parquet(f"{BASE}/mendelian_traits/positives.parquet")
    .with_columns(label=pl.lit(True))
)
common = (
    pl.read_parquet(f"{BASE}/gnomad/common.parquet")
    .with_columns(label=pl.lit(False))
)
exon = pl.read_parquet(f"{BASE}/intervals/exon.parquet")
tss = pl.read_parquet(f"{BASE}/intervals/tss.parquet")
print(f"  positives={positives.height} commons={common.height}", flush=True)

for tss_prox in [1000, 500, 250, 100]:
    print(f"\n========== tss_proximal_dist={tss_prox} bp ==========", flush=True)
    V = pl.concat([positives, common], how="diagonal_relaxed")
    V = build_dataset(
        V,
        exon,
        tss,
        cfg["exclude_consequences"],
        cfg["exon_proximal_dist"],
        tss_prox,
        cfg["consequence_groups"],
    )
    n_tss_prox_pos = V.filter(pl.col("label") & (pl.col("consequence_final") == "tss_proximal")).height
    print(f"  build_dataset: rows={V.height}, pos={V.filter(pl.col('label')).height}, tss_proximal_pos={n_tss_prox_pos}", flush=True)
    V_m = match_features(
        V.filter(pl.col("label")),
        V.filter(~pl.col("label")),
        ["tss_dist", "exon_dist"],
        ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id"],
        k=1,
    )
    pos = V_m.filter(pl.col("label"))
    print(f"  matched pos: {pos.height}", flush=True)
    # leakage on tss_proximal subset
    tss_subset = V_m.filter(pl.col("consequence_group") == "tss_proximal")
    if tss_subset.height > 0:
        n_tss = tss_subset.filter(pl.col("label")).height
        labels = tss_subset["label"].cast(pl.Int8).to_numpy()
        for f in ["tss_dist", "exon_dist"]:
            try:
                auroc = roc_auc_score(labels, tss_subset[f].to_numpy())
            except Exception:
                auroc = float("nan")
            pacc = pairwise_acc(tss_subset, f)
            print(f"    tss_proximal n={n_tss} {f}: AUROC={auroc:.3f} pacc={pacc:.3f}")
    # all subsets summary
    print(f"  per-subset positive counts:")
    counts = pos.group_by("consequence_group").agg(pl.len()).sort("len", descending=True)
    for row in counts.iter_rows(named=True):
        print(f"    {row['consequence_group']:<40s}  {row['len']:>5d}")
