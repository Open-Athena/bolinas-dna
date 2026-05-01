"""For each (dataset, subset, matched continuous feature) report AUROC and
mean pairwise accuracy (within match_groups). Both should be ~0.5 if
matching balanced the feature inside each subset; deviation flags leakage.

Reads the produced parquets from the cluster's local cache OR S3:
  s3://oa-bolinas/snakemake/evals/results/dataset_unsplit/{mendelian,complex}_traits.parquet
"""
import polars as pl
from sklearn.metrics import roc_auc_score


PATHS = {
    "mendelian_traits": "s3://oa-bolinas/snakemake/evals/results/dataset_unsplit/mendelian_traits.parquet",
    "complex_traits": "s3://oa-bolinas/snakemake/evals/results/dataset_unsplit/complex_traits.parquet",
}
FEATURES = {
    "mendelian_traits": ["tss_dist", "exon_dist"],
    "complex_traits": ["tss_dist", "exon_dist", "MAF", "ld_score"],
}


def pairwise_acc(V: pl.DataFrame, feature: str) -> tuple[float, int]:
    """Per match_group, fraction with feature(pos) > feature(neg); ties = 0.5."""
    pos = (
        V.filter(pl.col("label"))
        .select(["match_group", feature])
        .rename({feature: "x_pos"})
    )
    neg = (
        V.filter(~pl.col("label"))
        .select(["match_group", feature])
        .rename({feature: "x_neg"})
    )
    paired = pos.join(neg, on="match_group", how="inner")
    n = paired.height
    if n == 0:
        return float("nan"), 0
    diff = (paired["x_pos"] - paired["x_neg"]).to_numpy()
    wins = (diff > 0).sum()
    ties = (diff == 0).sum()
    return float((wins + 0.5 * ties) / n), int(n)


def report(name: str) -> None:
    print(f"\n========== {name} ==========")
    V = pl.read_parquet(PATHS[name])
    for subset in sorted(V["consequence_group"].unique().to_list()):
        sub = V.filter(pl.col("consequence_group") == subset)
        n_pos = sub.filter(pl.col("label")).height
        n_neg = sub.filter(~pl.col("label")).height
        if n_pos == 0 or n_neg == 0:
            continue
        labels = sub["label"].cast(pl.Int8).to_numpy()
        parts = [f"{subset:38s}  n={n_pos:4d}"]
        for feat in FEATURES[name]:
            scores = sub[feat].to_numpy()
            try:
                auroc = roc_auc_score(labels, scores)
            except Exception:
                auroc = float("nan")
            acc, _ = pairwise_acc(sub, feat)
            parts.append(f"{feat}: AUROC={auroc:.3f} pacc={acc:.3f}")
        print("  " + "  |  ".join(parts))


for name in PATHS:
    report(name)
