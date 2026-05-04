"""Same leakage metrics as scratch/leakage.py but break down by consequence_final
(not just consequence_group). Especially useful for splicing, which lumps
5 splice subtypes that the matcher already keeps distinct.

Reads dataset_unsplit/{mendelian,complex}_traits.parquet from S3.
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


def pairwise_acc(V: pl.DataFrame, feat: str) -> tuple[float, int]:
    pos = V.filter(pl.col("label")).select(["match_group", feat]).rename({feat: "p"})
    neg = V.filter(~pl.col("label")).select(["match_group", feat]).rename({feat: "n"})
    paired = pos.join(neg, on="match_group", how="inner")
    n = paired.height
    if n == 0:
        return float("nan"), 0
    diff = (paired["p"] - paired["n"]).to_numpy()
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / n), n


def report(name: str) -> None:
    print(f"\n========== {name} (by consequence_final) ==========")
    V = pl.read_parquet(PATHS[name])
    finals = sorted(V["consequence_final"].unique().to_list())
    for cf in finals:
        sub = V.filter(pl.col("consequence_final") == cf)
        n_pos = sub.filter(pl.col("label")).height
        n_neg = sub.filter(~pl.col("label")).height
        if n_pos == 0 or n_neg == 0:
            continue
        labels = sub["label"].cast(pl.Int8).to_numpy()
        cg = sub["consequence_group"].unique().to_list()
        cg_str = cg[0] if len(cg) == 1 else "/".join(cg)
        parts = [f"{cf:38s}  ({cg_str:18s})  n={n_pos:5d}"]
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
