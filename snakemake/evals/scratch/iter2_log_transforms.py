"""Iter 2 — Direction A: log10 transforms on continuous features before scaling.

The MAF distribution and tss/exon distance distributions are heavy-tailed.
A log transform compresses the tail and could change relative weights when
the matcher computes Euclidean distance after RobustScaler.

Tries log10(x + 1) for distances and log10(x + 1e-6) for MAF.
"""
import polars as pl
import numpy as np
from sklearn.metrics import roc_auc_score

from bolinas.evals.matching import match_features


BASE = "s3://oa-bolinas/snakemake/evals/results"


def pairwise_acc(V: pl.DataFrame, feat: str) -> float:
    pos = V.filter(pl.col("label")).select(["match_group", feat]).rename({feat: "p"})
    neg = V.filter(~pl.col("label")).select(["match_group", feat]).rename({feat: "n"})
    paired = pos.join(neg, on="match_group", how="inner")
    if paired.height == 0:
        return float("nan")
    diff = (paired["p"] - paired["n"]).to_numpy()
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / paired.height)


def run(name: str, dataset_all_path: str, log_specs: dict[str, str]) -> None:
    """log_specs maps feature -> "dist" or "maf" indicating transform type."""
    print(f"\n========== {name} (log10 + global RobustScaler) ==========", flush=True)
    cols = [
        "chrom", "pos", "ref", "alt", "label",
        "consequence_final", "consequence_group",
        "tss_closest_gene_id", "exon_closest_gene_id",
        *log_specs.keys(),
    ]
    V_all = pl.read_parquet(dataset_all_path, columns=cols)
    print(f"  read dataset_all: {V_all.height} rows", flush=True)

    log_cols = []
    for feat, kind in log_specs.items():
        log_col = f"{feat}_log"
        log_cols.append(log_col)
        if kind == "dist":
            V_all = V_all.with_columns(pl.col(feat).log(base=10).alias(log_col))
        elif kind == "maf":
            V_all = V_all.with_columns(
                ((pl.col(feat) + 1e-6).log(base=10)).alias(log_col)
            )
        else:
            raise ValueError(kind)
    # Distance log will produce -inf for pos=0; replace with -1 (one bp).
    V_all = V_all.with_columns(
        *[pl.when(pl.col(c).is_infinite()).then(-1.0).otherwise(pl.col(c)).alias(c) for c in log_cols]
    )

    V_m = match_features(
        V_all.filter(pl.col("label")),
        V_all.filter(~pl.col("label")),
        log_cols,
        ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id"],
        k=1,
        scale=True,  # global RobustScaler on log-transformed features
    )
    pos = V_m.filter(pl.col("label"))
    print(f"  total positives: {pos.height}", flush=True)

    # report on RAW features (not log) for comparability with prior iters
    raw_features = list(log_specs.keys())
    print(f"  {'subset':<40s} {'n':>5s}  " + "  ".join(f"{f}: AUROC/pacc" for f in raw_features))
    for subset in sorted(V_m["consequence_group"].unique().to_list()):
        sub = V_m.filter(pl.col("consequence_group") == subset)
        n_pos = sub.filter(pl.col("label")).height
        if n_pos == 0:
            continue
        labels = sub["label"].cast(pl.Int8).to_numpy()
        parts = []
        for f in raw_features:
            try:
                auroc = roc_auc_score(labels, sub[f].to_numpy())
            except Exception:
                auroc = float("nan")
            pacc = pairwise_acc(sub, f)
            parts.append(f"{auroc:.3f}/{pacc:.3f}")
        print(f"  {subset:<40s} {n_pos:>5d}  " + "  ".join(parts))


run(
    "mendelian_traits",
    f"{BASE}/mendelian_traits/dataset_all.parquet",
    {"tss_dist": "dist", "exon_dist": "dist"},
)
run(
    "complex_traits",
    f"{BASE}/complex_traits/dataset_all.parquet",
    {"tss_dist": "dist", "exon_dist": "dist", "MAF": "maf", "ld_score": "dist"},
)
