"""Iter 7 — combined best-design candidate.

Mendelian: sweep tss_proximal_dist ∈ {100, 150, 200, 250, 1000} bp.
Complex: MAF 5 bins + ld_score 4 bins (best from iter 6) + same gene-matching.

Goal: report the full leakage table and per-subset retention for both
datasets at the candidate "next-production" parameters, so we can pick.
"""
import polars as pl
import yaml
from sklearn.metrics import roc_auc_score

from bolinas.evals.matching import match_features
from bolinas.evals.trait_intervals import build_dataset


BASE = "s3://oa-bolinas/snakemake/evals/results"


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


with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)


# ============= Mendelian: tss_proximal_dist sweep =============
print("============= MENDELIAN: tss_proximal_dist sweep =============", flush=True)
positives_m = (
    pl.read_parquet(f"{BASE}/mendelian_traits/positives.parquet")
    .with_columns(label=pl.lit(True))
)
common_m = (
    pl.read_parquet(f"{BASE}/gnomad/common.parquet")
    .with_columns(label=pl.lit(False))
)
exon = pl.read_parquet(f"{BASE}/intervals/exon.parquet")
tss = pl.read_parquet(f"{BASE}/intervals/tss.parquet")

for tss_prox in [1000, 250, 200, 150, 100]:
    print(f"\n--- tss_proximal_dist = {tss_prox} bp ---", flush=True)
    V = pl.concat([positives_m, common_m], how="diagonal_relaxed")
    V = build_dataset(
        V, exon, tss,
        cfg["exclude_consequences"],
        cfg["exon_proximal_dist"],
        tss_prox,
        cfg["consequence_groups"],
    )
    V_m = match_features(
        V.filter(pl.col("label")),
        V.filter(~pl.col("label")),
        ["tss_dist", "exon_dist"],
        ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id"],
        k=1,
    )
    pos = V_m.filter(pl.col("label"))
    print(f"  total positives: {pos.height}", flush=True)
    # report just the tss_proximal subset since that's the one that changes
    sub = V_m.filter(pl.col("consequence_group") == "tss_proximal")
    if sub.filter(pl.col("label")).height > 0:
        report_leakage(sub, ["tss_dist", "exon_dist"])
    # also report distal which gains members as window tightens
    sub2 = V_m.filter(pl.col("consequence_group") == "distal")
    if sub2.filter(pl.col("label")).height > 0:
        report_leakage(sub2, ["tss_dist", "exon_dist"])


# ============= Complex: combined best (MAF 5-bin + ld_score 4-bin) =============
print("\n\n============= COMPLEX: best-candidate (MAF 5 bins + ld_score 4 bins) =============", flush=True)
V_c = pl.read_parquet(
    f"{BASE}/complex_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label", "consequence_final",
             "consequence_group", "tss_closest_gene_id", "exon_closest_gene_id",
             "tss_dist", "exon_dist", "MAF", "ld_score"],
)
V_c = add_bin(V_c, "MAF", [0.0, 0.005, 0.02, 0.05, 0.2, 0.5], "MAF_bin")
V_c = add_bin(V_c, "ld_score", [0.0, 1.0, 5.0, 20.0, 1e6], "ld_score_bin")
V_m_c = match_features(
    V_c.filter(pl.col("label")),
    V_c.filter(~pl.col("label")),
    ["tss_dist", "exon_dist"],
    ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id",
     "MAF_bin", "ld_score_bin"],
    k=1,
)
pos_c = V_m_c.filter(pl.col("label"))
print(f"\n  total positives: {pos_c.height}/2066 = {100*pos_c.height/2066:.0f}%", flush=True)
print(f"  full per-subset leakage (vs production):", flush=True)
report_leakage(V_m_c, ["tss_dist", "exon_dist", "MAF", "ld_score"])
