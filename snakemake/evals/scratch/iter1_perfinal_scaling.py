"""Iter 1 — Direction A: scale tss_dist/exon_dist per consequence_final
before matching. Compare positive count + leakage to the global-scaling
production result.
"""
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from bolinas.evals.matching import match_features


def prescale_per_group(
    V: pl.DataFrame, features: list[str], group_col: str
) -> pl.DataFrame:
    """Add scaled columns f"{feat}_scaled" — RobustScaler fit per group on pos+neg combined."""
    out_chunks = []
    for grp_value in V[group_col].unique().to_list():
        sub = V.filter(pl.col(group_col) == grp_value)
        if sub.height < 2:
            # Not enough rows to fit RobustScaler; copy raw values.
            scaled_cols = {f"{f}_scaled": sub[f].cast(pl.Float64) for f in features}
        else:
            scaler = RobustScaler()
            arr = sub.select(features).to_pandas()
            scaler.fit(arr)
            transformed = scaler.transform(arr)
            scaled_cols = {
                f"{f}_scaled": pl.Series(transformed[:, i])
                for i, f in enumerate(features)
            }
        out_chunks.append(sub.with_columns(**scaled_cols))
    return pl.concat(out_chunks)


def pairwise_acc(V: pl.DataFrame, feat: str) -> float:
    pos = V.filter(pl.col("label")).select(["match_group", feat]).rename({feat: "p"})
    neg = V.filter(~pl.col("label")).select(["match_group", feat]).rename({feat: "n"})
    paired = pos.join(neg, on="match_group", how="inner")
    if paired.height == 0:
        return float("nan")
    diff = (paired["p"] - paired["n"]).to_numpy()
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / paired.height)


def run_match(name: str, dataset_all_path: str, features: list[str]) -> None:
    print(f"\n========== {name} (per-consequence_final scaling) ==========", flush=True)
    cols = ["chrom", "pos", "ref", "alt", "label", "consequence_final",
            "consequence_group", "tss_closest_gene_id", "exon_closest_gene_id",
            *features]
    V_all = pl.read_parquet(dataset_all_path, columns=cols)
    print(f"  read dataset_all: {V_all.height} rows", flush=True)
    V_pre = prescale_per_group(V_all, features, "consequence_final")
    print(f"  prescaled per consequence_final", flush=True)
    scaled_features = [f"{f}_scaled" for f in features]

    V_m = match_features(
        V_pre.filter(pl.col("label")),
        V_pre.filter(~pl.col("label")),
        scaled_features,
        ["chrom", "consequence_final", "tss_closest_gene_id", "exon_closest_gene_id"],
        k=1,
        scale=False,  # already pre-scaled
    )
    pos = V_m.filter(pl.col("label"))
    print(f"total positives: {pos.height}")

    print(f"{'subset':<40s}  {'n':>5s}  " + "  ".join(f"{f}: AUROC/pacc" for f in features))
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
        print(f"  {subset:<40s}  {n_pos:>5d}  " + "  ".join(parts))


BASE = "s3://oa-bolinas/snakemake/evals/results"
run_match("mendelian_traits", f"{BASE}/mendelian_traits/dataset_all.parquet", ["tss_dist", "exon_dist"])
run_match("complex_traits", f"{BASE}/complex_traits/dataset_all.parquet", ["tss_dist", "exon_dist", "MAF", "ld_score"])
