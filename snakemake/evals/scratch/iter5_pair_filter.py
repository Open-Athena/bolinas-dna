"""Iter 5 — cross-cutting: hard within-pair gap filter as a post-hoc step.

Take the production matched output and drop pairs whose pos vs neg differ
by more than X on a continuous feature. Sweep X to see the
leakage-vs-retention trade-off. Local only (works on the small produced
parquet files).
"""
import polars as pl
from sklearn.metrics import roc_auc_score


PATHS = {
    "mendelian_traits": "/tmp/mendelian_traits.parquet",
    "complex_traits": "/tmp/complex_traits.parquet",
}
# (feature, list of max-allowed within-pair gaps to try)
GAP_SWEEPS = {
    "mendelian_traits": [
        # tss_dist gaps in bp; positives' tss_dist range up to 100s of kb
        ("tss_dist", [10_000, 5_000, 2_000, 1_000, 500, 200, 100]),
        ("exon_dist", [10_000, 5_000, 2_000, 1_000, 500, 200, 100]),
    ],
    "complex_traits": [
        ("MAF", [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]),
        ("ld_score", [10, 5, 2, 1, 0.5]),
    ],
}


def pairwise_acc(V: pl.DataFrame, feat: str) -> float:
    pos = V.filter(pl.col("label")).select(["match_group", feat]).rename({feat: "p"})
    neg = V.filter(~pl.col("label")).select(["match_group", feat]).rename({feat: "n"})
    paired = pos.join(neg, on="match_group", how="inner")
    if paired.height == 0:
        return float("nan")
    diff = (paired["p"] - paired["n"]).to_numpy()
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / paired.height)


def filter_by_pair_gap(V: pl.DataFrame, feature: str, max_gap: float) -> pl.DataFrame:
    """Drop both rows of any match_group where |feature(pos) - feature(neg)| > max_gap."""
    pos = V.filter(pl.col("label")).select(["match_group", feature]).rename({feature: "p"})
    neg = V.filter(~pl.col("label")).select(["match_group", feature]).rename({feature: "n"})
    keep = (
        pos.join(neg, on="match_group", how="inner")
        .filter((pl.col("p") - pl.col("n")).abs() <= max_gap)
        .select("match_group")
    )
    return V.join(keep, on="match_group", how="inner")


def report(name: str) -> None:
    V = pl.read_parquet(PATHS[name])
    n_orig = V.filter(pl.col("label")).height
    print(f"\n========== {name} (n_pos baseline = {n_orig}) ==========", flush=True)
    for feature, gaps in GAP_SWEEPS[name]:
        print(f"\n  --- gap filter on {feature} ---", flush=True)
        # Baseline: AUROC + pacc on the full produced dataset
        labels0 = V["label"].cast(pl.Int8).to_numpy()
        try:
            auroc0 = roc_auc_score(labels0, V[feature].to_numpy())
        except Exception:
            auroc0 = float("nan")
        pacc0 = pairwise_acc(V, feature)
        print(f"    no filter:           n_pairs={n_orig:5d}   AUROC={auroc0:.3f}  pacc={pacc0:.3f}")
        for gap in gaps:
            Vf = filter_by_pair_gap(V, feature, gap)
            n_kept = Vf.filter(pl.col("label")).height
            if n_kept == 0:
                print(f"    max gap = {gap:>8g}: n_pairs={n_kept:5d}   (all filtered)")
                continue
            labels = Vf["label"].cast(pl.Int8).to_numpy()
            try:
                auroc = roc_auc_score(labels, Vf[feature].to_numpy())
            except Exception:
                auroc = float("nan")
            pacc = pairwise_acc(Vf, feature)
            pct = 100 * n_kept / n_orig
            print(f"    max gap = {gap:>8g}: n_pairs={n_kept:5d} ({pct:.0f}%)  AUROC={auroc:.3f}  pacc={pacc:.3f}")


for name in PATHS:
    report(name)
