"""Per-(dataset, subset, matched continuous feature) pairwise accuracy +
binomial sign-test p-value for the per-biotype matching design.

Mirrors `leakage.py` but uses the per-biotype distance columns
(distance_tss_pc, distance_tss_nc, distance_exon_pc, distance_exon_nc) plus
MAF/ld_score for complex_traits/eqtl. Reads from local /tmp parquets that
`rematch_per_biotype.py` emits — keeps the iteration loop entirely local.

Reports PA + p-value with Bonferroni-flavored thresholds so we can see at a
glance which (subset, feature) pairs still leak.
"""

from pathlib import Path

import polars as pl
from scipy.stats import binomtest

PATHS = {
    "mendelian_traits": "/tmp/mendelian_traits_unsplit_v2.parquet",
    "complex_traits": "/tmp/complex_traits_unsplit_v2.parquet",
    "eqtl": "/tmp/eqtl_unsplit_v2.parquet",
}
FEATURES = {
    "mendelian_traits": [
        "distance_tss_pc",
        "distance_tss_nc",
        "distance_exon_pc",
        "distance_exon_nc",
    ],
    "complex_traits": [
        "distance_tss_pc",
        "distance_tss_nc",
        "distance_exon_pc",
        "distance_exon_nc",
        "MAF",
        "ld_score",
    ],
    "eqtl": [
        "distance_tss_pc",
        "distance_tss_nc",
        "distance_exon_pc",
        "distance_exon_nc",
        "MAF",
    ],
}


def pairwise_acc_and_p(V: pl.DataFrame, feature: str) -> tuple[float, int, float]:
    """Per match_group: PA = (wins + 0.5*ties)/n, two-sided sign-test p on
    wins vs (wins+losses).  Ties excluded from the p-value denominator.
    """
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
        return float("nan"), 0, float("nan")
    diff = (paired["x_pos"] - paired["x_neg"]).to_numpy()
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    pa = (wins + 0.5 * ties) / n
    decisive = wins + losses
    if decisive == 0:
        return pa, n, float("nan")
    pval = binomtest(wins, decisive, p=0.5, alternative="two-sided").pvalue
    return pa, n, pval


def report(name: str) -> None:
    path = Path(PATHS[name])
    if not path.exists():
        print(f"\n========== {name} (missing: {path}) ==========")
        return
    print(f"\n========== {name} ==========")
    V = pl.read_parquet(path)
    feats = [f for f in FEATURES[name] if f in V.columns]
    missing = [f for f in FEATURES[name] if f not in V.columns]
    if missing:
        print(f"  missing columns (skipping): {missing}")
    subsets = sorted(V["consequence_group"].unique().to_list())
    # Bonferroni: count cells = subsets × feats
    n_cells = sum(
        1
        for s in subsets
        if V.filter(pl.col("consequence_group") == s).filter(pl.col("label")).height > 0
        for _ in feats
    )
    alpha_bonf = 0.05 / max(n_cells, 1)
    print(f"  n_cells={n_cells}  alpha_bonf={alpha_bonf:.2e}")
    for subset in subsets:
        sub = V.filter(pl.col("consequence_group") == subset)
        n_pos = sub.filter(pl.col("label")).height
        if n_pos == 0:
            continue
        parts = [f"{subset:18s}  n={n_pos:4d}"]
        for feat in feats:
            pa, _, pval = pairwise_acc_and_p(sub, feat)
            star = "**" if pval < alpha_bonf else "*" if pval < 0.05 else "  "
            parts.append(f"{feat:>20}: PA={pa:.3f} p={pval:.1e} {star}")
        print("  " + "\n    ".join(parts))


for name in PATHS:
    report(name)
