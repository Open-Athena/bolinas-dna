"""Local-quantile MAF_bin sweep on complex_traits + eqtl.

Replaces the global MAF_bin (iter 24's 20-edge log-spaced scheme) with a
LOCAL quantile bin: within each categorical match group
(chrom × consequence_final × 4 closest_*_gene_id), assign each row a quantile
bucket of MAF based on the JOINT (pos+neg) distribution of that group.

The local-quantile bin replaces the global MAF_bin in the categorical match
key. MAF stays as a continuous feature for cdist tie-break within each
sub-group. Mendelian skipped — no MAF column.

Sweep n_quantiles ∈ {2, 4, 8, 16, 32}. For each scheme reports:
  (a) per-subset matched-pair count
  (b) per-subset MAF pairwise-accuracy (PA) and binomial p-value

Compares against `no_bin` baseline (iter 25). The iter-26/27 global 20bin
result is in the previous comment for reference.
"""
import os

for var in (
    "POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(var, "1")

import gc
import time

import boto3
import polars as pl
from scipy.stats import binomtest

from bolinas.evals.matching import match_features


CAT_BASE = [
    "chrom", "consequence_final",
    "tss_closest_pc_gene_id", "tss_closest_nc_gene_id",
    "exon_closest_pc_gene_id", "exon_closest_nc_gene_id",
]
CONT_BASE = [
    "distance_tss_pc", "distance_tss_nc",
    "distance_exon_pc", "distance_exon_nc",
    "MAF",
]
N_QUANTILES_GRID = [2, 4, 8, 16, 32]


def add_local_q_bin(df: pl.DataFrame, n_q: int) -> pl.DataFrame:
    """Quantile-bin MAF within each (CAT_BASE) group, joint pos+neg ref.

    Returns df + a `MAF_local_q_bin` String column. Empty / single-value /
    very-small groups collapse via allow_duplicates=True (they end up in
    bin 'q0').
    """
    labels = [f"q{i}" for i in range(n_q)]
    return df.with_columns(
        pl.col("MAF")
        .qcut(quantiles=n_q, labels=labels, allow_duplicates=True)
        .cast(pl.String)
        .over(CAT_BASE)
        .alias("MAF_local_q_bin")
    )


def pa_p(V: pl.DataFrame, feature: str) -> tuple[float, int, float]:
    pos = V.filter(pl.col("label")).select(["match_group", feature]).rename({feature: "pos"})
    neg = V.filter(~pl.col("label")).select(["match_group", feature]).rename({feature: "neg"})
    paired = pos.join(neg, on="match_group", how="inner")
    n = paired.height
    if n == 0:
        return float("nan"), 0, float("nan")
    diff = (paired["pos"] - paired["neg"]).to_numpy()
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = n - wins - losses
    pa = (wins + 0.5 * ties) / n
    decisive = wins + losses
    if decisive == 0:
        return pa, n, float("nan")
    p = binomtest(wins, decisive, p=0.5, alternative="two-sided").pvalue
    return pa, n, p


s3 = boto3.client("s3", region_name="us-east-2")


for dataset in ("complex_traits", "eqtl"):
    print(f"\n{'#' * 70}\n# {dataset}\n{'#' * 70}", flush=True)
    src = f"snakemake/evals/results/{dataset}/dataset_all.parquet"
    local_in = f"/tmp/{dataset}_da.parquet"
    if not os.path.exists(local_in):
        print(f"  downloading {src}...", flush=True)
        s3.download_file("oa-bolinas", src, local_in)
    df_full = pl.read_parquet(local_in)
    n_pos = df_full.filter(pl.col("label")).height
    print(f"  loaded {df_full.height} rows ({n_pos} pos)", flush=True)

    subset_pos = (
        df_full.filter(pl.col("label"))
        .group_by("consequence_group")
        .len()
        .rename({"len": "n_pos"})
    )
    subsets = subset_pos.sort("n_pos", descending=True)["consequence_group"].to_list()

    schemes: dict[str, dict] = {}
    pair_table: dict[str, dict[str, int]] = {s: {} for s in subsets}
    pa_table: dict[str, dict[str, float]] = {s: {} for s in subsets}
    p_table: dict[str, dict[str, float]] = {s: {} for s in subsets}
    totals: dict[str, int] = {}
    times: dict[str, float] = {}

    # No-bin baseline (for reference / Pareto comparison).
    scheme_names = ["no_bin"] + [f"q{n}" for n in N_QUANTILES_GRID]
    for scheme in scheme_names:
        t0 = time.time()
        if scheme == "no_bin":
            V = df_full
            cat = CAT_BASE
        else:
            n_q = int(scheme[1:])
            V = add_local_q_bin(df_full, n_q)
            cat = CAT_BASE + ["MAF_local_q_bin"]

        matched = match_features(
            V.filter(pl.col("label")),
            V.filter(~pl.col("label")),
            CONT_BASE, cat, k=1,
        )
        if scheme != "no_bin":
            del V
            gc.collect()

        n_total = matched.filter(pl.col("label")).height
        totals[scheme] = n_total
        for s in subsets:
            sub = matched.filter(pl.col("consequence_group") == s)
            pair_table[s][scheme] = sub.filter(pl.col("label")).height
            pa, _, p = pa_p(sub, "MAF")
            pa_table[s][scheme] = pa
            p_table[s][scheme] = p
        elapsed = time.time() - t0
        times[scheme] = elapsed
        print(f"  scheme={scheme:7s}  total={n_total:6d}  ({elapsed:5.1f}s)", flush=True)

    # --- print 3 tables ---
    print(f"\n--- {dataset}: matched pair count per (subset, scheme) ---")
    print(f"{'subset':25s}" + "".join(f"  {s:>7}" for s in scheme_names))
    for s in subsets:
        n_pos = subset_pos.filter(pl.col("consequence_group") == s)["n_pos"][0]
        line = f"{s[:24]:25s}"
        for sch in scheme_names:
            line += f"  {pair_table[s][sch]:7d}"
        line += f"   (of {n_pos} pos)"
        print(line)
    line = f"{'TOTAL':25s}"
    for sch in scheme_names:
        line += f"  {totals[sch]:7d}"
    print(line)

    print(f"\n--- {dataset}: MAF pairwise-accuracy (PA) per (subset, scheme) ---")
    print(f"{'subset':25s}" + "".join(f"  {s:>7}" for s in scheme_names))
    for s in subsets:
        line = f"{s[:24]:25s}"
        for sch in scheme_names:
            pa = pa_table[s][sch]
            line += f"  {pa:7.3f}" if pa == pa else f"      NaN"
        print(line)

    print(f"\n--- {dataset}: MAF p-value per (subset, scheme) — '★' = Bonferroni-significant ---")
    bonf = 0.05 / len(subsets)
    print(f"  Bonferroni threshold (per scheme, MAF only across {len(subsets)} subsets) = {bonf:.2e}")
    print(f"{'subset':25s}" + "".join(f"  {s:>7}" for s in scheme_names))
    for s in subsets:
        line = f"{s[:24]:25s}"
        for sch in scheme_names:
            p = p_table[s][sch]
            if p != p:
                line += f"      NaN"
            else:
                star = "★" if p < bonf else " "
                line += f"  {p:6.0e}{star}"
        print(line)
