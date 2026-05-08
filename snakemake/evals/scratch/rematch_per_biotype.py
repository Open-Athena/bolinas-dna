"""Re-match: pull dataset_all parquets from S3, apply the matching design
under exploration, save dataset_unsplit locally (NO S3 / HF push).

Single-threaded everywhere (kernel thrashes BLAS at 16 threads on a 12M-row
dataset).

Iter 32 design (current — MAF tentatively locked, distance bins re-added):

  Continuous: 4 per-biotype distances (+ MAF for complex / eqtl)
  Categorical:
    - chrom, consequence_final
    - tss_closest_pc_gene_id, tss_closest_nc_gene_id
    - exon_closest_pc_gene_id, exon_closest_nc_gene_id
    - distance_tss_pc_bin    (tss_proximal subset only, iter-22 edges)
    - distance_exon_pc_bin   (splicing subset only, iter-22 edges)
    - MAF_bin                (per-dataset, per-subset scheme — iter 31 locked)

  Per-dataset MAF scheme (iter 31 winners):
    mendelian_traits : no MAF column → no MAF_bin
    complex_traits   : tiered_v1 (per-subset global edges)
    eqtl             : tiered_log8_distal_only (tiered_v1 + log_local_8 for distal)

Sweeps to inform the locked MAF choice are in maf_*_sweep.py siblings.
"""
import os

# Cap every thread source BEFORE importing numpy / sklearn / scipy / polars.
for var in (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(var, "1")

import gc
import resource
import time

import boto3
import polars as pl

from bolinas.evals.matching import (
    BIN_NA,
    MAF_BIN_EDGES,
    bin_feature,
    match_features,
)


_t0 = time.time()


def stamp(tag: str) -> None:
    elapsed = time.time() - _t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [+{elapsed:6.1f}s  peak={peak_mb:7.0f} MB] {tag}", flush=True)


s3 = boto3.client("s3", region_name="us-east-2")

# === locked iter-22 distance-bin edges ===
TSS_EDGES = [0, 50, 100, 200, 500, 1000]
EXON_EDGES = [0, 5, 20, 30]

# === MAF scheme tiers (iter 31) ===
BINS_20 = MAF_BIN_EDGES                                                       # 20 buckets
BINS_10 = [0.0, 0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5]  # 10
BINS_5 = [0.0, 0.001, 0.01, 0.05, 0.2, 0.5]                                   # 5
LOG_LOCAL = "log_local"
LOG_LOCAL_N = 8

TIERED_V1: dict[str, list[float] | str] = {
    "distal": BINS_20,
    "tss_proximal": BINS_20,
    "non_coding_transcript_exon_variant": BINS_20,
    "3_prime_UTR_variant": BINS_10,
    "5_prime_UTR_variant": BINS_10,
    "missense_variant": BINS_10,
    "synonymous_variant": BINS_5,
    "splicing": BINS_5,
    "mature_miRNA_variant": BINS_5,
    "stop_retained_variant": BINS_5,
    "coding_sequence_variant": BINS_5,
}
TIERED_LOG8_DISTAL_ONLY: dict[str, list[float] | str] = {
    **TIERED_V1,
    "distal": LOG_LOCAL,
}

DATASET_CONFIG = {
    "mendelian_traits": {"maf_scheme": None},
    "complex_traits": {"maf_scheme": TIERED_V1},
    "eqtl": {"maf_scheme": TIERED_LOG8_DISTAL_ONLY},
}
DATASETS = ("mendelian_traits", "complex_traits", "eqtl")

CAT_BASE = [
    "chrom",
    "consequence_final",
    "tss_closest_pc_gene_id",
    "tss_closest_nc_gene_id",
    "exon_closest_pc_gene_id",
    "exon_closest_nc_gene_id",
]
CONT_BASE = [
    "distance_tss_pc",
    "distance_tss_nc",
    "distance_exon_pc",
    "distance_exon_nc",
]


def add_distance_bins(df: pl.DataFrame) -> pl.DataFrame:
    """Iter 33: per-biotype TSS bins for tss_proximal (pc + nc) and exon
    bin for splicing (pc only — exon nc didn't leak in mendelian baseline).

    distance_tss_pc_bin  : tss_proximal only (else BIN_NA)
    distance_tss_nc_bin  : tss_proximal only (else BIN_NA)
    distance_exon_pc_bin : splicing     only (else BIN_NA)
    """
    return df.with_columns(
        pl.when(pl.col("consequence_group") == "tss_proximal")
        .then(bin_feature("distance_tss_pc", TSS_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_tss_pc_bin"),
        pl.when(pl.col("consequence_group") == "tss_proximal")
        .then(bin_feature("distance_tss_nc", TSS_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_tss_nc_bin"),
        pl.when(pl.col("consequence_group") == "splicing")
        .then(bin_feature("distance_exon_pc", EXON_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_exon_pc_bin"),
    )


def add_maf_bin(df: pl.DataFrame, scheme: dict) -> pl.DataFrame:
    """Per-subset MAF_bin via consequence_group lookup.

    `scheme[subset]` is either:
      - list[float] for global right-closed edges
      - LOG_LOCAL sentinel for local equal-width log10 bins (n = LOG_LOCAL_N),
        joint pos+neg ref over CAT_BASE.
    """
    needs_log = any(v == LOG_LOCAL for v in scheme.values())
    if needs_log:
        log_maf = pl.col("MAF").clip(1e-10, 1.0).log10()
        df = df.with_columns(
            log_maf.min().over(CAT_BASE).alias("_lo"),
            log_maf.max().over(CAT_BASE).alias("_hi"),
        )
        width = (pl.col("_hi") - pl.col("_lo")) / LOG_LOCAL_N
        log_local_idx = (
            ((log_maf - pl.col("_lo")) / width)
            .floor()
            .cast(pl.Int64, strict=False)
            .fill_null(0)
            .clip(0, LOG_LOCAL_N - 1)
        )
        log_local_label = pl.format("ll:{}", log_local_idx)

    expr = pl.lit("UNKNOWN")
    for subset, val in scheme.items():
        if val == LOG_LOCAL:
            bin_expr = log_local_label
        else:
            bin_expr = pl.format(
                "{}:{}",
                pl.lit(subset[:8]),
                bin_feature("MAF", val, right_closed=True),
            )
        expr = pl.when(pl.col("consequence_group") == subset).then(bin_expr).otherwise(expr)

    df = df.with_columns(expr.alias("MAF_bin"))
    if needs_log:
        df = df.drop(["_lo", "_hi"])
    return df


for name in DATASETS:
    cfg = DATASET_CONFIG[name]
    has_maf = cfg["maf_scheme"] is not None
    print(f"\n=== {name} (with_maf={has_maf}) ===", flush=True)
    src = f"snakemake/evals/results/{name}/dataset_all.parquet"
    local_in = f"/tmp/{name}_da.parquet"
    if not os.path.exists(local_in):
        print(f"  downloading {src}...", flush=True)
        s3.download_file("oa-bolinas", src, local_in)

    V = pl.read_parquet(local_in)
    if has_maf:
        # log_local computation can't handle null/NaN MAF → drop them up-front
        # (consistent with the iter 28-31 sweeps).
        V = V.filter(pl.col("MAF").is_finite() & pl.col("MAF").is_not_null())
    print(
        f"  dataset_all: {V.height} rows ({V.filter(pl.col('label')).height} pos)",
        flush=True,
    )
    stamp("after read_parquet")

    V = add_distance_bins(V)
    cat = CAT_BASE + [
        "distance_tss_pc_bin",
        "distance_tss_nc_bin",
        "distance_exon_pc_bin",
    ]
    cont = CONT_BASE.copy()
    if has_maf:
        V = add_maf_bin(V, cfg["maf_scheme"])
        cat.append("MAF_bin")
        cont.append("MAF")

    pos_V = V.filter(pl.col("label"))
    neg_V = V.filter(~pl.col("label"))
    del V
    gc.collect()
    stamp(f"before match_features (pos={pos_V.height} neg={neg_V.height})")
    matched = (
        match_features(pos_V, neg_V, cont, cat, k=1)
        .with_columns(subset=pl.col("consequence_group"))
    )
    del pos_V, neg_V
    gc.collect()
    stamp("after match_features")

    out = f"/tmp/{name}_unsplit_v2.parquet"
    matched.write_parquet(out)
    n_pairs = matched.filter(pl.col("label")).height
    print(f"  matched pairs: {n_pairs}  → {out}", flush=True)
    print("  per-subset:", flush=True)
    print(
        matched.filter(pl.col("label"))
        .group_by("subset")
        .len()
        .sort("len", descending=True)
    )
