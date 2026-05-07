"""Re-match: pull dataset_all parquets from S3, apply the new per-biotype
matching design, save dataset_unsplit locally (NO S3 / HF push).

Mirrors the *_dataset rules in the smk files. Single-threaded everywhere —
the previous 16-thread BLAS default thrashed the kernel during cdist on a
12M-row dataset and killed the cluster's SSH.

Per-dataset bin behaviour:
  - mendelian_traits / eqtl: TSS bins active only for `tss_proximal`.
  - complex_traits:          TSS bins active for `tss_proximal` + `distal`
    (testing whether 1-50kb sub-binning lifts the distal leakage shown in
    matched-feature PA, since complex_traits distal is the bulk of that
    dataset).
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

import boto3
import polars as pl

from bolinas.evals.matching import (
    BIN_NA,
    EXON_DIST_BIN_EDGES,
    MAF_BIN_EDGES,
    TSS_DIST_BIN_EDGES,
    bin_feature,
    match_features,
)


def memlog(tag: str) -> None:
    """Print current peak RSS so we can see which step blew up if the kernel
    OOM-kills the process before the next print line lands.
    """
    # ru_maxrss on linux is in KB; divide by 1024 for MB.
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [mem peak={peak_mb:7.0f} MB] {tag}", flush=True)


s3 = boto3.client("s3", region_name="us-east-2")

# Per-dataset config. Aligned with the *_dataset rules in the smk files:
# PC TSS bin is active for {tss_proximal, distal} on all three (the iter-26
# fix for distal distance_tss_pc leakage); nc TSS bin stays tss_proximal-only
# everywhere — distal didn't leak on distance_tss_nc and adding the nc
# constraint thinned distal too aggressively (audit: 100+ pos lost from the
# nc bin alone, plus heavy interaction with MAF_bin/gene_id columns).
DATASET_CONFIG = {
    "mendelian_traits": {
        "with_maf": False,
        "tss_pc_bin_subsets": ["tss_proximal", "distal"],
        "tss_nc_bin_subsets": ["tss_proximal"],
    },
    "complex_traits": {
        "with_maf": True,
        "tss_pc_bin_subsets": ["tss_proximal", "distal"],
        "tss_nc_bin_subsets": ["tss_proximal"],
    },
    "eqtl": {
        "with_maf": True,
        "tss_pc_bin_subsets": ["tss_proximal", "distal"],
        "tss_nc_bin_subsets": ["tss_proximal"],
    },
}
DATASETS = ("complex_traits",)  # iterate only complex_traits this round


def add_bins(
    V: pl.DataFrame,
    with_maf: bool,
    tss_pc_bin_subsets: list[str],
    tss_nc_bin_subsets: list[str],
) -> pl.DataFrame:
    cols = [
        pl.when(pl.col("consequence_group").is_in(tss_pc_bin_subsets))
        .then(bin_feature("distance_tss_pc", TSS_DIST_BIN_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_tss_pc_bin"),
        pl.when(pl.col("consequence_group").is_in(tss_nc_bin_subsets))
        .then(bin_feature("distance_tss_nc", TSS_DIST_BIN_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_tss_nc_bin"),
        pl.when(pl.col("consequence_group") == "splicing")
        .then(bin_feature("distance_exon_pc", EXON_DIST_BIN_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_exon_pc_bin"),
        pl.when(pl.col("consequence_group") == "splicing")
        .then(bin_feature("distance_exon_nc", EXON_DIST_BIN_EDGES))
        .otherwise(pl.lit(BIN_NA))
        .alias("distance_exon_nc_bin"),
    ]
    if with_maf:
        cols.append(
            bin_feature("MAF", MAF_BIN_EDGES, right_closed=True).alias("MAF_bin")
        )
    return V.with_columns(cols)


for name in DATASETS:
    cfg = DATASET_CONFIG[name]
    print(
        f"\n=== {name} "
        f"(tss_pc={cfg['tss_pc_bin_subsets']} tss_nc={cfg['tss_nc_bin_subsets']}) ===",
        flush=True,
    )
    src = f"snakemake/evals/results/{name}/dataset_all.parquet"
    local_in = f"/tmp/{name}_da.parquet"
    if not os.path.exists(local_in):
        print(f"  downloading {src}...", flush=True)
        s3.download_file("oa-bolinas", src, local_in)

    df = pl.read_parquet(local_in)
    print(
        f"  dataset_all: {df.height} rows "
        f"({df.filter(pl.col('label')).height} pos)",
        flush=True,
    )
    memlog("after read_parquet")

    V = add_bins(
        df,
        with_maf=cfg["with_maf"],
        tss_pc_bin_subsets=cfg["tss_pc_bin_subsets"],
        tss_nc_bin_subsets=cfg["tss_nc_bin_subsets"],
    )
    del df
    gc.collect()
    memlog("after add_bins (df freed)")

    cont = ["distance_tss_pc", "distance_tss_nc", "distance_exon_pc", "distance_exon_nc"]
    cat = [
        "chrom",
        "consequence_final",
        "tss_closest_pc_gene_id",
        "tss_closest_nc_gene_id",
        "exon_closest_pc_gene_id",
        "exon_closest_nc_gene_id",
        "distance_tss_pc_bin",
        "distance_tss_nc_bin",
        "distance_exon_pc_bin",
        "distance_exon_nc_bin",
    ]
    if cfg["with_maf"]:
        cont.append("MAF")
        cat.append("MAF_bin")

    pos_V = V.filter(pl.col("label"))
    neg_V = V.filter(~pl.col("label"))
    del V
    gc.collect()
    memlog(f"before match_features (pos={pos_V.height} neg={neg_V.height})")
    matched = (
        match_features(pos_V, neg_V, cont, cat, k=1)
        .with_columns(subset=pl.col("consequence_group"))
    )
    del pos_V, neg_V
    gc.collect()
    memlog("after match_features")

    out = f"/tmp/{name}_unsplit_v2.parquet"
    matched.write_parquet(out)
    n_pairs = matched.filter(pl.col("label")).height
    print(f"  matched pairs: {n_pairs}  → {out}", flush=True)
    print(f"  per-subset:", flush=True)
    print(matched.filter(pl.col("label")).group_by("subset").len().sort("len", descending=True))
