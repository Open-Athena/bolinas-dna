"""Re-match: pull dataset_all parquets from S3, apply the matching design
under exploration, save dataset_unsplit locally (NO S3 / HF push).

Single-threaded everywhere — the previous 16-thread BLAS default thrashed the
kernel during cdist on a 12M-row dataset and killed the cluster's SSH.

Current design under exploration (iter 26):
  - Continuous: 4 per-biotype distances (+ MAF for complex / eqtl).
  - Categorical: chrom, consequence_final, 4 closest_*_gene_id, **MAF_bin
    always-on** for complex / eqtl (the iter-25 baseline showed Bonf-significant
    MAF leakage in every major subset there — adding the bin globally should
    close the largest single signal in the diagnostic).
  - No TSS / exon distance bins yet — will revisit subset-specific bins next.
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

from bolinas.evals.matching import MAF_BIN_EDGES, bin_feature, match_features


_t0 = time.time()


def stamp(tag: str) -> None:
    """Per-step elapsed wall-clock + peak RSS, so iteration speed and the
    spot where things blow up are both visible from one log line.
    """
    elapsed = time.time() - _t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [+{elapsed:6.1f}s  peak={peak_mb:7.0f} MB] {tag}", flush=True)


s3 = boto3.client("s3", region_name="us-east-2")

# Per-dataset config: only `with_maf` for now (drives whether MAF joins the
# continuous match features). complex_traits + eqtl have a usable MAF
# column; mendelian doesn't, so we skip it there.
DATASET_CONFIG = {
    "mendelian_traits": {"with_maf": False},
    "complex_traits": {"with_maf": True},
    "eqtl": {"with_maf": True},
}
DATASETS = ("mendelian_traits", "complex_traits", "eqtl")

CAT = [
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


for name in DATASETS:
    cfg = DATASET_CONFIG[name]
    print(f"\n=== {name} (with_maf={cfg['with_maf']}) ===", flush=True)
    src = f"snakemake/evals/results/{name}/dataset_all.parquet"
    local_in = f"/tmp/{name}_da.parquet"
    if not os.path.exists(local_in):
        print(f"  downloading {src}...", flush=True)
        s3.download_file("oa-bolinas", src, local_in)

    V = pl.read_parquet(local_in)
    print(
        f"  dataset_all: {V.height} rows "
        f"({V.filter(pl.col('label')).height} pos)",
        flush=True,
    )
    stamp("after read_parquet")

    cont = CONT_BASE.copy()
    cat = CAT.copy()
    if cfg["with_maf"]:
        cont.append("MAF")
        cat.append("MAF_bin")
        V = V.with_columns(
            bin_feature("MAF", MAF_BIN_EDGES, right_closed=True).alias("MAF_bin")
        )

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
    print(f"  per-subset:", flush=True)
    print(
        matched.filter(pl.col("label"))
        .group_by("subset")
        .len()
        .sort("len", descending=True)
    )
