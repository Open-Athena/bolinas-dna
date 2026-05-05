import bioframe as bf
import pandas as pd
import polars as pl
from pathlib import Path

from biofoundation.data import Genome
from cyvcf2 import VCF
from datasets import Dataset
from huggingface_hub import HfApi

from bolinas.evals.materialize import materialize_sequences
from bolinas.evals.matching import (
    BIN_NA,
    EXON_DIST_BIN_EDGES,
    MAF_BIN_EDGES,
    TSS_DIST_BIN_EDGES,
    bin_feature,
    match_features,
    splice_prefilter,
)
from bolinas.evals.trait_intervals import (
    add_exon,
    add_tss,
    build_dataset,
    get_exon,
    get_tss,
)
from bolinas.evals.variants import (
    COORDINATES,
    NUCLEOTIDES,
    check_ref_alt,
    filter_chroms,
    filter_snp,
    lift_hg19_to_hg38,
)


CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
SPLIT_CHROMS = {
    "train": CHROMS[::2],  # odd chroms
    "test": CHROMS[1::2],  # even chroms
}
SPLITS = list(SPLIT_CHROMS.keys())
COORDS = ["chrom", "pos", "ref", "alt"]

# Column order for HF: coordinates, label, subset, match_group, then everything
# else. Datasets missing any of these columns just skip them.
PRIMARY_COLS = COORDS + ["label", "subset", "match_group"]


def _reorder_columns(df):
    primary = [c for c in PRIMARY_COLS if c in df.columns]
    rest = [c for c in df.columns if c not in primary]
    return df[primary + rest]


rule download_genome:
    output:
        "results/genome.fa.gz",
    params:
        url=config["genome_url"],
    shell:
        "wget {params.url} -O {output}"


rule split_dataset_by_chrom:
    input:
        "results/dataset_unsplit/{dataset}.parquet",
    output:
        expand("results/dataset/{{dataset}}/{split}.parquet", split=SPLITS),
    run:
        V = _reorder_columns(pd.read_parquet(input[0]))
        for split, path in zip(SPLITS, output):
            V[V.chrom.isin(SPLIT_CHROMS[split])].to_parquet(path, index=False)


rule materialize_eval_harness_dataset:
    input:
        parquet="results/dataset/{dataset}/{split}.parquet",
        genome="results/genome.fa.gz",
    output:
        "results/dataset/{dataset}_harness_{window_size}/{split}.parquet",
    run:
        genome = Genome(input.genome)
        ds = Dataset.from_parquet(input.parquet)
        ds = materialize_sequences(ds, genome, int(wildcards.window_size))
        ds.to_parquet(output[0])


rule hf_upload:
    input:
        expand("results/dataset/{{dataset}}/{split}.parquet", split=SPLITS),
    output:
        touch("results/upload.done/{dataset}"),
    params:
        repo_name=lambda wildcards: f"{config['output_hf_prefix']}_{wildcards.dataset}",
    run:
        api = HfApi()
        api.create_repo(params.repo_name, repo_type="dataset", exist_ok=True)
        for f in input:
            split = Path(f).stem
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f"{split}.parquet",
                repo_id=params.repo_name,
                repo_type="dataset",
            )


ruleorder: materialize_eval_harness_dataset > split_dataset_by_chrom
