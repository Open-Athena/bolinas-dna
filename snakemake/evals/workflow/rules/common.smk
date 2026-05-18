import bioframe as bf
import pandas as pd
import polars as pl
from pathlib import Path

from bolinas.data.genome import Genome
from cyvcf2 import VCF
from datasets import Dataset
from huggingface_hub import HfApi

from bolinas.pipelines.evals.labeling import label_variants_by_pip
from bolinas.pipelines.evals.materialize import materialize_sequences
from bolinas.pipelines.evals.matching import (
    CAT_BASE,
    add_subset_distance_bins_v2,
    match_features,
)
from bolinas.pipelines.evals.matching_qc import compute_matching_qc
from bolinas.pipelines.evals.trait_intervals import (
    add_exon,
    add_tss,
    build_dataset,
    get_exon,
    get_tss,
)
from bolinas.pipelines.evals.variants import (
    COORDINATES,
    NUCLEOTIDES,
    attach_per_chrom_consequences,
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

# Continuous features over which to compute per-subset AUPRC leak in the
# matching diagnostic. Mirrors the `continuous` list passed to
# `match_features` in each `{task}_dataset` rule.
QC_CONTINUOUS_FEATURES = {
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
    ],
}


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
    output:
        "results/dataset/{dataset}_harness_{window_size}/{split}.parquet",
    params:
        # Canonical bgzipped + indexed GRCh38 (Ensembl release 115; sequence is
        # byte-identical to 113/114). pyfaidx reads it directly from S3 by
        # byte-range via fsspec/s3fs — no full download. Bypasses the
        # download_genome rule (Ensembl ships plain gzip; the new Genome class
        # post-#182 requires BGZF + .gzi index).
        genome_path=config["canonical_genome_path"],
    run:
        genome = Genome(params.genome_path)
        ds = Dataset.from_parquet(input.parquet)
        n_in = len(ds)
        ds = materialize_sequences(ds, genome, int(wildcards.window_size))
        # Sanity: two rows per input variant, exactly the two strand tags.
        assert len(ds) == 2 * n_in, (
            f"expected {2 * n_in} rows (2x input), got {len(ds)}"
        )
        strands = set(ds.unique("strand"))
        assert strands == {"+", "-"}, f"unexpected strand set: {strands}"
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


rule dataset_matching_qc:
    """Per-subset matching diagnostics: subsampling drops + per-feature AUPRC leak."""
    input:
        pre="results/{dataset}/dataset_all.parquet",
        post="results/dataset_unsplit/{dataset}.parquet",
    output:
        "results/qc/{dataset}.parquet",
    wildcard_constraints:
        dataset="|".join(QC_CONTINUOUS_FEATURES.keys()),
    run:
        qc = compute_matching_qc(
            pl.read_parquet(input.pre),
            pl.read_parquet(input.post),
            QC_CONTINUOUS_FEATURES[wildcards.dataset],
        )
        qc.write_parquet(output[0])
