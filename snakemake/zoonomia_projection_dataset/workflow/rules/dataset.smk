"""Training-shard assembly + HF upload."""

PIPELINE_VERSION = config["pipeline_version"]
INTERVALS_VERSIONS = list(config["intervals_versions"])
HF_OWNER = config["hf_owner"]
ADD_RC = bool(config.get("add_rc", True))
N_SHARDS = int(config.get("n_shards", 64))
SHUFFLE_SEED = int(config.get("shuffle_seed", 42))
assert N_SHARDS < 10_000  # zero-padding width below assumes 4 digits.
SHARDS = [f"shard_{i:04d}" for i in range(N_SHARDS)]

INTERVALS_SOURCES = {
    "v1": f"results/projection/min{PROJECT_MIN_P}/all_species_with_sequence.parquet",
    "v2": f"results/projection/min{PROJECT_MIN_P}/subsets/v2.parquet",
}


rule prepare_training_shards:
    """Read source Parquet → RC augment → shuffle → shard to JSONL."""
    input:
        lambda wc: INTERVALS_SOURCES[wc.intervals_version],
    output:
        temp(
            local(
                expand(
                    "results/dataset/zoonomia-{{pipeline_version}}-{{intervals_version}}/data/train/{shard}.jsonl",
                    shard=SHARDS,
                )
            )
        ),
    threads: workflow.cores
    resources:
        mem_mb=120000,
    run:
        from bolinas.projection.dataset import prepare_shards

        prepare_shards(
            parquet_path=str(input[0]),
            shard_paths=[str(p) for p in output],
            add_rc=ADD_RC,
            shuffle_seed=SHUFFLE_SEED,
        )


rule compress_shard:
    input:
        local("{anything}.jsonl"),
    output:
        local("{anything}.jsonl.zst"),
    threads: 8
    shell:
        "zstd -T{threads} {input} -o {output}"


rule hf_upload_dataset:
    """Upload a dataset's compressed shard folder to HF Hub (single train split)."""
    input:
        local(
            expand(
                "results/dataset/zoonomia-{{pipeline_version}}-{{intervals_version}}/data/train/{shard}.jsonl.zst",
                shard=SHARDS,
            )
        ),
    output:
        # Explicit `touch` in shell: S3 default-storage doesn't auto-create touch() markers.
        "results/upload.done/zoonomia-{pipeline_version}-{intervals_version}",
    params:
        repo=lambda wc: f"{HF_OWNER}/zoonomia-{wc.pipeline_version}-{wc.intervals_version}",
        data_dir=lambda wc: f"results/dataset/zoonomia-{wc.pipeline_version}-{wc.intervals_version}",
    wildcard_constraints:
        pipeline_version=r"v\d+",
        intervals_version=r"v\d+",
    threads: workflow.cores
    shell:
        """
        hf upload-large-folder {params.repo} --repo-type dataset {params.data_dir}
        mkdir -p $(dirname {output})
        touch {output}
        """


rule all_hf:
    """Trigger HF push for every (PIPELINE_VERSION × INTERVALS_VERSIONS) combo."""
    input:
        expand(
            "results/upload.done/zoonomia-{p}-{iv}",
            p=[PIPELINE_VERSION],
            iv=INTERVALS_VERSIONS,
        ),
