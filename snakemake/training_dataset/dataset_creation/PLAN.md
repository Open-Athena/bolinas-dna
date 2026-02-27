# Plan: Sequence Similarity-Based Train/Test Split Filtering

Issue #28 — use MMseqs2 search to remove training sequences similar to validation sequences.

## Revised pipeline flow

```
intervals.smk (existing)
    → results/intervals/windows/recipe/{recipe}/{w}/{s}/{g}.bed.gz
    ↓
split_bed_by_chrom (NEW — polars, replaces prepare_intervals_for_window_seq + window_seq + make_dataset_genome)
    → results/intervals_split/{intervals}/{g}/train.bed (temp)
    → results/intervals_split/{intervals}/{g}/validation.bed (temp)
    ↓
window_seq_split (NEW — twoBitToFa on each split BED)
    → results/dataset_genome/{intervals}/{g}/train.fasta (temp)
    → results/dataset_genome/{intervals}/{g}/validation.fasta (temp)
    │
    ├──→ make_val_parquet: load val FASTA, add RC → validation.parquet
    │
    ├──→ write_val_fasta: cat per-genome val FASTAs → genome_set val FASTA
    │       ↓
    │    create_val_db → valDB (temp)
    │
    ├──→ create_train_db → trainDB (temp)
    │       ↓
    │    search_leakage (val=query, train=target, per genome)
    │       ↓
    │    extract_leakage_hits → hits.tsv (S3, per genome × genome_set × thresholds)
    │       ↓
    └──→ make_filtered_train_parquet: load train FASTA, remove hit IDs, add RC → train.parquet
    ↓
merge_datasets (reads filtered train parquets + val parquets)
    → sharded JSONL → compress → upload
```

Key properties:
- **BED split with polars** — no FASTA parsing for the split, no RC complexity
- **FASTAs have no RC, no strand suffixes** — go straight into mmseqs2
- **Hit IDs are base IDs** (e.g. `NC_000019.10:64000-64256`) — clean expansion to `_+` and `_-` when filtering
- **Per-genome filtering** — no aggregation; each genome's hits.tsv feeds its own filtered parquet
- **Removes 3 old rules**: `prepare_intervals_for_window_seq`, `window_seq`, `make_dataset_genome`

## Modified: `workflow/rules/dataset.smk`

Replace `prepare_intervals_for_window_seq`, `window_seq`, and `make_dataset_genome` with the
new rules below. Keep `merge_datasets`, `compress_shard`, `hf_upload` (modified).

```python


rule split_bed_by_chrom:
    """Split windowed BED intervals into train/validation by chromosome.

    Also adds a placeholder name column (".") required by twoBitToFa -bedPos.
    """
    input:
        bed="results/intervals/windows/recipe/{recipe}/{w}/{s}/{g}.bed.gz",
    output:
        train=temp("results/intervals_split/{recipe}/{w}/{s}/{g}/train.bed"),
        val=temp("results/intervals_split/{recipe}/{w}/{s}/{g}/validation.bed"),
    run:
        df = pl.read_csv(
            input.bed,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end"],
        )
        val_chroms = config["validation_chroms"]
        for split_name, path, is_val in [
            ("train", output.train, False),
            ("validation", output.val, True),
        ]:
            split_df = df.filter(
                pl.col("chrom").is_in(val_chroms) if is_val
                else ~pl.col("chrom").is_in(val_chroms)
            )
            # Add "." name column for twoBitToFa -bedPos
            split_df = split_df.with_columns(pl.lit(".").alias("name"))
            split_df.write_csv(path, separator="\t", include_header=False)


rule window_seq_split:
    """Extract sequences from 2bit genome using split BED intervals."""
    input:
        genome="results/genome/{g}.2bit",
        bed="results/intervals_split/{intervals}/{g}/{split}.bed",
    output:
        fasta=temp("results/dataset_genome/{intervals}/{g}/{split}.fasta"),
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitToFa {input.genome} {output.fasta} -bed={input.bed} -bedPos"


rule make_val_parquet:
    """Convert validation FASTA to parquet, optionally adding reverse complements."""
    input:
        fasta="results/dataset_genome/{intervals}/{g}/validation.fasta",
    output:
        parquet="results/dataset_genome/{intervals}/{g}/validation.parquet",
    run:
        df = load_fasta(input.fasta).to_frame().reset_index(names="id")
        if len(df) == 0:
            pl.DataFrame(
                {"id": [], "seq": []}, schema={"id": pl.String, "seq": pl.String}
            ).write_parquet(output.parquet)
        else:
            df.id = df.id.astype(str)
            if config["add_rc"]:
                df = add_rc(df)
            pl.from_pandas(df[["id", "seq"]]).write_parquet(output.parquet)


rule write_val_fasta:
    """Concatenate per-genome validation FASTAs into one per genome_set."""
    input:
        fastas=expand(
            "results/dataset_genome/{{intervals}}/{g}/validation.fasta",
            g=lambda wildcards: genome_sets[wildcards.genome_set],
        ),
    output:
        fasta=temp("results/leakage_filter/{genome_set}/{intervals}/validation.fasta"),
    wildcard_constraints:
        genome_set="|".join(genome_sets_list),
    shell:
        "cat {input.fastas} > {output.fasta}"


rule create_train_db:
    """Create MMseqs2 database from a genome's train FASTA."""
    input:
        fasta="results/dataset_genome/{intervals}/{g}/train.fasta",
    output:
        db=temp("results/leakage_filter/{intervals}/{g}/trainDB"),
        db_type=temp("results/leakage_filter/{intervals}/{g}/trainDB.dbtype"),
    params:
        db_prefix="results/leakage_filter/{intervals}/{g}/trainDB",
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        "mmseqs createdb {input.fasta} {params.db_prefix} --mask-lower-case 1"


rule create_val_db:
    """Create MMseqs2 database from a genome_set's validation FASTA."""
    input:
        fasta="results/leakage_filter/{genome_set}/{intervals}/validation.fasta",
    output:
        db=temp("results/leakage_filter/{genome_set}/{intervals}/valDB"),
        db_type=temp("results/leakage_filter/{genome_set}/{intervals}/valDB.dbtype"),
    params:
        db_prefix="results/leakage_filter/{genome_set}/{intervals}/valDB",
    wildcard_constraints:
        genome_set="|".join(genome_sets_list),
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        "mmseqs createdb {input.fasta} {params.db_prefix} --mask-lower-case 1"


rule search_leakage:
    """Search validation sequences against a genome's training sequences.

    Val = query (smaller, ~14K seqs), train = target (larger, indexed once).
    --strand 2 searches both strands (default is forward only!).
    --cov-mode 0 = bidirectional coverage.
    """
    input:
        query_db="results/leakage_filter/{genome_set}/{intervals}/valDB",
        query_db_type="results/leakage_filter/{genome_set}/{intervals}/valDB.dbtype",
        target_db="results/leakage_filter/{intervals}/{g}/trainDB",
        target_db_type="results/leakage_filter/{intervals}/{g}/trainDB.dbtype",
    output:
        result_index=temp(
            "results/leakage_filter/{genome_set}/{intervals}/{g}/{identity}/{coverage}/resultDB.index"
        ),
        result_db_type=temp(
            "results/leakage_filter/{genome_set}/{intervals}/{g}/{identity}/{coverage}/resultDB.dbtype"
        ),
    params:
        query_prefix="results/leakage_filter/{genome_set}/{intervals}/valDB",
        target_prefix="results/leakage_filter/{intervals}/{g}/trainDB",
        result_prefix="results/leakage_filter/{genome_set}/{intervals}/{g}/{identity}/{coverage}/resultDB",
        tmp_dir="results/leakage_filter/{genome_set}/{intervals}/{g}/{identity}/{coverage}/tmp",
        identity=lambda wildcards: float(wildcards.identity),
        coverage=lambda wildcards: float(wildcards.coverage),
    threads: 1
    resources:
        mem_mb=4000,
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mkdir -p {params.tmp_dir}
        mmseqs search \
            {params.query_prefix} \
            {params.target_prefix} \
            {params.result_prefix} \
            {params.tmp_dir} \
            --search-type 3 \
            --mask-lower-case 1 \
            --strand 2 \
            --min-seq-id {params.identity} \
            -c {params.coverage} \
            --cov-mode 0 \
            --threads {threads}
        rm -rf {params.tmp_dir}
        """


rule extract_leakage_hits:
    """Convert binary search results to TSV for a single genome."""
    input:
        query_db="results/leakage_filter/{genome_set}/{intervals}/valDB",
        query_db_type="results/leakage_filter/{genome_set}/{intervals}/valDB.dbtype",
        target_db="results/leakage_filter/{intervals}/{g}/trainDB",
        target_db_type="results/leakage_filter/{intervals}/{g}/trainDB.dbtype",
        result_index="results/leakage_filter/{genome_set}/{intervals}/{g}/{identity}/{coverage}/resultDB.index",
        result_db_type="results/leakage_filter/{genome_set}/{intervals}/{g}/{identity}/{coverage}/resultDB.dbtype",
    output:
        tsv="results/leakage_filter/{genome_set}/{intervals}/{g}/{identity}/{coverage}/hits.tsv",
    params:
        query_prefix="results/leakage_filter/{genome_set}/{intervals}/valDB",
        target_prefix="results/leakage_filter/{intervals}/{g}/trainDB",
        result_prefix="results/leakage_filter/{genome_set}/{intervals}/{g}/{identity}/{coverage}/resultDB",
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mmseqs convertalis \
            {params.query_prefix} \
            {params.target_prefix} \
            {params.result_prefix} \
            {output.tsv} \
            --format-output "query,target,fident,qcov,tcov"
        """


rule make_filtered_train_parquet:
    """Load train FASTA, remove sequences with leakage hits, add RC, write parquet.

    Hit IDs are base IDs (no strand suffix) from the FASTA. When add_rc is
    enabled, both _+ and _- versions of each hit ID are filtered from the
    final parquet.
    """
    input:
        fasta="results/dataset_genome/{intervals}/{g}/train.fasta",
        hits="results/leakage_filter/{genome_set}/{intervals}/{g}/{identity}/{coverage}/hits.tsv",
    output:
        parquet="results/dataset_genome_filtered/{genome_set}/{intervals}/{g}/{identity}/{coverage}/train.parquet",
    run:
        df = load_fasta(input.fasta).to_frame().reset_index(names="id")

        if len(df) == 0:
            pl.DataFrame(
                {"id": [], "seq": []}, schema={"id": pl.String, "seq": pl.String}
            ).write_parquet(output.parquet)
        else:
            df.id = df.id.astype(str)

            # Read hit IDs to filter (base IDs, no strand suffix)
            hits = pl.read_csv(
                input.hits,
                separator="\t",
                has_header=False,
                new_columns=["query", "target", "fident", "qcov", "tcov"],
            )
            if hits.height > 0:
                hit_ids = set(hits["target"].unique().to_list())
                n_before = len(df)
                df = df[~df.id.isin(hit_ids)]
                n_removed = n_before - len(df)
                print(
                    f"Leakage filter ({wildcards.g}, id={wildcards.identity} "
                    f"cov={wildcards.coverage}): removed {n_removed:,} / {n_before:,} "
                    f"({100 * n_removed / n_before:.2f}%)"
                )

            # Add reverse complements after filtering
            if config["add_rc"]:
                df = add_rc(df)

            pl.from_pandas(df[["id", "seq"]]).write_parquet(output.parquet)


rule merge_datasets:
    input:
        parquets=lambda wildcards: (
            expand(
                "results/dataset_genome_filtered/{genome_set}/{intervals}/{g}/{identity}/{coverage}/train.parquet",
                genome_set=wildcards.genome_set,
                intervals=wildcards.intervals,
                g=genome_sets[wildcards.genome_set],
                identity=wildcards.identity,
                coverage=wildcards.coverage,
            )
            if wildcards.split == "train"
            else expand(
                "results/dataset_genome/{intervals}/{g}/validation.parquet",
                intervals=wildcards.intervals,
                g=genome_sets[wildcards.genome_set],
            )
        ),
    output:
        temp(
            expand(
                "results/dataset/{{genome_set}}/{{intervals}}/{{identity}}/{{coverage}}/data/{{split}}/{shard}.jsonl",
                shard=SHARDS,
            )
        ),
    wildcard_constraints:
        genome_set="|".join(genome_sets_list),
        intervals="|".join(intervals_list),
    threads: workflow.cores
    run:
        df = pl.concat(
            tqdm((pl.read_parquet(path) for path in input.parquets), total=len(input.parquets)),
        ).sample(fraction=1, shuffle=True, seed=config["shuffle_seed"])
        split_pairs = get_array_split_pairs(len(df), len(output))
        for path, (start, end) in tqdm(zip(output, split_pairs), total=len(output)):
            df.slice(start, end - start).write_ndjson(path)


rule compress_shard:
    # ... unchanged (generic {anything}.jsonl → {anything}.jsonl.zst) ...


rule hf_upload:
    input:
        expand(
            "results/dataset/{{genome_set}}/{{intervals}}/{{identity}}/{{coverage}}/data/{split}/{shard}.jsonl.zst",
            split=SPLITS,
            shard=SHARDS,
        ),
    output:
        touch("results/upload.done/{genome_set}/{intervals}/{identity}/{coverage}"),
    wildcard_constraints:
        genome_set="|".join(genome_sets_list),
        intervals="|".join(intervals_list),
    params:
        name=lambda wildcards: (
            config["output_hf_prefix"]
            + "-genome_set-" + wildcards.genome_set
            + "-intervals-" + wildcards.intervals.replace("/", "_")
            + "-id" + wildcards.identity
            + "_cov" + wildcards.coverage
        ),
        data_dir=lambda wildcards: (
            f"results/dataset/{wildcards.genome_set}/{wildcards.intervals}"
            f"/{wildcards.identity}/{wildcards.coverage}"
        ),
    threads: workflow.cores
    shell:
        "hf upload-large-folder {params.name} --repo-type dataset {params.data_dir}"
```

## Modified: `workflow/Snakefile`

```python
configfile: "config/config.yaml"


SPLITS = ["train", "validation"]
N_SHARDS = config["n_shards"]
assert N_SHARDS < 1e4
SHARDS = [f"shard_{i:04}" for i in range(N_SHARDS)]

LEAKAGE_IDENTITIES = config["leakage_filter"]["identity"]
LEAKAGE_COVERAGES = config["leakage_filter"]["coverage"]


include: "rules/common.smk"


genomes = load_genome_info(config["genomes_path"])
genome_sets = load_genome_sets(genomes, config["genome_sets"])
genome_sets_list = list(genome_sets.keys())
intervals_list = config["intervals"]


include: "rules/download.smk"
include: "rules/intervals.smk"
include: "rules/stats.smk"
include: "rules/dataset.smk"
include: "rules/eda.smk"


rule all:
    input:
        expand(
            "results/upload.done/{genome_set}/{intervals}/{identity}/{coverage}",
            genome_set=genome_sets_list,
            intervals=config["intervals"],
            identity=LEAKAGE_IDENTITIES,
            coverage=LEAKAGE_COVERAGES,
        ),


# network-constrained
rule all_download:
    input:
        expand("results/genome/{g}.2bit", g=genomes.index),
```

## Modified: `config/config.yaml`

Add at end:

```yaml
leakage_filter:
  identity: [0.3]
  coverage: [0.3]
```

## New file: `workflow/envs/mmseqs2.yaml`

```yaml
channels:
  - conda-forge
  - bioconda
dependencies:
  - mmseqs2
```

## Storage decisions (temp vs S3)

| File | temp()? | Why |
|------|---------|-----|
| `train.bed` / `validation.bed` | yes | Small, regenerated from intervals |
| `train.fasta` / `validation.fasta` | yes | Large, regenerated from twoBitToFa |
| `validation.fasta` (genome_set) | yes | Just a cat of per-genome FASTAs |
| `trainDB` + side-effects | yes | Large binary, regenerated from FASTA |
| `valDB` + side-effects | yes | Large binary, regenerated from FASTA |
| `resultDB` + side-effects | yes | Large binary, only needed until hits.tsv |
| **`hits.tsv`** | **no (S3)** | Small, enables incremental re-runs |
| **`validation.parquet`** | **no (S3)** | Per-genome unfiltered val |
| **`filtered train.parquet`** | **no (S3)** | Per-genome filtered train |
