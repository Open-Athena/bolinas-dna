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
    threads: workflow.cores
    run:
        df = pl.concat(
            tqdm((pl.read_parquet(path) for path in input.parquets), total=len(input.parquets)),
        ).sample(fraction=1, shuffle=True, seed=config["shuffle_seed"])
        split_pairs = get_array_split_pairs(len(df), len(output))
        for path, (start, end) in tqdm(zip(output, split_pairs), total=len(output)):
            df.slice(start, end - start).write_ndjson(path)


rule compress_shard:
    input:
        "{anything}.jsonl",
    output:
        "{anything}.jsonl.zst",
    threads: 8
    shell:
        "zstd -T{threads} {input} -o {output}"


rule hf_upload:
    input:
        expand(
            "results/dataset/{{genome_set}}/{{intervals}}/{{identity}}/{{coverage}}/data/{split}/{shard}.jsonl.zst",
            split=SPLITS,
            shard=SHARDS,
        ),
    output:
        touch("results/upload.done/{genome_set}/{intervals}/{identity}/{coverage}"),
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
