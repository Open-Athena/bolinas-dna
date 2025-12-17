# just adds a placeholder column with "." (sequence name)
rule prepare_intervals_for_window_seq:
    input:
        "results/intervals/{intervals}/{g}.bed.gz",
    output:
        temp("results/intervals_for_window_seq/{intervals}/{g}.bed.gz"),
    shell:
        """
        zcat {input} |
        awk 'BEGIN {{OFS="\t"}} {{print $1, $2, $3, "."}}' |
        gzip > {output}
        """


rule window_seq:
    input:
        "results/genome/{g}.2bit",
        "results/intervals_for_window_seq/windows/recipe/{recipe}/{w}/{s}/{g}.bed.gz",
    output:
        temp("results/intervals_seq/{recipe}/{w}/{s}/{g}.fa"),
    shell:
        "twoBitToFa {input[0]} {output} -bed={input[1]} -bedPos"


rule make_dataset_genome:
    input:
        "results/intervals_seq/{intervals}/{g}.fa",
    output:
        expand(
            "results/dataset_genome/{{intervals}}/{{g}}/{split}.parquet",
            split=SPLITS,
        ),
    threads: 2
    run:
        df = load_fasta(input[0]).to_frame().reset_index(names="id")
        assert len(df) > 0, "No windows found"
        df.id = df.id.astype(str)  # to handle empty dataframes
        if config["add_rc"]:
            df = add_rc(df)
        df["chrom"] = df.id.str.split(":").str[0]
        chrom_split = pd.DataFrame(dict(chrom=df.chrom.unique()))
        chrom_split["split"] = "train"
        chrom_split.loc[
            chrom_split.chrom.isin(config["validation_chroms"]), "split"
        ] = "validation"
        chrom_split.loc[chrom_split.chrom.isin(config["test_chroms"]), "split"] = "test"
        df = df.merge(chrom_split, on="chrom", how="left")
        df = pl.from_pandas(df[["id", "seq", "split"]])
        for path, split in zip(output, SPLITS):
            df.filter(split=split).drop("split").write_parquet(path)


rule merge_datasets:
    input:
        expand(
            "results/dataset_genome/{{intervals}}/{g}/{{split}}.parquet",
            g=genomes.index,
        ),
    output:
        temp(
            expand(
                "results/dataset/{{intervals}}/data/{{split}}/{shard}.jsonl",
                shard=SHARDS,
            )
        ),
    threads: workflow.cores
    run:
        df = pl.concat(
            tqdm((pl.read_parquet(path) for path in input), total=len(input)),
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
            "results/dataset/{{intervals}}/data/{split}/{shard}.jsonl.zst",
            split=SPLITS,
            shard=SHARDS,
        ),
    output:
        touch("results/upload.done/{intervals}"),
    params:
        lambda wildcards: config["output_hf_prefix"]
        + wildcards.intervals.replace("/", "_"),
    shell:
        "hf upload-large-folder {params} --repo-type dataset results/dataset/{wildcards.intervals}"
