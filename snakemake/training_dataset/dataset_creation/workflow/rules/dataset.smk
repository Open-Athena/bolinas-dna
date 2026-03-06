rule prepare_intervals_for_window_seq:
    """Add placeholder name column ('.') required by twoBitToFa -bedPos."""
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
    """Extract sequences from 2bit genome using windowed BED intervals."""
    input:
        "results/genome/{g}.2bit",
        "results/intervals_for_window_seq/windows/recipe/{recipe}/{w}/{s}/{g}.bed.gz",
    output:
        temp("results/intervals_seq/{recipe}/{w}/{s}/{g}.fa"),
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitToFa {input[0]} {output} -bed={input[1]} -bedPos"


rule make_parquet:
    """Convert FASTA to parquet, optionally adding reverse complements."""
    input:
        "results/intervals_seq/{intervals}/{g}.fa",
    output:
        "results/dataset_genome/{intervals}/{g}.parquet",
    run:
        df = load_fasta(input[0]).to_frame().reset_index(names="id")
        if len(df) == 0:
            pl.DataFrame(
                {"id": [], "seq": []}, schema={"id": pl.String, "seq": pl.String}
            ).write_parquet(output[0])
        else:
            df.id = df.id.astype(str)
            if config["add_rc"]:
                df = add_rc(df)
            pl.from_pandas(df[["id", "seq"]]).write_parquet(output[0])


rule create_functional_validation:
    """Create validation parquet with phyloP conservation case encoding.

    Sequences are subsampled from the human genome. For each base,
    uppercase iff phyloP >= threshold, lowercase otherwise (NaN -> lowercase).
    """
    input:
        fasta="results/intervals_seq/{recipe}/{w}/{s}/" + VALIDATION_GENOME + ".fa",
        chrom_mapping=local("config/human_chrom_mapping.tsv"),
        bigwig=config["validation"]["conservation_bigwig"],
    output:
        "results/validation/{recipe}/{w}/{s}/validation.parquet",
    run:
        val_config = config["validation"]
        threshold = val_config["phylop_threshold"]

        # Load chrom name mapping (RefSeq -> UCSC)
        chrom_map = dict(
            pl.read_csv(input.chrom_mapping, separator="\t")
            .iter_rows()
        )

        # Load and subsample sequences
        series = load_fasta(input.fasta)
        df = series.to_frame().reset_index(names="id")
        if len(df) == 0:
            pl.DataFrame(
                {"id": [], "seq": []}, schema={"id": pl.String, "seq": pl.String}
            ).write_parquet(output[0])
            return

        df.id = df.id.astype(str)
        max_samples = val_config["max_samples"]

        # Filter to sequences on mapped chromosomes before subsampling
        df["chrom"] = df["id"].apply(lambda x: x.rsplit(":", 1)[0])
        df = df[df["chrom"].isin(chrom_map)]
        df = df.drop(columns=["chrom"])

        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=val_config["seed"])

        bw = pyBigWig.open(input.bigwig)

        def encode_case(row):
            """Encode conservation as case: uppercase iff phyloP >= threshold."""
            chrom_refseq, coords = row["id"].rsplit(":", 1)
            start, end = (int(x) for x in coords.split("-"))
            chrom_ucsc = chrom_map[chrom_refseq]
            scores = bw.values(chrom_ucsc, start, end)
            # NaN (missing data) compares False, so NaN -> lowercase
            return "".join(
                b.upper() if s >= threshold else b.lower()
                for b, s in zip(row["seq"], scores)
            )

        df["seq"] = df.apply(encode_case, axis=1)
        bw.close()

        pl.from_pandas(df[["id", "seq"]]).write_parquet(output[0])


rule merge_datasets:
    input:
        train=lambda wildcards: expand(
            "results/dataset_genome/{intervals}/{g}.parquet",
            intervals=wildcards.intervals,
            g=genome_sets[wildcards.genome_set],
        ),
        validation="results/validation/{intervals}/validation.parquet",
    output:
        temp(local(
            expand(
                "results/dataset/{{genome_set}}/{{intervals}}/data/{split}/{shard}.jsonl",
                split=SPLITS,
                shard=SHARDS,
            )
        )),
    threads: workflow.cores
    run:
        output_by_split = {}
        for path in output:
            split = "validation" if "/validation/" in path else "train"
            output_by_split.setdefault(split, []).append(path)

        for split in SPLITS:
            if split == "train":
                parquets = input.train
            else:
                parquets = [input.validation]
            df = pl.concat(
                tqdm(
                    (pl.read_parquet(path) for path in parquets),
                    total=len(parquets),
                ),
            ).sample(fraction=1, shuffle=True, seed=config["shuffle_seed"])
            split_outputs = output_by_split[split]
            split_pairs = get_array_split_pairs(len(df), len(split_outputs))
            for path, (start, end) in tqdm(
                zip(split_outputs, split_pairs), total=len(split_outputs)
            ):
                df.slice(start, end - start).write_ndjson(path)


rule compress_shard:
    input:
        local("{anything}.jsonl"),
    output:
        local("{anything}.jsonl.zst"),
    threads: 8
    shell:
        "zstd -T{threads} {input} -o {output}"


rule hf_upload:
    input:
        local(expand(
            "results/dataset/{{genome_set}}/{{intervals}}/data/{split}/{shard}.jsonl.zst",
            split=SPLITS,
            shard=SHARDS,
        )),
    output:
        touch("results/upload.done/{genome_set}/{intervals}"),
    params:
        name=lambda wildcards: (
            config["output_hf_prefix"]
            + "-genome_set-" + wildcards.genome_set
            + "-intervals-" + wildcards.intervals.replace("/", "_")
        ),
        data_dir=lambda wildcards: (
            f"results/dataset/{wildcards.genome_set}/{wildcards.intervals}"
        ),
    threads: workflow.cores
    shell:
        "hf upload-large-folder {params.name} --repo-type dataset {params.data_dir}"
