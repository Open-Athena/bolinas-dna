rule download_genome:
    output:
        "results/genome/{species}.fa.gz",
    params:
        url=lambda wildcards: config["genome_urls"][wildcards.species],
    shell:
        "wget -O {output} {params.url}"


rule genome_to_2bit:
    input:
        "results/genome/{species}.fa.gz",
    output:
        "results/genome/{species}.2bit",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "zcat {input} | faToTwoBit stdin {output}"


rule chrom_sizes:
    input:
        "results/genome/{species}.2bit",
    output:
        "results/genome/{species}.chrom.sizes",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} {output}"


rule chrom_sizes_filtered:
    input:
        "results/genome/{species}.chrom.sizes",
    output:
        "results/genome/{species}.chrom.sizes.filtered",
    run:
        standard = config["standard_chroms"][wildcards.species]
        df = pd.read_csv(input[0], sep="\t", header=None, names=["chrom", "size"])
        df[df["chrom"].isin(standard)].to_csv(
            output[0], sep="\t", header=False, index=False
        )


rule genome_bed:
    input:
        "results/genome/{species}.chrom.sizes",
    output:
        "results/genome/{species}.genome.bed",
    run:
        standard = config["standard_chroms"][wildcards.species]
        df = pd.read_csv(input[0], sep="\t", header=None, names=["chrom", "size"])
        df = df[df["chrom"].isin(standard)]
        df["start"] = 0
        df.rename(columns={"size": "end"})[["chrom", "start", "end"]].to_csv(
            output[0], sep="\t", header=False, index=False
        )


rule undefined_regions:
    input:
        "results/genome/{species}.2bit",
    output:
        "results/genome/{species}.undefined.bed",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} /dev/stdout -nBed > {output}"


rule download_cre:
    output:
        temp("results/cre/{species}/raw.tsv"),
    params:
        url=lambda wildcards: config["cre_urls"][wildcards.species],
    shell:
        "wget -O {output} {params.url}"


rule process_cre:
    input:
        "results/cre/{species}/raw.tsv",
    output:
        "results/cre/{species}/all.parquet",
    run:
        standard = config["standard_chroms"][wildcards.species]
        (
            pl.read_csv(
                input[0],
                separator="\t",
                has_header=False,
                columns=[0, 1, 2, 5],
                new_columns=["chrom", "start", "end", "cre_class"],
            )
            .with_columns(pl.col("chrom").str.replace("chr", ""))
            .filter(pl.col("chrom").is_in(standard))
            .write_parquet(output[0])
        )


rule filter_enhancers:
    input:
        "results/cre/{species}/all.parquet",
    output:
        "results/cre/{species}/ELS.parquet",
    run:
        (
            pl.read_parquet(input[0])
            .filter(pl.col("cre_class").is_in(ENHANCER_CRE_CLASSES))
            .select(["chrom", "start", "end"])
            .write_parquet(output[0])
        )


rule download_conservation:
    output:
        "results/conservation/{species}/{conservation}.bw",
    params:
        url=lambda wildcards: config["conservation"][wildcards.species][wildcards.conservation]["url"],
    shell:
        "wget -O {output} {params.url}"


rule cre_conservation:
    input:
        cre="results/cre/{species}/ELS.parquet",
        conservation="results/conservation/{species}/{conservation}.bw",
    output:
        "results/cre/{species}/ELS_conservation/{conservation}.parquet",
    run:
        threshold = config["conservation"][wildcards.species][wildcards.conservation]["threshold"]
        conservation_window = config["conservation_window"]
        df = pl.read_parquet(input.cre)

        # Resize to center conservation_window bp for scoring
        size = df["end"] - df["start"]
        diff = conservation_window - size
        left_adj = diff // 2
        right_adj = diff - left_adj
        scored = df.with_columns(
            (pl.col("start") - left_adj).alias("score_start"),
            (pl.col("end") + right_adj).alias("score_end"),
        )

        bw = pyBigWig.open(input.conservation)
        stats = scored.select(
            pl.struct(["chrom", "score_start", "score_end"]).map_elements(
                lambda x: {
                    "total_bases": x["score_end"] - x["score_start"],
                    "conserved_bases": int(
                        np.sum(
                            bw.values(
                                "chr" + x["chrom"], x["score_start"], x["score_end"], numpy=True
                            )
                            >= threshold
                        )
                    ),
                },
                return_dtype=pl.Struct(
                    {"total_bases": pl.Int64, "conserved_bases": pl.Int64}
                ),
            )
        ).unnest("chrom")
        bw.close()

        result = df.hstack(stats).with_columns(
            (pl.col("conserved_bases") / pl.col("total_bases"))
            .cast(pl.Float32)
            .alias("pct_conserved")
        )
        result.write_parquet(output[0])


rule plot_conservation:
    input:
        "results/cre/{species}/ELS_conservation/{conservation}.parquet",
    output:
        "results/plots/conservation/{species}/{conservation}.svg",
    run:
        df = pl.read_parquet(input[0])
        threshold = config["conservation"][wildcards.species][wildcards.conservation]["threshold"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.hist(df["conserved_bases"].to_numpy(), bins=50, edgecolor="black", linewidth=0.5)
        ax1.set_xlabel("Conserved bases")
        ax1.set_ylabel("Count")
        ax1.set_title(f"Conserved bases (threshold={threshold})")

        ax2.hist(df["pct_conserved"].to_numpy(), bins=50, edgecolor="black", linewidth=0.5)
        ax2.set_xlabel("Proportion conserved")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Proportion conserved (threshold={threshold})")

        fig.suptitle(f"{wildcards.conservation} — {wildcards.species}")
        fig.tight_layout()
        fig.savefig(output[0])
        plt.close(fig)


rule cre_filter_conserved:
    input:
        "results/cre/{species}/ELS_conservation/{conservation}.parquet",
    output:
        "results/cre/{species}/ELS_conserved/{conservation}/{n}.parquet",
    run:
        min_conserved = int(wildcards.n)
        (
            pl.read_parquet(input[0])
            .filter(pl.col("conserved_bases") >= min_conserved)
            .select(["chrom", "start", "end"])
            .write_parquet(output[0])
        )


rule make_positives:
    input:
        intervals="results/cre/{species}/{intervals}.parquet",
        genome="results/genome/{species}.genome.bed",
        undefined="results/genome/{species}.undefined.bed",
    output:
        "results/intervals/{intervals}/{species}/positives.bed",
    run:
        window_size = config["window_size"]

        intervals = GenomicSet(pl.read_parquet(input.intervals))
        intervals = intervals.resize(window_size)

        genome = GenomicSet.read_bed(input.genome)
        undefined = GenomicSet.read_bed(input.undefined)
        defined = genome - undefined
        intervals = intervals & defined
        intervals = intervals.filter_size(min_size=window_size, max_size=window_size)

        intervals.write_bed(output[0])


rule make_exclusion:
    input:
        positives="results/intervals/{intervals}/{species}/positives.bed",
        undefined="results/genome/{species}.undefined.bed",
    output:
        temp("results/intervals/{intervals}/{species}/exclusion.bed"),
    run:
        pos = GenomicSet.read_bed(input.positives)
        undef = GenomicSet.read_bed(input.undefined)
        (pos | undef).write_bed(output[0])


rule sample_negatives:
    input:
        positives="results/intervals/{intervals}/{species}/positives.bed",
        exclusion="results/intervals/{intervals}/{species}/exclusion.bed",
        chrom_sizes="results/genome/{species}.chrom.sizes.filtered",
    output:
        "results/intervals/{intervals}/{species}/negatives.bed",
    params:
        seed=config["seed"],
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        """
        bedtools shuffle \
            -i {input.positives} \
            -g {input.chrom_sizes} \
            -excl {input.exclusion} \
            -noOverlapping \
            -chrom \
            -seed {params.seed} \
            > {output}
        """


rule prepare_bed_for_seq:
    input:
        "results/intervals/{intervals}/{species}/{label_type}.bed",
    output:
        temp(local("results/intervals/{intervals}/{species}/{label_type}.4col.bed")),
    shell:
        """
        awk 'BEGIN {{OFS="\\t"}} {{print $1, $2, $3, "."}}' {input} > {output}
        """


rule extract_sequences:
    input:
        twobit="results/genome/{species}.2bit",
        bed=local("results/intervals/{intervals}/{species}/{label_type}.4col.bed"),
    output:
        temp(local("results/sequences/{intervals}/{species}/{label_type}.fa")),
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitToFa {input.twobit} {output} -bed={input.bed} -bedPos"


rule make_species_parquet:
    input:
        pos=local("results/sequences/{intervals}/{species}/positives.fa"),
        neg=local("results/sequences/{intervals}/{species}/negatives.fa"),
    output:
        "results/parquet/{intervals}/{species}.parquet",
    run:
        window_size = config["window_size"]

        pos = fasta_to_df(input.pos, label=1, genome=wildcards.species)
        neg = fasta_to_df(input.neg, label=0, genome=wildcards.species)
        combined = pd.concat([pos, neg], ignore_index=True)

        assert (combined["seq"].str.len() == window_size).all(), "Not all seqs are 255bp"

        pl.from_pandas(combined).write_parquet(output[0])


rule build_dataset:
    input:
        lambda wc: [
            f"results/parquet/"
            f"{config['datasets'][wc.dataset]['intervals']}/{species}.parquet"
            for species, splits in (
                config["splits"][config["datasets"][wc.dataset]["split"]].items()
            )
            if wc.split_name in splits
        ],
    output:
        "results/dataset/{dataset}/{split_name}.parquet",
    run:
        seed = config["seed"]
        dataset_config = config["datasets"][wildcards.dataset]
        split_config = config["splits"][dataset_config["split"]]

        dfs = []
        for parquet_path in input:
            species = parquet_path.split("/")[-1].removesuffix(".parquet")
            chroms = split_config[species][wildcards.split_name]
            df = pl.read_parquet(parquet_path).to_pandas()
            dfs.append(df[df["chrom"].isin(chroms)])

        combined = pd.concat(dfs, ignore_index=True)

        # Add reverse complement for training splits
        if wildcards.split_name == TRAIN_SPLIT:
            combined["id"] = ""
            combined = add_rc(combined)
            combined["strand"] = combined["id"].str[-1]
            combined = combined.drop(columns=["id"])

        # Subsample if configured for this split
        max_samples = config.get("max_samples", {}).get(wildcards.split_name)
        if max_samples and len(combined) > max_samples:
            combined = combined.sample(n=max_samples, random_state=seed)

        # Shuffle and write
        combined = (
            combined.sample(frac=1, random_state=seed).reset_index(drop=True)
        )
        pl.from_pandas(combined).write_parquet(output[0])
