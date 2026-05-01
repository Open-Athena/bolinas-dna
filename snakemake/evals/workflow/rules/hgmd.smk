rule hgmd_download:
    output:
        temp("results/hgmd/variants.csv"),
    params:
        url=config["mendelian_traits"]["hgmd_url"],
    shell:
        """
        mkdir -p results/hgmd && cd results/hgmd && \
        wget {params.url} && \
        tar -xzvf pathogenic_mutations.tar.gz && \
        mv pathogenic_mutations/sei_data/hgmd.raw.matrix.csv variants.csv && \
        rm -rf pathogenic_mutations pathogenic_mutations.tar.gz
        """


rule hgmd_process:
    input:
        "results/hgmd/variants.csv",
    output:
        "results/hgmd/variants.parquet",
    run:
        (
            pl.read_csv(
                input[0],
                columns=["CHROM_hg19", "POS_hg19", "REF", "ALT", "phenotype"],
            )
            .select(
                pl.col("CHROM_hg19").str.replace("chr", "").alias("chrom"),
                pl.col("POS_hg19").alias("pos"),
                pl.col("REF").alias("ref"),
                pl.col("ALT").alias("alt"),
                pl.col("phenotype").alias("trait"),
            )
            .pipe(filter_chroms)
            .pipe(filter_snp)
            .pipe(lift_hg19_to_hg38)
            .filter(pl.col("pos") != -1)
            .sort(COORDINATES)
            .write_parquet(output[0])
        )
