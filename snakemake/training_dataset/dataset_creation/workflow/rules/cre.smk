rule cre_download:
    output:
        temp("results/cre/all.tsv"),
    shell:
        "wget -O {output} https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed"


rule cre_process:
    input:
        "results/cre/all.tsv",
    output:
        "results/cre/all.parquet",
    run:
        (
            pl.read_csv(
                input[0],
                separator="\t",
                has_header=False,
                columns=[0, 1, 2, 5],
                new_columns=["chrom", "start", "end", "cre_class"],
            )
            .with_columns(pl.col("chrom").str.replace("chr", ""))
            .filter(pl.col("chrom").is_in(STANDARD_CHROMS))
            .write_parquet(output[0])
        )


rule cre_filter_enhancers:
    input:
        "results/cre/all.parquet",
    output:
        "results/cre/ELS.parquet",
    run:
        (
            pl.read_parquet(input[0])
            .filter(pl.col("cre_class").is_in(ENHANCER_CRE_CLASSES))
            .select(["chrom", "start", "end"])
            .write_parquet(output[0])
        )
