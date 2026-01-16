rule cre_download:
    output:
        temp("results/intervals/cre.tsv"),
    shell:
        "wget -O {output} https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed"


rule cre_process:
    input:
        "results/intervals/cre.tsv",
    output:
        "results/intervals/cre.parquet",
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
            .filter(pl.col("chrom").is_in(CHROMS))
            .write_parquet(output[0])
        )
