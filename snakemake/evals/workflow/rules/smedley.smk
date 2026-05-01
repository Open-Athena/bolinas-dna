# Regulatory variants associated with Mendelian diseases curated from the
# literature by Smedley et al. (2016), Table S6:
# Smedley, Damian, et al. "A whole-genome analysis framework for effective
# identification of pathogenic regulatory variants in Mendelian disease."
# AJHG 99.3 (2016): 595-606.


rule smedley_download:
    output:
        local(temp("results/smedley/variants.xlsx")),
    params:
        url=config["mendelian_traits"]["smedley_url"],
    shell:
        "wget -O {output} {params.url}"


rule smedley_process:
    input:
        local("results/smedley/variants.xlsx"),
    output:
        "results/smedley/variants.parquet",
    run:
        # Using pandas for Excel reading (polars Excel support is limited)
        xls = pd.ExcelFile(input[0])
        dfs = [pd.read_excel(input[0], sheet_name=name) for name in xls.sheet_names[1:]]
        (
            pl.from_pandas(pd.concat(dfs))
            .select(
                pl.col("Chr").str.replace("chr", "").alias("chrom"),
                pl.col("Position").alias("pos"),
                pl.col("Ref").alias("ref"),
                pl.col("Alt").alias("alt"),
                pl.col("OMIM").alias("trait"),
            )
            .pipe(filter_chroms)
            .pipe(filter_snp)
            .pipe(lift_hg19_to_hg38)
            .filter(pl.col("pos") != -1)
            .sort(COORDINATES)
            .write_parquet(output[0])
        )
