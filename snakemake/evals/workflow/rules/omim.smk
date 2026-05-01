rule clinvar_download:
    output:
        temp("results/clinvar/all.vcf.gz"),
        temp("results/clinvar/all.vcf.gz.tbi"),
    params:
        vcf_url=lambda wc: f"https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/clinvar_{config['mendelian_traits']['clinvar_release']}.vcf.gz",
    shell:
        "wget {params.vcf_url} -O {output[0]} && wget {params.vcf_url}.tbi -O {output[1]}"


rule clinvar_process:
    input:
        "results/clinvar/all.vcf.gz",
        "results/clinvar/all.vcf.gz.tbi",
    output:
        temp("results/clinvar/all.parquet"),
    run:
        rows = []
        for variant in VCF(input[0]):
            if variant.INFO.get("CLNVC") != "single_nucleotide_variant":
                continue
            if len(variant.ALT) != 1:
                continue
            rows.append(
                [
                    variant.CHROM,
                    variant.POS,
                    variant.REF,
                    variant.ALT[0],
                    int(variant.ID),
                ]
            )
        (
            pl.DataFrame(rows, schema=COORDINATES + ["clinvar_id"], orient="row")
            .pipe(filter_snp)
            .pipe(filter_chroms)
            .sort(COORDINATES)
            .write_parquet(output[0])
        )


rule clinvar_submission_summary_download:
    output:
        temp("results/clinvar/submission_summary.txt.gz"),
    params:
        url=lambda wc: f"https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/submission_summary_{config['mendelian_traits']['submission_summary_date']}.txt.gz",
    shell:
        "wget {params.url} -O {output}"


rule clinvar_submission_summary_process:
    input:
        "results/clinvar/submission_summary.txt.gz",
    output:
        temp("results/clinvar/submission_summary.parquet"),
    run:
        # Using pandas due to malformed rows in this file
        pd.read_csv(input[0], sep="\t", skiprows=18).to_parquet(output[0], index=False)


rule clinvar_submission_summary_omim:
    input:
        "results/clinvar/submission_summary.parquet",
    output:
        temp("results/clinvar/submission_summary_omim.parquet"),
    run:
        df = (
            pl.read_parquet(
                input[0],
                columns=[
                    "#VariationID",
                    "ClinicalSignificance",
                    "SubmittedPhenotypeInfo",
                    "ReportedPhenotypeInfo",
                    "CollectionMethod",
                    "Submitter",
                ],
            )
            .rename({"#VariationID": "clinvar_id"})
            .filter(
                pl.col("Submitter") == "OMIM",
                pl.col("CollectionMethod") == "literature only",
            )
            .drop(["CollectionMethod", "Submitter"])
        )
        unambiguous_ids = (
            df.group_by("clinvar_id")
            .n_unique()
            .filter(pl.col("ClinicalSignificance") == 1)["clinvar_id"]
        )
        df = df.filter(
            pl.col("clinvar_id").is_in(unambiguous_ids),
            pl.col("ClinicalSignificance") == "Pathogenic",
            ~pl.col("SubmittedPhenotypeInfo").str.contains("RECLASSIFIED"),
        ).drop(["ClinicalSignificance", "SubmittedPhenotypeInfo"])
        df = (
            df.group_by("clinvar_id")
            .agg(pl.col("ReportedPhenotypeInfo").unique(maintain_order=True))
            .with_columns(pl.col("ReportedPhenotypeInfo").list.join("|").alias("trait"))
            .drop("ReportedPhenotypeInfo")
        )
        df.write_parquet(output[0])


rule clinvar_omim:
    input:
        "results/clinvar/all.parquet",
        "results/clinvar/submission_summary_omim.parquet",
    output:
        "results/omim/variants.parquet",
    run:
        V = pl.read_parquet(input[0])
        df = pl.read_parquet(input[1])
        V = V.join(df, on="clinvar_id", how="inner").sort(COORDINATES)
        V.write_parquet(output[0])
