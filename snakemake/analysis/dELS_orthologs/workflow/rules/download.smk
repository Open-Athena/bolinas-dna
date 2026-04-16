"""Download cCRE BEDs, ortholog gold standard, and genome 2bit files."""


rule download_cre:
    output:
        temp("results/cre/{species}/raw.bed"),
    params:
        url=lambda wildcards: config["cre_urls"][wildcards.species],
    shell:
        "wget -O {output} {params.url}"


rule process_cre:
    """Parse Registry-V4 BED (cols: chrom, start, end, rDHS_acc, cCRE_acc, class).

    Keep the cCRE accession (col 4) — it joins against the ortholog TSV.
    """
    input:
        "results/cre/{species}/raw.bed",
    output:
        "results/cre/{species}/cres.parquet",
    run:
        df = pl.read_csv(
            input[0],
            separator="\t",
            has_header=False,
            columns=[0, 1, 2, 4, 5],
            new_columns=["chrom", "start", "end", "accession", "cre_class"],
        )
        df.write_parquet(output[0])


rule filter_cres_to_dels:
    """Restrict to the configured cCRE class (`dELS` per the analysis scope).

    See https://github.com/Open-Athena/bolinas-dna/issues/120 for the dELS-only
    motivation.
    """
    input:
        "results/cre/{species}/cres.parquet",
    output:
        "results/cre/{species}/dels.parquet",
    params:
        cre_class=config["cre_class"],
    run:
        df = pl.read_parquet(input[0]).filter(pl.col("cre_class") == params.cre_class)
        df.write_parquet(output[0])
        print(f"  {wildcards.species}: {df.height} {params.cre_class} cCREs")


rule download_orthologs:
    output:
        "results/orthologs/hg38_mm10.tsv",
    params:
        url=config["ortholog_url"],
    shell:
        "wget -O {output} {params.url}"


rule download_genome_2bit:
    output:
        "results/genome/{species}.2bit",
    params:
        url=lambda wildcards: config["genome_urls"][wildcards.species],
    shell:
        "wget -O {output} {params.url}"
