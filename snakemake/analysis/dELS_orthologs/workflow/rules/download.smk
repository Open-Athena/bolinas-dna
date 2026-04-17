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


rule restrict_hg38_dels_to_gold_standard:
    """hg38 only: keep dELS that appear in the Cactus-derived gold-standard TSV.

    Queries without a gold-standard partner cannot contribute to recall
    against the gold standard — including them would just be compute waste
    (aligning ~1.1 M extra dELS whose results are vacuous for the metric).
    The `per_query_report` filter that restricts eval to "hg38 partner in
    query set AND mm10 partner in candidate pool" would drop them at eval
    time anyway; this just moves the filter up to the pipeline boundary.
    """
    input:
        dels="results/cre/hg38/dels.parquet",
        gold="results/orthologs/hg38_mm10.tsv",
    output:
        "results/cre/hg38/dels_in_gold.parquet",
    run:
        gold_hg38 = (
            pl.read_csv(
                input.gold,
                separator="\t",
                has_header=False,
                new_columns=["hg38", "mm10"],
            )["hg38"]
            .unique()
            .to_list()
        )
        df = pl.read_parquet(input.dels).filter(pl.col("accession").is_in(gold_hg38))
        df.write_parquet(output[0])
        print(f"  hg38: {df.height} dELS have a gold-standard mm10 partner")


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


rule chrom_sizes:
    input:
        "results/genome/{species}.2bit",
    output:
        "results/genome/{species}.chrom.sizes",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} {output}"
