"""mmseqs2 nucleotide search of hg38 query dELS against the mm10 ZRS window.

Flag set lifted from snakemake/analysis/sequence_similarity/workflow/rules/search.smk:
- --search-type 3 forces nucleotide mode (no auto-detect surprises)
- --strand 2 searches forward + reverse complement targets
- --mask-lower-case 1 excludes soft-masked repeats from k-mer seeding
"""


rule create_query_db:
    input:
        fasta="results/cre/hg38/query.filtered.fasta",
    output:
        db="results/search/queryDB",
        dbtype="results/search/queryDB.dbtype",
    threads: 1
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        "mmseqs createdb {input.fasta} {output.db} --mask-lower-case 1"


rule create_target_db:
    input:
        fasta="results/target/mm10_window.fasta",
    output:
        db="results/search/targetDB",
        dbtype="results/search/targetDB.dbtype",
    threads: 1
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        "mmseqs createdb {input.fasta} {output.db} --mask-lower-case 1"


rule search:
    input:
        query_db="results/search/queryDB",
        query_dbtype="results/search/queryDB.dbtype",
        target_db="results/search/targetDB",
        target_dbtype="results/search/targetDB.dbtype",
    output:
        result_index="results/search/resultDB.index",
        result_dbtype="results/search/resultDB.dbtype",
    params:
        result_prefix="results/search/resultDB",
        tmp_dir="results/search/tmp",
        sensitivity=config["mmseqs2"]["sensitivity"],
        max_accept=config["mmseqs2"]["max_accept"],
    threads: workflow.cores
    resources:
        mem_mb=14000,
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mkdir -p {params.tmp_dir}
        mmseqs search \
            {input.query_db} \
            {input.target_db} \
            {params.result_prefix} \
            {params.tmp_dir} \
            --search-type 3 \
            --strand 2 \
            --mask-lower-case 1 \
            --split-memory-limit 12G \
            -s {params.sensitivity} \
            --max-accept {params.max_accept} \
            --threads {threads}
        rm -rf {params.tmp_dir}
        """


rule convertalis:
    input:
        query_db="results/search/queryDB",
        target_db="results/search/targetDB",
        result_index="results/search/resultDB.index",
    output:
        tsv="results/search/hits.tsv",
    params:
        result_prefix="results/search/resultDB",
    threads: 1
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mmseqs convertalis \
            {input.query_db} \
            {input.target_db} \
            {params.result_prefix} \
            {output.tsv} \
            --format-output "query,target,tstart,tend,bits,evalue,fident,qcov,tcov"
        """


rule normalize_mmseqs2_hits:
    """Project mmseqs2 convertalis output to the aligner-agnostic unified schema.

    Unified schema (tab-separated, with header):
        query  hit_chrom  hit_start  hit_end  rev_strand  score  fident  evalue  qcov  tcov

    Coordinates are absolute 0-based half-open mm10 BED coords.
    """
    input:
        "results/search/hits.tsv",
    output:
        "results/align/mmseqs2/hits.tsv",
    run:
        chrom, win_start, _ = get_search_window("mm10")
        raw = pl.read_csv(
            input[0],
            separator="\t",
            has_header=False,
            new_columns=[
                "query",
                "target",
                "tstart",
                "tend",
                "bits",
                "evalue",
                "fident",
                "qcov",
                "tcov",
            ],
        )
        unified = raw.select(
            pl.col("query"),
            pl.lit(chrom).alias("hit_chrom"),
            (pl.min_horizontal("tstart", "tend") - 1 + win_start).alias("hit_start"),
            pl.max_horizontal("tstart", "tend").add(win_start).alias("hit_end"),
            (pl.col("tend") < pl.col("tstart")).alias("rev_strand"),
            pl.col("bits").alias("score"),
            pl.col("fident"),
            pl.col("evalue"),
            pl.col("qcov"),
            pl.col("tcov"),
        )
        unified.write_csv(output[0], separator="\t", include_header=True)
