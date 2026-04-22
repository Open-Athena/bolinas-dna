"""mmseqs2 nucleotide search of hg38 query dELS against the mm10 target.

Flag set lifted from snakemake/analysis/sequence_similarity/workflow/rules/search.smk:
- --search-type 3 forces nucleotide mode (no auto-detect surprises)
- --strand 2 searches forward + reverse complement targets
- --mask-lower-case 1 excludes soft-masked repeats from k-mer seeding

The query side carries the `{flank}` wildcard (symmetric with align_minimap2.smk);
the target side is flank-independent (`mm10_window.fasta` is one FASTA per run).
"""


rule create_query_db:
    input:
        fasta="results/cre/hg38/flank_{flank}/query.filtered.fasta",
    output:
        db="results/search/flank_{flank}/queryDB",
        dbtype="results/search/flank_{flank}/queryDB.dbtype",
    wildcard_constraints:
        flank=r"-?\d+",
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
        query_db="results/search/flank_{flank}/queryDB",
        query_dbtype="results/search/flank_{flank}/queryDB.dbtype",
        target_db="results/search/targetDB",
        target_dbtype="results/search/targetDB.dbtype",
    output:
        result_index="results/search/flank_{flank}/resultDB.index",
        result_dbtype="results/search/flank_{flank}/resultDB.dbtype",
    params:
        result_prefix="results/search/flank_{flank}/resultDB",
        tmp_dir="results/search/flank_{flank}/tmp",
        sensitivity=config["mmseqs2"]["sensitivity"],
        max_accept=config["mmseqs2"]["max_accept"],
        split_memory_limit=config["mmseqs2"].get("split_memory_limit", "12G"),
    wildcard_constraints:
        flank=r"-?\d+",
    threads: workflow.cores
    resources:
        mem_mb=config["mmseqs2"].get("mem_mb", 14000),
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
            --split-memory-limit {params.split_memory_limit} \
            -s {params.sensitivity} \
            --max-accept {params.max_accept} \
            --threads {threads}
        rm -rf {params.tmp_dir}
        """


rule convertalis:
    input:
        query_db="results/search/flank_{flank}/queryDB",
        target_db="results/search/targetDB",
        result_index="results/search/flank_{flank}/resultDB.index",
    output:
        tsv="results/search/flank_{flank}/hits.tsv",
    params:
        result_prefix="results/search/flank_{flank}/resultDB",
    wildcard_constraints:
        flank=r"-?\d+",
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

    `hit_chrom` is taken from mmseqs's per-hit `target` column (which equals
    the FASTA record name = the chromosome name, since make_target_window_bed
    names every record after its chrom). `win_start` adds the windowed-mode
    offset; it is 0 in whole-chrom and whole-genome modes.
    """
    input:
        "results/search/flank_{flank}/hits.tsv",
    output:
        "results/align/mmseqs2/flank_{flank}/hits.tsv",
    wildcard_constraints:
        flank=r"-?\d+",
    run:
        _, win_start, _ = get_search_window("mm10")
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
            pl.col("target").alias("hit_chrom"),
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
