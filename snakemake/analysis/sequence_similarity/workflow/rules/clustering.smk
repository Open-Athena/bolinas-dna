"""Rules for MMseqs2 sequence clustering."""


rule create_mmseqs_db:
    """Create MMseqs2 database from FASTA file."""
    input:
        fasta="results/data/{dataset}/all_sequences.fasta",
    output:
        db="results/mmseqs/{dataset}/seqDB",
        db_type="results/mmseqs/{dataset}/seqDB.dbtype",
    params:
        db_prefix="results/mmseqs/{dataset}/seqDB",
    threads: 1
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mmseqs createdb {input.fasta} {params.db_prefix}
        """


rule cluster_sequences:
    """Cluster sequences using MMseqs2 linclust (linear time clustering).

    This uses linclust for scalability - it can handle billions of sequences
    in linear time O(N).
    """
    input:
        db="results/mmseqs/{dataset}/seqDB",
        db_type="results/mmseqs/{dataset}/seqDB.dbtype",
    output:
        cluster_db="results/mmseqs/{dataset}/clusters_{identity}/clusterDB",
        cluster_db_type="results/mmseqs/{dataset}/clusters_{identity}/clusterDB.dbtype",
    params:
        db_prefix="results/mmseqs/{dataset}/seqDB",
        cluster_prefix="results/mmseqs/{dataset}/clusters_{identity}/clusterDB",
        tmp_dir="results/mmseqs/{dataset}/tmp_{identity}",
        identity=lambda wildcards: float(wildcards.identity),
        coverage=config["mmseqs2"]["coverage"],
        cov_mode=config["mmseqs2"]["cov_mode"],
        cluster_mode=config["mmseqs2"]["cluster_mode"],
    threads: config["mmseqs2"]["threads"]
    resources:
        mem_mb=64000,
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mkdir -p {params.tmp_dir}
        mmseqs linclust \
            {params.db_prefix} \
            {params.cluster_prefix} \
            {params.tmp_dir} \
            --min-seq-id {params.identity} \
            -c {params.coverage} \
            --cov-mode {params.cov_mode} \
            --cluster-mode {params.cluster_mode} \
            --threads {threads}
        rm -rf {params.tmp_dir}
        """


rule extract_cluster_tsv:
    """Extract cluster assignments to TSV format."""
    input:
        db="results/mmseqs/{dataset}/seqDB",
        db_type="results/mmseqs/{dataset}/seqDB.dbtype",
        cluster_db="results/mmseqs/{dataset}/clusters_{identity}/clusterDB",
        cluster_db_type="results/mmseqs/{dataset}/clusters_{identity}/clusterDB.dbtype",
    output:
        tsv="results/clustering/{dataset}/clusters_{identity}.tsv",
    params:
        db_prefix="results/mmseqs/{dataset}/seqDB",
        cluster_prefix="results/mmseqs/{dataset}/clusters_{identity}/clusterDB",
    threads: 1
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mmseqs createtsv \
            {params.db_prefix} \
            {params.db_prefix} \
            {params.cluster_prefix} \
            {output.tsv}
        """
