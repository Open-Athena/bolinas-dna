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
        mmseqs createdb {input.fasta} {params.db_prefix} --mask-lower-case 1
        """


rule cluster_sequences:
    """Cluster sequences using MMseqs2 cluster with lowercase repeat masking."""
    input:
        db="results/mmseqs/{dataset}/seqDB",
        db_type="results/mmseqs/{dataset}/seqDB.dbtype",
    output:
        cluster_db="results/mmseqs/{dataset}/clusters_id{identity}_cov{coverage}/clusterDB.index",
        cluster_db_type="results/mmseqs/{dataset}/clusters_id{identity}_cov{coverage}/clusterDB.dbtype",
    params:
        db_prefix="results/mmseqs/{dataset}/seqDB",
        cluster_prefix="results/mmseqs/{dataset}/clusters_id{identity}_cov{coverage}/clusterDB",
        tmp_dir="results/mmseqs/{dataset}/tmp_id{identity}_cov{coverage}",
        identity=lambda wildcards: float(wildcards.identity),
        coverage=lambda wildcards: float(wildcards.coverage),
        cov_mode=MMSEQS_COV_MODE,
        cluster_mode=MMSEQS_CLUSTER_MODE,
    threads: workflow.cores
    resources:
        mem_mb=64000,
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mkdir -p {params.tmp_dir}
        mmseqs cluster \
            {params.db_prefix} \
            {params.cluster_prefix} \
            {params.tmp_dir} \
            --mask-lower-case 1 \
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
        cluster_db="results/mmseqs/{dataset}/clusters_id{identity}_cov{coverage}/clusterDB.index",
        cluster_db_type="results/mmseqs/{dataset}/clusters_id{identity}_cov{coverage}/clusterDB.dbtype",
    output:
        tsv="results/clustering/{dataset}/clusters_id{identity}_cov{coverage}.tsv",
    params:
        db_prefix="results/mmseqs/{dataset}/seqDB",
        cluster_prefix="results/mmseqs/{dataset}/clusters_id{identity}_cov{coverage}/clusterDB",
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
