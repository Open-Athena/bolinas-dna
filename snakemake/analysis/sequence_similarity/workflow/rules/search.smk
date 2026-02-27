"""Rules for MMseqs2 search-based leakage analysis.

Searches validation sequences against training sequences using direct pairwise
alignments (no transitive closure). For each val sequence, counts how many
train sequences are similar at each identity x coverage threshold.

``--strand 2`` searches both strands, so reverse complements are detected
without needing to canonicalize sequences at download time.
"""


rule create_search_query_db:
    """Create MMseqs2 database from validation sequences (search query)."""
    input:
        fasta="results/data/{dataset}/validation.fasta",
    output:
        db="results/search/{dataset}/queryDB",
        db_type="results/search/{dataset}/queryDB.dbtype",
    params:
        db_prefix="results/search/{dataset}/queryDB",
    threads: 1
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mmseqs createdb {input.fasta} {params.db_prefix} --mask-lower-case 1
        """


rule create_search_target_db:
    """Create MMseqs2 database from train sequences (search target)."""
    input:
        fasta="results/data/{dataset}/train.fasta",
    output:
        db="results/search/{dataset}/targetDB",
        db_type="results/search/{dataset}/targetDB.dbtype",
    params:
        db_prefix="results/search/{dataset}/targetDB",
    threads: 1
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mmseqs createdb {input.fasta} {params.db_prefix} --mask-lower-case 1
        """


rule search_val_against_train:
    """Search validation sequences against training sequences.

    Reports direct pairwise alignments (no transitive closure).
    --search-type 3 forces nucleotide mode to avoid auto-detection issues.
    --strand 2 searches both forward and reverse complement strands.
    """
    input:
        query_db="results/search/{dataset}/queryDB",
        query_db_type="results/search/{dataset}/queryDB.dbtype",
        target_db="results/search/{dataset}/targetDB",
        target_db_type="results/search/{dataset}/targetDB.dbtype",
    output:
        result_index="results/search/{dataset}/{identity}/{coverage}/resultDB.index",
        result_db_type="results/search/{dataset}/{identity}/{coverage}/resultDB.dbtype",
    params:
        query_prefix="results/search/{dataset}/queryDB",
        target_prefix="results/search/{dataset}/targetDB",
        result_prefix="results/search/{dataset}/{identity}/{coverage}/resultDB",
        tmp_dir="results/search/{dataset}/{identity}/{coverage}/tmp",
        identity=lambda wildcards: float(wildcards.identity),
        coverage=lambda wildcards: float(wildcards.coverage),
        cov_mode=MMSEQS_COV_MODE,
    threads: workflow.cores
    resources:
        mem_mb=64000,
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mkdir -p {params.tmp_dir}
        mmseqs search \
            {params.query_prefix} \
            {params.target_prefix} \
            {params.result_prefix} \
            {params.tmp_dir} \
            --search-type 3 \
            --strand 2 \
            --mask-lower-case 1 \
            --min-seq-id {params.identity} \
            -c {params.coverage} \
            --cov-mode {params.cov_mode} \
            --threads {threads}
        rm -rf {params.tmp_dir}
        """


rule extract_search_hits:
    """Convert binary search result DB to TSV."""
    input:
        query_db="results/search/{dataset}/queryDB",
        query_db_type="results/search/{dataset}/queryDB.dbtype",
        target_db="results/search/{dataset}/targetDB",
        target_db_type="results/search/{dataset}/targetDB.dbtype",
        result_index="results/search/{dataset}/{identity}/{coverage}/resultDB.index",
        result_db_type="results/search/{dataset}/{identity}/{coverage}/resultDB.dbtype",
    output:
        tsv="results/search/{dataset}/{identity}/{coverage}/hits.tsv",
    params:
        query_prefix="results/search/{dataset}/queryDB",
        target_prefix="results/search/{dataset}/targetDB",
        result_prefix="results/search/{dataset}/{identity}/{coverage}/resultDB",
    threads: 1
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mmseqs convertalis \
            {params.query_prefix} \
            {params.target_prefix} \
            {params.result_prefix} \
            {output.tsv} \
            --format-output "query,target,fident,qcov,tcov"
        """


rule compute_leakage_stats:
    """Compute per-dataset leakage statistics from search hits.

    For each validation sequence, count how many training sequences are
    direct alignment hits at the given identity x coverage threshold.
    Also count how many unique training sequences would be filtered
    (have at least one val hit).
    """
    input:
        hits="results/search/{dataset}/{identity}/{coverage}/hits.tsv",
        metadata="results/data/{dataset}/metadata.parquet",
    output:
        stats="results/search/{dataset}/{identity}/{coverage}/stats.parquet",
    run:
        hits = pl.read_csv(
            input.hits,
            separator="\t",
            has_header=False,
            new_columns=["query", "target", "fident", "qcov", "tcov"],
        )

        metadata = pl.read_parquet(input.metadata)
        val_ids = metadata.filter(pl.col("split") == "validation").select("id")
        total_val = val_ids.height
        total_train = metadata.filter(pl.col("split") == "train").height

        # Count distinct train hits per val query
        hits_per_query = (
            hits
            .group_by("query")
            .agg(pl.col("target").n_unique().alias("train_matches"))
        )

        # Left-join with all val sequences so missing queries get 0
        val_matches = (
            val_ids
            .join(hits_per_query, left_on="id", right_on="query", how="left")
            .with_columns(pl.col("train_matches").fill_null(0))
        )

        train_counts = val_matches["train_matches"]

        # Count unique train sequences with at least one val hit
        filtered_train_count = hits["target"].n_unique() if hits.height > 0 else 0
        filtered_train_pct = 100.0 * filtered_train_count / total_train

        stats = pl.DataFrame({
            "dataset": [wildcards.dataset],
            "identity_threshold": [float(wildcards.identity)],
            "coverage_threshold": [float(wildcards.coverage)],
            "total_train": [total_train],
            "total_val": [total_val],
            "train_matches_min": [int(train_counts.min())],
            "train_matches_max": [int(train_counts.max())],
            "train_matches_mean": [float(train_counts.mean())],
            "train_matches_median": [float(train_counts.median())],
            "filtered_train_count": [filtered_train_count],
            "filtered_train_pct": [filtered_train_pct],
        })

        stats.write_parquet(output.stats)
        print(f"\n{wildcards.dataset} @ id={wildcards.identity} cov={wildcards.coverage}:")
        print(f"  Train hits per val seq: min={stats['train_matches_min'][0]}, median={stats['train_matches_median'][0]:.0f}, mean={stats['train_matches_mean'][0]:.1f}, max={stats['train_matches_max'][0]}")
        print(f"  Filtered train seqs: {filtered_train_count:,} / {total_train:,} ({filtered_train_pct:.2f}%)")


rule aggregate_leakage_stats:
    """Aggregate search statistics across all datasets and thresholds."""
    input:
        stats=expand(
            "results/search/{dataset}/{identity}/{coverage}/stats.parquet",
            dataset=get_all_datasets(),
            identity=get_identity_thresholds(),
            coverage=get_coverage_thresholds(),
        ),
    output:
        summary="results/search/summary.parquet",
    run:
        dfs = [pl.read_parquet(f) for f in input.stats]
        summary = pl.concat(dfs)
        summary = summary.sort(["dataset", "coverage_threshold", "identity_threshold"])

        summary.write_parquet(output.summary)
        print("\n=== Leakage Summary ===")
        print(summary.to_pandas().to_string(index=False))


rule plot_train_matches_median:
    """Heatmap of median train matches per val sequence."""
    input:
        summary="results/search/summary.parquet",
    output:
        plot="results/plots/train_matches_median.svg",
    run:
        _plot_train_matches(input.summary, output.plot, "train_matches_median", "Median", ".0f")


rule plot_train_matches_mean:
    """Heatmap of mean train matches per val sequence."""
    input:
        summary="results/search/summary.parquet",
    output:
        plot="results/plots/train_matches_mean.svg",
    run:
        _plot_train_matches(input.summary, output.plot, "train_matches_mean", "Mean", ".1f")


rule plot_pct_filtered_train:
    """Heatmap of % training sequences filtered (have at least one val hit)."""
    input:
        summary="results/search/summary.parquet",
    output:
        plot="results/plots/pct_filtered_train.svg",
    run:
        _plot_train_matches(input.summary, output.plot, "filtered_train_pct", "% Filtered", ".2f")
