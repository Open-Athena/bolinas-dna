"""Rules for analyzing clustering results and computing leakage statistics."""


# =============================================================================
# Sanity Check: Validation self-similarity
# =============================================================================
# This verifies the pipeline works correctly by clustering validation sequences
# against themselves. Expected results:
# - With RC augmentation: each sequence should cluster with its reverse complement
# - With 128bp overlap: adjacent windows should have ~50% coverage similarity
# =============================================================================


rule create_validation_only_db:
    """Create MMseqs2 database from validation sequences only (for sanity check)."""
    input:
        fasta="results/data/{dataset}/validation.fasta",
    output:
        db="results/mmseqs/{dataset}/valDB",
        db_type="results/mmseqs/{dataset}/valDB.dbtype",
    params:
        db_prefix="results/mmseqs/{dataset}/valDB",
    threads: 1
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mmseqs createdb {input.fasta} {params.db_prefix} --mask-lower-case 1
        """


rule cluster_validation_self:
    """Cluster validation sequences against themselves (sanity check)."""
    input:
        db="results/mmseqs/{dataset}/valDB",
        db_type="results/mmseqs/{dataset}/valDB.dbtype",
    output:
        cluster_db="results/mmseqs/{dataset}/val_self_id{identity}_cov{coverage}/clusterDB.index",
        cluster_db_type="results/mmseqs/{dataset}/val_self_id{identity}_cov{coverage}/clusterDB.dbtype",
    params:
        db_prefix="results/mmseqs/{dataset}/valDB",
        cluster_prefix="results/mmseqs/{dataset}/val_self_id{identity}_cov{coverage}/clusterDB",
        tmp_dir="results/mmseqs/{dataset}/val_tmp_id{identity}_cov{coverage}",
        identity=lambda wildcards: float(wildcards.identity),
        coverage=lambda wildcards: float(wildcards.coverage),
        cov_mode=MMSEQS_COV_MODE,
        cluster_mode=MMSEQS_CLUSTER_MODE,
    threads: workflow.cores
    resources:
        mem_mb=32000,
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


rule extract_validation_self_clusters:
    """Extract validation self-clustering results to TSV."""
    input:
        db="results/mmseqs/{dataset}/valDB",
        db_type="results/mmseqs/{dataset}/valDB.dbtype",
        cluster_db="results/mmseqs/{dataset}/val_self_id{identity}_cov{coverage}/clusterDB.index",
        cluster_db_type="results/mmseqs/{dataset}/val_self_id{identity}_cov{coverage}/clusterDB.dbtype",
    output:
        tsv="results/sanity_check/{dataset}/val_self_id{identity}_cov{coverage}.tsv",
    params:
        db_prefix="results/mmseqs/{dataset}/valDB",
        cluster_prefix="results/mmseqs/{dataset}/val_self_id{identity}_cov{coverage}/clusterDB",
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


rule analyze_validation_self_similarity:
    """Analyze validation self-clustering results (sanity check).

    Expected results:
    - At 100% identity: sequences should cluster with their reverse complements
    - At lower identity with coverage: adjacent sliding windows should cluster
    """
    input:
        clusters="results/sanity_check/{dataset}/val_self_id{identity}_cov{coverage}.tsv",
        metadata="results/data/{dataset}/metadata.parquet",
    output:
        stats="results/sanity_check/{dataset}/val_self_stats_id{identity}_cov{coverage}.parquet",
    run:
        # Load cluster assignments
        clusters = pl.read_csv(
            input.clusters,
            separator="\t",
            has_header=False,
            new_columns=["representative", "member"],
        )

        # Load metadata
        metadata = pl.read_parquet(input.metadata)
        val_metadata = metadata.filter(pl.col("split") == "validation")
        total_val = val_metadata.height

        # Compute cluster statistics
        cluster_sizes = (
            clusters
            .group_by("representative")
            .agg(pl.count().alias("cluster_size"))
        )

        n_clusters = cluster_sizes.height
        n_singletons = cluster_sizes.filter(pl.col("cluster_size") == 1).height
        n_non_singletons = n_clusters - n_singletons

        # Sequences in non-singleton clusters (have at least one similar sequence)
        seqs_with_matches = total_val - n_singletons
        pct_with_matches = (seqs_with_matches / total_val * 100) if total_val > 0 else 0

        # Average cluster size (excluding singletons)
        non_singleton_clusters = cluster_sizes.filter(pl.col("cluster_size") > 1)
        avg_cluster_size = (
            non_singleton_clusters["cluster_size"].mean()
            if non_singleton_clusters.height > 0 else 0
        )
        max_cluster_size = cluster_sizes["cluster_size"].max()

        stats = pl.DataFrame({
            "dataset": [wildcards.dataset],
            "identity_threshold": [float(wildcards.identity)],
            "coverage_threshold": [float(wildcards.coverage)],
            "total_val_sequences": [total_val],
            "n_clusters": [n_clusters],
            "n_singletons": [n_singletons],
            "n_non_singleton_clusters": [n_non_singletons],
            "seqs_with_matches": [seqs_with_matches],
            "pct_with_matches": [pct_with_matches],
            "avg_cluster_size": [avg_cluster_size],
            "max_cluster_size": [max_cluster_size],
        })

        stats.write_parquet(output.stats)

        print(f"\n=== Sanity Check: {wildcards.dataset} @ id={wildcards.identity} cov={wildcards.coverage} ===")
        print(f"  Total validation sequences: {total_val:,}")
        print(f"  Number of clusters: {n_clusters:,}")
        print(f"  Singletons (no matches): {n_singletons:,} ({n_singletons/total_val*100:.1f}%)")
        print(f"  Sequences with matches: {seqs_with_matches:,} ({pct_with_matches:.1f}%)")
        print(f"  Avg cluster size (non-singleton): {avg_cluster_size:.1f}")
        print(f"  Max cluster size: {max_cluster_size}")


rule aggregate_sanity_check:
    """Aggregate sanity check results across identity × coverage thresholds."""
    input:
        stats=expand(
            "results/sanity_check/{{dataset}}/val_self_stats_id{identity}_cov{coverage}.parquet",
            identity=get_identity_thresholds(),
            coverage=get_coverage_thresholds(),
        ),
    output:
        summary="results/sanity_check/{dataset}/summary.parquet",
    run:
        dfs = [pl.read_parquet(f) for f in input.stats]
        summary = pl.concat(dfs).sort(["coverage_threshold", "identity_threshold"])

        summary.write_parquet(output.summary)

        print(f"\n=== Sanity Check Summary: {wildcards.dataset} ===")
        print(summary.to_pandas().to_string(index=False))
        print("\nExpected behavior:")
        print("  - With RC: most sequences should have at least 1 match (their RC)")
        print("  - At lower thresholds: more matches from overlapping windows")


# =============================================================================
# Main Analysis: Train/Validation Leakage
# =============================================================================


rule compute_similarity_stats:
    """Compute train/val similarity statistics from cluster assignments.

    For each validation sequence, count how many training sequences share
    its cluster at the given identity × coverage threshold.
    """
    input:
        clusters="results/clustering/{dataset}/clusters_id{identity}_cov{coverage}.tsv",
        metadata="results/data/{dataset}/metadata.parquet",
    output:
        stats="results/analysis/{dataset}/similarity_stats_id{identity}_cov{coverage}.parquet",
    run:
        clusters = pl.read_csv(
            input.clusters,
            separator="\t",
            has_header=False,
            new_columns=["representative", "member"],
        )

        metadata = pl.read_parquet(input.metadata)

        clusters_with_split = clusters.join(
            metadata.select(["id", "split"]),
            left_on="member",
            right_on="id",
            how="left",
        )

        # For each cluster, count train and val members
        cluster_composition = (
            clusters_with_split
            .group_by("representative")
            .agg([
                pl.count().alias("cluster_size"),
                (pl.col("split") == "train").sum().alias("n_train"),
                (pl.col("split") == "validation").sum().alias("n_val"),
            ])
        )

        # For each val sequence, how many train sequences share its cluster?
        train_counts = (
            clusters_with_split
            .join(cluster_composition, on="representative")
            .filter(pl.col("split") == "validation")
            ["n_train"]
        )

        total_val = metadata.filter(pl.col("split") == "validation").height
        total_train = metadata.filter(pl.col("split") == "train").height

        stats = pl.DataFrame({
            "dataset": [wildcards.dataset],
            "identity_threshold": [float(wildcards.identity)],
            "coverage_threshold": [float(wildcards.coverage)],
            "total_train": [total_train],
            "total_val": [total_val],
            "n_clusters": [cluster_composition.height],
            "train_matches_min": [int(train_counts.min()) if train_counts.len() > 0 else 0],
            "train_matches_max": [int(train_counts.max()) if train_counts.len() > 0 else 0],
            "train_matches_mean": [float(train_counts.mean()) if train_counts.len() > 0 else 0.0],
            "train_matches_median": [float(train_counts.median()) if train_counts.len() > 0 else 0.0],
        })

        stats.write_parquet(output.stats)
        print(f"\n{wildcards.dataset} @ id={wildcards.identity} cov={wildcards.coverage}:")
        print(f"  Train matches per val seq: min={stats['train_matches_min'][0]}, median={stats['train_matches_median'][0]:.0f}, mean={stats['train_matches_mean'][0]:.1f}, max={stats['train_matches_max'][0]}")


rule aggregate_similarity_stats:
    """Aggregate similarity statistics across all datasets and thresholds."""
    input:
        stats=expand(
            "results/analysis/{dataset}/similarity_stats_id{identity}_cov{coverage}.parquet",
            dataset=get_all_datasets(),
            identity=get_identity_thresholds(),
            coverage=get_coverage_thresholds(),
        ),
    output:
        summary="results/analysis/similarity_summary.parquet",
    run:
        dfs = [pl.read_parquet(f) for f in input.stats]
        summary = pl.concat(dfs)
        summary = summary.sort(["dataset", "coverage_threshold", "identity_threshold"])

        summary.write_parquet(output.summary)
        print("\n=== Similarity Summary ===")
        print(summary.to_pandas().to_string(index=False))


rule plot_train_matches_heatmap:
    """Create heatmap of median and mean train matches per val sequence.

    Layout: rows = interval_type × metric, columns = genome_set (config order).
    Dataset names are split on the last '_' to extract genome_set and interval_type.
    """
    input:
        summary="results/analysis/similarity_summary.parquet",
    output:
        plot="results/plots/train_matches_heatmap.svg",
    run:
        df = pl.read_parquet(input.summary).to_pandas()
        dataset_order = [d["name"] for d in config["datasets"]]
        datasets = [d for d in dataset_order if d in df["dataset"].values]

        # Parse dataset names into genome_set and interval_type
        def parse_dataset(name):
            parts = name.rsplit("_", 1)
            return parts[0], parts[1]  # genome_set, interval_type

        # Deduplicate while preserving config order
        genome_sets = list(dict.fromkeys(parse_dataset(d)[0] for d in datasets))
        interval_types = list(dict.fromkeys(parse_dataset(d)[1] for d in datasets))

        metrics = [
            ("train_matches_median", "Median", ".0f"),
            ("train_matches_mean", "Mean", ".1f"),
        ]

        n_rows = len(interval_types) * len(metrics)
        n_cols = len(genome_sets)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            squeeze=False,
        )

        for it_idx, interval_type in enumerate(interval_types):
            for m_idx, (col_name, label, fmt) in enumerate(metrics):
                row = it_idx * len(metrics) + m_idx
                # Compute global vmin/vmax across all datasets for this metric
                vmin = df[col_name].min()
                vmax = df[col_name].max()
                for col, genome_set in enumerate(genome_sets):
                    ax = axes[row, col]
                    dataset_name = f"{genome_set}_{interval_type}"
                    if dataset_name not in df["dataset"].values:
                        ax.set_visible(False)
                        continue
                    subset = df[df["dataset"] == dataset_name]
                    pivot = subset.pivot(
                        index="identity_threshold",
                        columns="coverage_threshold",
                        values=col_name,
                    )
                    sns.heatmap(
                        pivot,
                        annot=True,
                        fmt=fmt,
                        cmap="YlOrRd",
                        vmin=vmin,
                        vmax=vmax,
                        cbar_kws={"label": f"{label} train matches"},
                        ax=ax,
                    )
                    ax.set_xlabel("Coverage Threshold")
                    ax.set_ylabel("Identity Threshold")
                    ax.set_title(f"{genome_set} {interval_type} ({label.lower()})")

        fig.suptitle("Train Matches per Validation Sequence", fontsize=14)
        plt.tight_layout()
        plt.savefig(output.plot, format="svg")
        plt.close()
        print(f"Saved heatmap to {output.plot}")
