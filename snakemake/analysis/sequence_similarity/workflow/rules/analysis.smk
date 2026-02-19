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
        cov_mode=config["mmseqs2"]["cov_mode"],
        cluster_mode=config["mmseqs2"]["cluster_mode"],
    threads: config["mmseqs2"]["threads"]
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
            identity=get_sanity_check_identity_thresholds(),
            coverage=get_sanity_check_coverage_thresholds(),
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


rule compute_leakage_stats:
    """Compute leakage statistics from cluster assignments.

    For each cluster, check if it contains both validation and training sequences.
    A validation sequence that clusters with training sequences represents
    potential data leakage.
    """
    input:
        clusters="results/clustering/{dataset}/clusters_{identity}.tsv",
        metadata="results/data/{dataset}/metadata.parquet",
    output:
        stats="results/analysis/{dataset}/leakage_stats_{identity}.parquet",
    run:
        # Load cluster assignments
        # Format: representative_id \t member_id
        clusters = pl.read_csv(
            input.clusters,
            separator="\t",
            has_header=False,
            new_columns=["representative", "member"],
        )

        # Load metadata to get split info
        metadata = pl.read_parquet(input.metadata)

        # Join to get split information for each member
        clusters_with_split = clusters.join(
            metadata.select(["id", "split"]),
            left_on="member",
            right_on="id",
            how="left",
        )

        # For each cluster (representative), check composition
        cluster_composition = (
            clusters_with_split
            .group_by("representative")
            .agg([
                pl.count().alias("cluster_size"),
                (pl.col("split") == "train").sum().alias("n_train"),
                (pl.col("split") == "validation").sum().alias("n_val"),
            ])
        )

        # Identify "leaked" validation sequences
        # A validation sequence is leaked if it's in a cluster with training sequences
        val_in_mixed_clusters = (
            clusters_with_split
            .join(cluster_composition, on="representative")
            .filter(
                (pl.col("split") == "validation") &
                (pl.col("n_train") > 0)
            )
        )

        # Compute summary statistics
        total_val = metadata.filter(pl.col("split") == "validation").height
        total_train = metadata.filter(pl.col("split") == "train").height
        leaked_val = val_in_mixed_clusters.height
        leaked_pct = (leaked_val / total_val * 100) if total_val > 0 else 0

        # Also count validation sequences in pure validation clusters (no leakage)
        val_in_pure_clusters = total_val - leaked_val

        stats = pl.DataFrame({
            "dataset": [wildcards.dataset],
            "identity_threshold": [float(wildcards.identity)],
            "total_train": [total_train],
            "total_val": [total_val],
            "leaked_val": [leaked_val],
            "leaked_pct": [leaked_pct],
            "pure_val": [val_in_pure_clusters],
            "pure_val_pct": [(val_in_pure_clusters / total_val * 100) if total_val > 0 else 0],
            "n_clusters": [cluster_composition.height],
            "n_mixed_clusters": [cluster_composition.filter(
                (pl.col("n_train") > 0) & (pl.col("n_val") > 0)
            ).height],
        })

        stats.write_parquet(output.stats)
        print(f"\n{wildcards.dataset} @ {wildcards.identity} identity:")
        print(f"  Total validation sequences: {total_val:,}")
        print(f"  Leaked (cluster with train): {leaked_val:,} ({leaked_pct:.2f}%)")
        print(f"  Pure (no train in cluster): {val_in_pure_clusters:,}")


rule aggregate_leakage_stats:
    """Aggregate leakage statistics across all datasets and thresholds."""
    input:
        stats=expand(
            "results/analysis/{dataset}/leakage_stats_{identity}.parquet",
            dataset=get_all_datasets(),
            identity=get_all_identity_thresholds(),
        ),
    output:
        summary="results/analysis/leakage_summary.parquet",
    run:
        dfs = [pl.read_parquet(f) for f in input.stats]
        summary = pl.concat(dfs)

        # Sort by dataset and threshold
        summary = summary.sort(["dataset", "identity_threshold"])

        summary.write_parquet(output.summary)
        print("\n=== Leakage Summary ===")
        print(summary.to_pandas().to_string(index=False))


rule plot_leakage_heatmap:
    """Create heatmap of leakage percentages across datasets and thresholds."""
    input:
        summary="results/analysis/leakage_summary.parquet",
    output:
        plot="results/plots/leakage_heatmap.png",
    run:
        df = pl.read_parquet(input.summary).to_pandas()

        # Pivot for heatmap
        pivot = df.pivot(
            index="identity_threshold",
            columns="dataset",
            values="leaked_pct",
        )

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "Validation sequences with train similarity (%)"},
            ax=ax,
        )
        ax.set_xlabel("Dataset (Taxonomic Scope)")
        ax.set_ylabel("Sequence Identity Threshold")
        ax.set_title("Train/Validation Sequence Similarity (Potential Leakage)")

        plt.tight_layout()
        plt.savefig(output.plot, dpi=150)
        plt.close()
        print(f"Saved heatmap to {output.plot}")


rule plot_leakage_by_threshold:
    """Create line plot of leakage vs threshold for each dataset."""
    input:
        summary="results/analysis/leakage_summary.parquet",
    output:
        plot="results/plots/leakage_by_threshold.png",
    run:
        df = pl.read_parquet(input.summary).to_pandas()

        fig, ax = plt.subplots(figsize=(10, 6))

        for dataset in df["dataset"].unique():
            subset = df[df["dataset"] == dataset]
            ax.plot(
                subset["identity_threshold"],
                subset["leaked_pct"],
                marker="o",
                label=dataset,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Sequence Identity Threshold")
        ax.set_ylabel("Validation Sequences with Similar Train Seq (%)")
        ax.set_title("Cross-Genome Sequence Similarity at Different Thresholds")
        ax.legend(title="Training Dataset")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.45, 0.95)
        ax.set_ylim(0, None)

        plt.tight_layout()
        plt.savefig(output.plot, dpi=150)
        plt.close()
        print(f"Saved line plot to {output.plot}")
