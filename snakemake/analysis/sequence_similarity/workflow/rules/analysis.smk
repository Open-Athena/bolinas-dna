"""Rules for analyzing clustering results and computing leakage statistics."""


GENOME_SET_SPECIES = {
    "humans": 1,
    "primates": 11,
    "mammals": 81,
}


def _plot_train_matches(summary_path, output_path, col_name, label, fmt):
    """Plot a single heatmap: rows = coverage × region, cols = genome sets."""
    import numpy as np

    df = pl.read_parquet(summary_path).to_pandas()
    dataset_order = [d["name"] for d in config["datasets"]]
    datasets = [d for d in dataset_order if d in df["dataset"].values]

    def parse_dataset(name):
        return name.rsplit("_", 1)  # (genome_set, interval_type)

    genome_sets = list(dict.fromkeys(parse_dataset(d)[0] for d in datasets))
    interval_types = list(dict.fromkeys(parse_dataset(d)[1] for d in datasets))
    identity_thresholds = sorted(df["identity_threshold"].unique())
    coverage_thresholds = sorted(df["coverage_threshold"].unique())

    # Build matrix: rows = (region, coverage), cols = (genome_set, identity)
    n_cov = len(coverage_thresholds)
    n_id = len(identity_thresholds)
    n_rows = len(interval_types) * n_cov
    n_cols = len(genome_sets) * n_id
    matrix = np.full((n_rows, n_cols), np.nan)

    row_labels = []
    for interval_type in interval_types:
        for cov in coverage_thresholds:
            row_labels.append(str(cov))

    for i, interval_type in enumerate(interval_types):
        for j, cov in enumerate(coverage_thresholds):
            row = i * n_cov + j
            for k, genome_set in enumerate(genome_sets):
                for m, ident in enumerate(identity_thresholds):
                    col = k * n_id + m
                    dataset_name = f"{genome_set}_{interval_type}"
                    mask = (
                        (df["dataset"] == dataset_name)
                        & (df["identity_threshold"] == ident)
                        & (df["coverage_threshold"] == cov)
                    )
                    vals = df.loc[mask, col_name]
                    if len(vals) > 0:
                        matrix[row, col] = vals.iloc[0]

    # Sub-column labels: identity thresholds repeated per genome set
    col_labels = [str(ident) for _ in genome_sets for ident in identity_thresholds]

    fig, ax = plt.subplots(figsize=(1.4 * n_cols + 1.5, 0.6 * n_rows + 2))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="YlOrRd",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={
            "label": f"{label} train matches",
            "orientation": "horizontal",
            "shrink": 0.5,
            "pad": 0.15,
        },
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )

    # Identity ticks at the bottom, genome set headers at the top
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position("bottom")
    ax.set_xlabel("Identity threshold")
    ax.set_ylabel("Coverage threshold")

    # Add genome set group labels at the top
    for k, gs in enumerate(genome_sets):
        n_sp = GENOME_SET_SPECIES.get(gs, "?")
        sp_label = "1 species" if n_sp == 1 else f"{n_sp} species"
        center_x = k * n_id + n_id / 2
        ax.text(
            center_x, -0.4, f"{gs}\n({sp_label})",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            transform=ax.transData,
        )

    # Draw vertical separators between genome set groups
    for k in range(1, len(genome_sets)):
        x = k * n_id
        ax.axvline(x, color="black", linewidth=2)

    # Add region labels on the right via a secondary y-axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([
        i * n_cov + n_cov / 2
        for i in range(len(interval_types))
    ])
    ax2.set_yticklabels(interval_types, fontsize=12, fontweight="bold")
    ax2.tick_params(right=False, pad=15)

    # Draw horizontal separator between region groups
    for i in range(1, len(interval_types)):
        y = i * n_cov
        ax.axhline(y, color="black", linewidth=2)

    fig.suptitle(f"{label} Train Matches per Validation Sequence", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {output_path}")


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


rule plot_train_matches_median:
    """Heatmap of median train matches per val sequence.

    Rows = regions (promoters, cds), columns = genome sets (humans, primates, …).
    Dataset names are split on the last '_' to extract genome_set and region.
    """
    input:
        summary="results/analysis/similarity_summary.parquet",
    output:
        plot="results/plots/train_matches_median.svg",
    run:
        _plot_train_matches(input.summary, output.plot, "train_matches_median", "Median", ".0f")


rule plot_train_matches_mean:
    """Heatmap of mean train matches per val sequence.

    Rows = regions (promoters, cds), columns = genome sets (humans, primates, …).
    """
    input:
        summary="results/analysis/similarity_summary.parquet",
    output:
        plot="results/plots/train_matches_mean.svg",
    run:
        _plot_train_matches(input.summary, output.plot, "train_matches_mean", "Mean", ".1f")
