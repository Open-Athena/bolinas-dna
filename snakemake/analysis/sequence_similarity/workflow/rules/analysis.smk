"""Rules for analyzing clustering results and computing leakage statistics."""


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
        import polars as pl

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
        import polars as pl

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
        import matplotlib.pyplot as plt
        import polars as pl
        import seaborn as sns

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
        import matplotlib.pyplot as plt
        import polars as pl

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
