# Functional regions to include in plots (excluding promoters since radius is arbitrary)
PLOT_REGIONS = [r for r in config["functional_regions"] if r != "promoters"]


rule all_functional_region_stats:
    input:
        expand(
            "results/stats/functional_regions/{g}.parquet",
            g=config["genome_subset_analysis"],
        ),


rule all_stats_plots:
    input:
        # Functional region stats per genome
        expand(
            "results/plots/functional_regions/{g}.svg",
            g=config["genome_subset_analysis"],
        ),
        # Size histograms per region (global)
        expand(
            "results/plots/functional_regions_size_histogram/{region}.svg",
            region=PLOT_REGIONS,
        ),
        # Global functional region summary
        "results/plots/functional_regions_overall.svg",
        # Annotation source plots
        "results/plots/annotation_sources.svg",
        "results/plots/annotation_sources_per_genome.svg",


rule all_annotation_source_stats:
    input:
        expand(
            "results/stats/annotation_sources/{g}.parquet",
            g=config["genome_subset_analysis"],
        ),
    output:
        "results/stats/annotation_sources_summary.parquet",
    run:
        dfs = []
        for g, path in zip(config["genome_subset_analysis"], input):
            df = pd.read_parquet(path)
            df["genome"] = g
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        # Pivot to get all sources as columns, fill missing with 0
        pivot = combined.pivot(
            index="genome", columns="source", values="tx_ratio"
        ).fillna(0)
        summary = pivot.agg(["mean", "median"]).T.reset_index()
        summary.columns = ["source", "mean", "median"]
        summary.to_parquet(output[0], index=False)


rule annotation_source_stats:
    input:
        "results/annotation/{g}.gtf.gz",
    output:
        "results/stats/annotation_sources/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        transcripts = ann.filter(pl.col("feature") == "transcript")
        stats_df = transcripts.group_by("source").len().to_pandas()
        stats_df.columns = ["source", "n_transcripts"]
        stats_df["tx_ratio"] = (
            stats_df["n_transcripts"] / stats_df["n_transcripts"].sum()
        )
        stats_df.to_parquet(output[0], index=False)


rule functional_region_stats:
    input:
        expand(
            "results/intervals/{region}/{{g}}.parquet",
            region=config["functional_regions"],
        ),
    output:
        "results/stats/functional_regions/{g}.parquet",
    run:
        quantiles = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
        rows = []
        for region, path in zip(config["functional_regions"], input):
            gs = GenomicSet.read_parquet(path)
            df = gs.to_polars()
            lengths = df["end"] - df["start"]
            row = {
                "region": region,
                "n_intervals": gs.n_intervals(),
                "size_total": gs.total_size(),
                "size_mean": lengths.mean(),
                "size_std": lengths.std(),
            }
            for q in quantiles:
                row[f"size_p{int(q*100)}"] = lengths.quantile(q)
            rows.append(row)
        stats_df = pd.DataFrame(rows)
        stats_df["size_total_ratio"] = (
            stats_df["size_total"] / stats_df["size_total"].sum()
        )
        stats_df["n_intervals_ratio"] = (
            stats_df["n_intervals"] / stats_df["n_intervals"].sum()
        )
        stats_df.to_parquet(output[0], index=False)


def load_genome_labels(genomes_path: str) -> dict[str, str]:
    """Load mapping from genome accession to species name."""
    df = pd.read_parquet(genomes_path)
    return dict(zip(df["Assembly Accession"], df["Organism Name"]))


rule combine_functional_region_stats:
    input:
        expand(
            "results/stats/functional_regions/{g}.parquet",
            g=config["genome_subset_analysis"],
        ),
    output:
        "results/stats/functional_regions_combined.parquet",
    run:
        dfs = []
        for g, path in zip(config["genome_subset_analysis"], input):
            df = pd.read_parquet(path)
            df["genome"] = g
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_parquet(output[0], index=False)


rule combine_annotation_source_stats:
    input:
        expand(
            "results/stats/annotation_sources/{g}.parquet",
            g=config["genome_subset_analysis"],
        ),
    output:
        "results/stats/annotation_sources_combined.parquet",
    run:
        dfs = []
        for g, path in zip(config["genome_subset_analysis"], input):
            df = pd.read_parquet(path)
            df["genome"] = g
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_parquet(output[0], index=False)


rule plot_functional_regions_per_genome:
    input:
        stats="results/stats/functional_regions/{g}.parquet",
        genomes=config["genomes_path"],
    output:
        "results/plots/functional_regions/{g}.svg",
    run:
        import matplotlib.pyplot as plt
        import numpy as np

        df = pd.read_parquet(input.stats)
        df = df[df["region"].isin(PLOT_REGIONS)]
        genome_labels = load_genome_labels(input.genomes)
        genome_name = genome_labels.get(wildcards.g, wildcards.g)

        regions = df["region"].tolist()
        x = np.arange(len(regions))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Number of intervals
        ax1 = axes[0]
        colors = plt.cm.Set2(np.linspace(0, 1, len(regions)))
        bars = ax1.bar(x, df["n_intervals"], color=colors)
        ax1.set_xticks(x)
        ax1.set_xticklabels(regions, rotation=45, ha="right")
        ax1.set_ylabel("Count")
        ax1.set_title("Number of Intervals")
        ax1.grid(axis="y", alpha=0.3)

        # Panel 2: Size distribution (box plot using quantiles)
        ax2 = axes[1]
        for i, (_, row) in enumerate(df.iterrows()):
            p10, p25, p50, p75, p90 = (
                row["size_p10"],
                row["size_p25"],
                row["size_p50"],
                row["size_p75"],
                row["size_p90"],
            )
            color = colors[i]
            ax2.fill_between([i - 0.3, i + 0.3], p25, p75, alpha=0.6, color=color)
            ax2.hlines(p50, i - 0.3, i + 0.3, colors="black", linewidth=2)
            ax2.vlines(i, p10, p25, colors="black", linewidth=1)
            ax2.vlines(i, p75, p90, colors="black", linewidth=1)
            ax2.hlines(p10, i - 0.15, i + 0.15, colors="black", linewidth=1)
            ax2.hlines(p90, i - 0.15, i + 0.15, colors="black", linewidth=1)
        ax2.set_xticks(x)
        ax2.set_xticklabels(regions, rotation=45, ha="right")
        ax2.set_ylabel("Interval Size (bp)")
        ax2.set_title("Size Distribution (p10-p90)")
        ax2.grid(axis="y", alpha=0.3)

        # Panel 3: Total size
        ax3 = axes[2]
        bars = ax3.bar(x, df["size_total"] / 1e6, color=colors)
        ax3.set_xticks(x)
        ax3.set_xticklabels(regions, rotation=45, ha="right")
        ax3.set_ylabel("Total Size (Mb)")
        ax3.set_title("Total Size")
        ax3.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Functional Region Statistics - {genome_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()


rule plot_functional_regions_overall:
    input:
        stats="results/stats/functional_regions_combined.parquet",
    output:
        "results/plots/functional_regions_overall.svg",
    run:
        import matplotlib.pyplot as plt
        import numpy as np

        df = pd.read_parquet(input.stats)
        df = df[df["region"].isin(PLOT_REGIONS)]
        regions = PLOT_REGIONS
        x = np.arange(len(regions))

        # Aggregate stats across genomes (using median interval size = size_p50)
        agg = (
            df.groupby("region")
            .agg(
                n_intervals_mean=("n_intervals", "mean"),
                n_intervals_std=("n_intervals", "std"),
                size_total_mean=("size_total", "mean"),
                size_total_std=("size_total", "std"),
                size_median_mean=("size_p50", "mean"),
                size_median_std=("size_p50", "std"),
            )
            .reindex(regions)
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = plt.cm.Set2(np.linspace(0, 1, len(regions)))

        # Panel 1: Number of intervals
        ax1 = axes[0]
        ax1.bar(
            x,
            agg["n_intervals_mean"],
            yerr=agg["n_intervals_std"],
            color=colors,
            capsize=3,
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(regions, rotation=45, ha="right")
        ax1.set_ylabel("Count")
        ax1.set_title("Number of Intervals (mean ± std)")
        ax1.grid(axis="y", alpha=0.3)

        # Panel 2: Median interval size
        ax2 = axes[1]
        ax2.bar(
            x,
            agg["size_median_mean"],
            yerr=agg["size_median_std"],
            color=colors,
            capsize=3,
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(regions, rotation=45, ha="right")
        ax2.set_ylabel("Median Interval Size (bp)")
        ax2.set_title("Median Interval Size (mean ± std)")
        ax2.grid(axis="y", alpha=0.3)

        # Panel 3: Total size
        ax3 = axes[2]
        ax3.bar(
            x,
            agg["size_total_mean"] / 1e6,
            yerr=agg["size_total_std"] / 1e6,
            color=colors,
            capsize=3,
        )
        ax3.set_xticks(x)
        ax3.set_xticklabels(regions, rotation=45, ha="right")
        ax3.set_ylabel("Total Size (Mb)")
        ax3.set_title("Total Size (mean ± std)")
        ax3.grid(axis="y", alpha=0.3)

        fig.suptitle(
            "Functional Region Statistics (Aggregated Across Genomes)", fontsize=14
        )
        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()


rule plot_functional_regions_size_histogram:
    input:
        intervals=expand(
            "results/intervals/{{region}}/{g}.parquet",
            g=config["genome_subset_analysis"],
        ),
    output:
        "results/plots/functional_regions_size_histogram/{region}.svg",
    run:
        import matplotlib.pyplot as plt
        import numpy as np

        # Collect all interval sizes across genomes
        all_sizes = []
        for path in input.intervals:
            gs = GenomicSet.read_parquet(path)
            df = gs.to_polars()
            sizes = (df["end"] - df["start"]).to_list()
            all_sizes.extend(sizes)

        all_sizes = np.array(all_sizes)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Log-scale histogram
        bins = 50
        ax.hist(all_sizes, bins=bins, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Interval Size (bp)")
        ax.set_ylabel("Count")
        ax.set_title(f"Size Distribution - {wildcards.region} (n={len(all_sizes):,})")
        ax.grid(axis="y", alpha=0.3)

        # Add summary statistics
        stats_text = (
            f"Median: {np.median(all_sizes):,.0f} bp\n"
            f"Mean: {np.mean(all_sizes):,.0f} bp\n"
            f"Total: {np.sum(all_sizes)/1e9:.2f} Gb"
        )
        ax.text(
            0.98,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()


rule plot_annotation_sources:
    input:
        "results/stats/annotation_sources_summary.parquet",
    output:
        "results/plots/annotation_sources.svg",
    run:
        import matplotlib.pyplot as plt

        df = pd.read_parquet(input[0])
        df = df.sort_values("mean", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))

        y_pos = range(len(df))
        ax.barh(y_pos, df["mean"], alpha=0.7, label="Mean")
        ax.scatter(df["median"], y_pos, color="red", zorder=5, label="Median", s=50)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(df["source"])
        ax.set_xlabel("Proportion of Transcripts")
        ax.set_title("Annotation Sources (Mean Proportion Across Genomes)")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()


rule plot_annotation_sources_per_genome:
    input:
        stats="results/stats/annotation_sources_combined.parquet",
        genomes=config["genomes_path"],
    output:
        "results/plots/annotation_sources_per_genome.svg",
    run:
        import matplotlib.pyplot as plt
        import numpy as np

        df = pd.read_parquet(input.stats)
        genome_labels = load_genome_labels(input.genomes)

        pivot = df.pivot(index="genome", columns="source", values="tx_ratio").fillna(0)

        # Sort by curated sources (BestRefSeq first, then RefSeq) in descending order
        curated_cols = [c for c in ["BestRefSeq", "RefSeq"] if c in pivot.columns]
        if curated_cols:
            pivot = pivot.sort_values(curated_cols, ascending=False)

        pivot.index = pivot.index.map(lambda g: genome_labels.get(g, g))

        fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.3)))

        bottom = np.zeros(len(pivot))
        colors = plt.cm.Set3(np.linspace(0, 1, len(pivot.columns)))

        for col, color in zip(pivot.columns, colors):
            ax.barh(pivot.index, pivot[col], left=bottom, label=col, color=color)
            bottom += pivot[col].values

        ax.set_xlabel("Proportion of Transcripts")
        ax.set_title("Annotation Sources by Species (Sorted by Curated Sources)")
        ax.legend(loc="lower right", bbox_to_anchor=(1.2, 0))
        ax.set_xlim(0, 1)

        plt.tight_layout()
        plt.savefig(output[0], format="svg", bbox_inches="tight")
        plt.close()
