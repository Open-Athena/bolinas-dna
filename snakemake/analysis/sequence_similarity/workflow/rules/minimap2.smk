"""Rules for minimap2-based sequence alignment analysis.

This implements the Chao et al. methodology for detecting homologous sequences
between training and validation sets using pairwise alignment.

Reference:
    Chao, Kuan-Hao et al. (2025). Predicting dynamic expression patterns in
    budding yeast with a fungal DNA language model. bioRxiv.
    https://www.biorxiv.org/content/10.1101/2025.09.19.677475v1

    "We aligned every training sequence against both the validation and test
    sets using Minimap2 (v2.28-r1209; assembly-to-assembly mode, '-x asm')
    to produce PAF files that report, for each query, the coordinates and
    match statistics of all detected alignments."

    Metrics:
    - Coverage = matching bases (PAF field 10) / query length (PAF field 2)
    - Identity = matching bases (PAF field 10) / alignment block length (PAF field 11)

    "We removed any training sequence for which both coverage >= 5% and
    identity >= 30% in any alignment"
"""


rule align_train_to_val:
    """Align training sequences against validation sequences using minimap2.

    This produces a PAF file with alignment statistics for each train-val pair.
    """
    input:
        query="results/data/{dataset}/train.fasta",
        target="results/data/{dataset}/validation.fasta",
    output:
        "results/minimap2/{dataset}/train_vs_val.paf",
    params:
        extra=f"-x {config['minimap2']['preset']}",
        sorting="none",
    threads: config["minimap2"]["threads"]
    log:
        "results/logs/minimap2/{dataset}/align_train_to_val.log",
    wrapper:
        "v9.0.1/bio/minimap2/aligner"


rule align_val_to_val:
    """Align validation sequences against themselves (sanity check).

    For the sanity check: sequences should align to themselves and their
    reverse complements (if RC augmentation is present).
    """
    input:
        query="results/data/{dataset}/validation.fasta",
        target="results/data/{dataset}/validation.fasta",
    output:
        "results/minimap2/{dataset}/val_vs_val.paf",
    params:
        extra=f"-x {config['minimap2']['preset']}",
        sorting="none",
    threads: config["minimap2"]["threads"]
    log:
        "results/logs/minimap2/{dataset}/align_val_to_val.log",
    wrapper:
        "v9.0.1/bio/minimap2/aligner"


rule parse_paf_alignments:
    """Parse PAF file and compute coverage/identity metrics.

    PAF format fields:
    0: query name
    1: query length
    2: query start
    3: query end
    4: strand (+/-)
    5: target name
    6: target length
    7: target start
    8: target end
    9: number of matching bases
    10: alignment block length
    11: mapping quality

    Computed metrics (following Chao et al.):
    - coverage = matching_bases / query_length
    - identity = matching_bases / alignment_block_length
    """
    input:
        paf="results/minimap2/{dataset}/{alignment_type}.paf",
    output:
        parquet="results/minimap2/{dataset}/{alignment_type}_parsed.parquet",
    run:
        import polars as pl

        # PAF column names
        columns = [
            "query_name", "query_length", "query_start", "query_end",
            "strand", "target_name", "target_length", "target_start",
            "target_end", "matching_bases", "alignment_length", "mapq"
        ]

        # Read PAF file (tab-separated, first 12 columns)
        df = pl.read_csv(
            input.paf,
            separator="\t",
            has_header=False,
            new_columns=columns,
            n_threads=4,
        )

        # Compute Chao et al. metrics
        df = df.with_columns([
            # Coverage = matching bases / query length
            (pl.col("matching_bases") / pl.col("query_length")).alias("coverage"),
            # Identity = matching bases / alignment block length
            (pl.col("matching_bases") / pl.col("alignment_length")).alias("identity"),
        ])

        df.write_parquet(output.parquet)

        print(f"\nParsed {len(df):,} alignments from {input.paf}")
        print(f"  Coverage range: {df['coverage'].min():.3f} - {df['coverage'].max():.3f}")
        print(f"  Identity range: {df['identity'].min():.3f} - {df['identity'].max():.3f}")


rule analyze_minimap2_leakage:
    """Analyze train-val alignments to identify potential leakage.

    Following Chao et al.: a training sequence is considered "leaked" if it has
    ANY alignment to a validation sequence with:
    - coverage >= threshold (default 5%)
    - identity >= threshold (default 30%)
    """
    input:
        alignments="results/minimap2/{dataset}/train_vs_val_parsed.parquet",
        metadata="results/data/{dataset}/metadata.parquet",
    output:
        stats="results/minimap2/{dataset}/leakage_stats.parquet",
        leaked_ids="results/minimap2/{dataset}/leaked_train_ids.txt",
    params:
        coverage_threshold=config["minimap2"]["coverage_threshold"],
        identity_threshold=config["minimap2"]["identity_threshold"],
    run:
        import polars as pl

        # Load alignments
        alignments = pl.read_parquet(input.alignments)

        # Load metadata to get total counts
        metadata = pl.read_parquet(input.metadata)
        total_train = metadata.filter(pl.col("split") == "train").height
        total_val = metadata.filter(pl.col("split") == "validation").height

        # Filter alignments by Chao et al. thresholds
        # A sequence is leaked if BOTH coverage AND identity exceed thresholds
        leaked_alignments = alignments.filter(
            (pl.col("coverage") >= params.coverage_threshold) &
            (pl.col("identity") >= params.identity_threshold)
        )

        # Get unique leaked training sequence IDs
        leaked_train_ids = leaked_alignments.select("query_name").unique()
        n_leaked = leaked_train_ids.height
        leaked_pct = (n_leaked / total_train * 100) if total_train > 0 else 0

        # Also count how many validation sequences have matches
        matched_val_ids = leaked_alignments.select("target_name").unique()
        n_val_matched = matched_val_ids.height
        val_matched_pct = (n_val_matched / total_val * 100) if total_val > 0 else 0

        # Create stats DataFrame
        stats = pl.DataFrame({
            "dataset": [wildcards.dataset],
            "coverage_threshold": [params.coverage_threshold],
            "identity_threshold": [params.identity_threshold],
            "total_train": [total_train],
            "total_val": [total_val],
            "total_alignments": [len(alignments)],
            "leaked_alignments": [len(leaked_alignments)],
            "leaked_train_seqs": [n_leaked],
            "leaked_train_pct": [leaked_pct],
            "matched_val_seqs": [n_val_matched],
            "matched_val_pct": [val_matched_pct],
        })

        stats.write_parquet(output.stats)

        # Save leaked IDs for potential filtering
        with open(output.leaked_ids, "w") as f:
            for row in leaked_train_ids.iter_rows():
                f.write(f"{row[0]}\n")

        print(f"\n=== Minimap2 Leakage Analysis: {wildcards.dataset} ===")
        print(f"  Thresholds: coverage >= {params.coverage_threshold}, identity >= {params.identity_threshold}")
        print(f"  Total train sequences: {total_train:,}")
        print(f"  Total val sequences: {total_val:,}")
        print(f"  Total alignments: {len(alignments):,}")
        print(f"  Leaked train sequences: {n_leaked:,} ({leaked_pct:.2f}%)")
        print(f"  Matched val sequences: {n_val_matched:,} ({val_matched_pct:.2f}%)")


rule analyze_minimap2_thresholds:
    """Analyze leakage across multiple coverage/identity threshold combinations.

    This generates a 2D heatmap of leaked sequences at different thresholds,
    helping to understand the sensitivity of the filtering.
    """
    input:
        alignments="results/minimap2/{dataset}/train_vs_val_parsed.parquet",
        metadata="results/data/{dataset}/metadata.parquet",
    output:
        stats="results/minimap2/{dataset}/threshold_analysis.parquet",
    params:
        coverage_thresholds=config["minimap2"]["coverage_thresholds"],
        identity_thresholds=config["minimap2"]["identity_thresholds"],
    run:
        import polars as pl

        # Load alignments
        alignments = pl.read_parquet(input.alignments)

        # Load metadata to get total counts
        metadata = pl.read_parquet(input.metadata)
        total_train = metadata.filter(pl.col("split") == "train").height

        # Analyze each threshold combination
        results = []
        for cov_thresh in params.coverage_thresholds:
            for id_thresh in params.identity_thresholds:
                leaked = alignments.filter(
                    (pl.col("coverage") >= cov_thresh) &
                    (pl.col("identity") >= id_thresh)
                ).select("query_name").unique().height

                results.append({
                    "dataset": wildcards.dataset,
                    "coverage_threshold": cov_thresh,
                    "identity_threshold": id_thresh,
                    "leaked_train_seqs": leaked,
                    "leaked_train_pct": (leaked / total_train * 100) if total_train > 0 else 0,
                })

        stats = pl.DataFrame(results)
        stats.write_parquet(output.stats)

        print(f"\n=== Threshold Analysis: {wildcards.dataset} ===")
        print(stats.to_pandas().pivot(
            index="coverage_threshold",
            columns="identity_threshold",
            values="leaked_train_pct"
        ).to_string())


rule plot_minimap2_scatter:
    """Create scatter plot of coverage vs identity for all alignments."""
    input:
        alignments="results/minimap2/{dataset}/train_vs_val_parsed.parquet",
    output:
        plot="results/plots/{dataset}_minimap2_scatter.png",
    params:
        coverage_threshold=config["minimap2"]["coverage_threshold"],
        identity_threshold=config["minimap2"]["identity_threshold"],
    run:
        import matplotlib.pyplot as plt
        import polars as pl

        alignments = pl.read_parquet(input.alignments).to_pandas()

        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot (subsample if too many points)
        n_points = len(alignments)
        if n_points > 50000:
            sample = alignments.sample(n=50000, random_state=42)
            ax.set_title(f"Coverage vs Identity ({wildcards.dataset}, {n_points:,} alignments, 50k shown)")
        else:
            sample = alignments
            ax.set_title(f"Coverage vs Identity ({wildcards.dataset}, {n_points:,} alignments)")

        ax.scatter(
            sample["coverage"],
            sample["identity"],
            alpha=0.3,
            s=1,
            c="blue",
        )

        # Add threshold lines (Chao et al. defaults)
        ax.axvline(x=params.coverage_threshold, color="red", linestyle="--",
                   label=f"Coverage = {params.coverage_threshold}")
        ax.axhline(y=params.identity_threshold, color="red", linestyle="--",
                   label=f"Identity = {params.identity_threshold}")

        # Highlight the "leaked" region
        ax.fill_between(
            [params.coverage_threshold, 1.0],
            params.identity_threshold, 1.0,
            alpha=0.1, color="red",
            label="Leaked region"
        )

        ax.set_xlabel("Coverage (matching bases / query length)")
        ax.set_ylabel("Identity (matching bases / alignment length)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output.plot, dpi=150)
        plt.close()

        print(f"Saved scatter plot to {output.plot}")


rule plot_minimap2_heatmap:
    """Create heatmap of leakage % across threshold combinations."""
    input:
        stats="results/minimap2/{dataset}/threshold_analysis.parquet",
    output:
        plot="results/plots/{dataset}_minimap2_heatmap.png",
    run:
        import matplotlib.pyplot as plt
        import polars as pl
        import seaborn as sns

        df = pl.read_parquet(input.stats).to_pandas()

        # Pivot for heatmap
        pivot = df.pivot(
            index="coverage_threshold",
            columns="identity_threshold",
            values="leaked_train_pct",
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "Leaked train sequences (%)"},
            ax=ax,
        )
        ax.set_xlabel("Identity Threshold")
        ax.set_ylabel("Coverage Threshold")
        ax.set_title(f"Train/Val Leakage by Threshold ({wildcards.dataset})")

        plt.tight_layout()
        plt.savefig(output.plot, dpi=150)
        plt.close()

        print(f"Saved heatmap to {output.plot}")


rule aggregate_minimap2_stats:
    """Aggregate minimap2 leakage stats across all datasets."""
    input:
        stats=expand(
            "results/minimap2/{dataset}/leakage_stats.parquet",
            dataset=[d["name"] for d in config["datasets"]],
        ),
    output:
        summary="results/minimap2/leakage_summary.parquet",
    run:
        import polars as pl

        dfs = [pl.read_parquet(f) for f in input.stats]
        summary = pl.concat(dfs)

        summary.write_parquet(output.summary)

        print("\n=== Minimap2 Leakage Summary (Chao et al. methodology) ===")
        print(summary.to_pandas().to_string(index=False))
