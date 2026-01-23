rule all_functional_region_stats:
    input:
        expand(
            "results/stats/functional_regions/{g}.parquet",
            g=config["genome_subset_analysis"],
        ),


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
