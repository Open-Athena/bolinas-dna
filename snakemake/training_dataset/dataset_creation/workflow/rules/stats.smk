rule all_stats:
    input:
        expand(
            "results/stats/{g}.parquet",
            g=config["genome_subset_analysis"],
        ),


rule functional_region_stats:
    input:
        expand(
            "results/intervals/{region}/{{g}}.parquet",
            region=config["functional_regions"],
        ),
    output:
        "results/stats/{g}.parquet",
    run:
        rows = []
        for region, path in zip(config["functional_regions"], input):
            gs = GenomicSet.read_parquet(path)
            df = gs.to_pandas()
            lengths = df["end"] - df["start"]
            rows.append(
                {
                    "region": region,
                    "n_intervals": gs.n_intervals(),
                    "total_size": gs.total_size(),
                    "mean_length": lengths.mean() if len(lengths) > 0 else 0.0,
                    "std_length": lengths.std() if len(lengths) > 0 else 0.0,
                }
            )
        stats_df = pd.DataFrame(rows)
        stats_df.to_parquet(output[0], index=False)
