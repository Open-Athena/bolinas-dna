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
