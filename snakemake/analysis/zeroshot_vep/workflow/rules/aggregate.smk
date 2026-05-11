"""Stage 4 (CPU): concat all per-(model, window, dataset) metric parquets into
one master table + CSV for the issue body.

Schema of the aggregated table::

    model, window, dataset, split, aggregation, subset, score,
    value, se, n_pairs, n_ties

That's 5 models × 3 windows × 3 datasets × 30 scores × (8 + 2) aggregations =
13,500 rows. Small enough to commit if we ever want to.
"""


rule aggregate_metrics:
    input:
        all_metric_paths(),
    output:
        parquet="results/metrics_aggregated.parquet",
        csv="results/metrics_aggregated.csv",
    run:
        frames = [pd.read_parquet(p) for p in input]
        agg = pd.concat(frames, ignore_index=True)

        # Column order for readability.
        cols = [
            "model", "window", "dataset", "split",
            "aggregation", "subset", "score",
            "value", "se", "n_pairs", "n_ties",
        ]
        agg = agg[cols].sort_values(["dataset", "model", "window", "aggregation", "subset", "score"])

        agg.to_parquet(output.parquet, index=False)
        agg.to_csv(output.csv, index=False)

        n_combos = agg.groupby(["model", "window", "dataset"]).ngroups
        print(
            f"[zeroshot_vep/aggregate] {len(agg)} metric rows across "
            f"{n_combos} (model, window, dataset) combos",
            flush=True,
        )
