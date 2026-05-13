"""Supervised classifier OOF predictions + PairwiseAccuracy per (model, dataset, recipe, classifier).

CPU-bound: the features cache already lives on S3; these rules read the cache,
run 3-fold chrom-grouped CV with the configured classifier, and pipe the
concatenated OOF predictions through the existing ``compute_pairwise_metrics``.

Wildcard guards:

* ``{model}`` / ``{dataset}`` / ``{recipe}`` / ``{classifier}`` are restricted
  to the names in the config — Snakemake will refuse to evaluate any other
  wildcard combination, so a typo in a target string fails immediately
  rather than producing a useless empty result.
"""


rule compute_oof_predictions:
    input:
        features="results/features/{model}/{dataset}.parquet",
    output:
        predictions="results/predictions/{model}/{dataset}/{recipe}/{classifier}.parquet",
        fold_records="results/fold_records/{model}/{dataset}/{recipe}/{classifier}.json",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
        recipe="|".join(RECIPES),
        classifier="|".join(CLASSIFIERS),
    params:
        n_splits=config["cv"]["n_splits"],
        n_splits_inner=config["cv"]["n_splits_inner"],
        mode=config["cv"]["mode"],
    # `threads: 1`: most sklearn classifiers (LogReg/liblinear, LinearSVC, KNN)
    # are single-threaded internally, so claiming more threads only blocks the
    # snakemake scheduler from running other combos in parallel. With threads=1
    # on the 4-vCPU SkyPilot node, snakemake fans out 4 combos at a time and
    # the BFS sweep takes ~4× less wall-clock.
    threads: 1
    run:
        # Guard against ineligible (recipe, dataset) combos (e.g. mean_delta on
        # complex_traits). Snakemake doesn't filter these from the wildcard
        # expansion automatically, but the top-level `rule all` only requests
        # the eligible ones, so this branch is purely a defensive assert.
        assert is_recipe_dataset_compatible(wildcards.recipe, wildcards.dataset), (
            f"recipe {wildcards.recipe!r} should not be run on dataset "
            f"{wildcards.dataset!r} (symmetry-incompatible)"
        )

        features = pd.read_parquet(input.features)
        preds_df, fold_records = fit_oof_predictions(
            features_df=features,
            recipe=wildcards.recipe,
            classifier=wildcards.classifier,
            n_splits=params.n_splits,
            n_splits_inner=params.n_splits_inner,
            mode=params.mode,
        )
        preds_df.to_parquet(output.predictions, index=False)
        write_fold_records_json(output.fold_records, fold_records)
        print(
            f"[supervised_vep] OOF {wildcards.model} {wildcards.dataset} "
            f"{wildcards.recipe} {wildcards.classifier} ({params.mode}): "
            f"n_pred={len(preds_df)}, folds={len(fold_records)}"
        )


rule compute_supervised_metrics:
    input:
        predictions="results/predictions/{model}/{dataset}/{recipe}/{classifier}.parquet",
    output:
        metrics="results/metrics/{model}/{dataset}/{recipe}/{classifier}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
        recipe="|".join(RECIPES),
        classifier="|".join(CLASSIFIERS),
    threads: 1
    run:
        preds = pd.read_parquet(input.predictions)
        metrics = compute_metrics_from_oof(preds, score_column="score")

        # Stamp the wildcard context onto every row so the leaderboard
        # aggregate can be reconstructed from a flat concat.
        metrics["model"] = wildcards.model
        metrics["dataset"] = wildcards.dataset
        metrics["recipe"] = wildcards.recipe
        metrics["classifier"] = wildcards.classifier
        metrics["family"] = "supervised"
        metrics.to_parquet(output.metrics, index=False)


rule compute_baseline_metrics:
    """Zero-shot baselines (no supervision) — one row per ``baseline_scores`` × subset."""
    input:
        features="results/features/{model}/{dataset}.parquet",
    output:
        metrics="results/baseline_metrics/{model}/{dataset}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
    threads: 1
    run:
        features = pd.read_parquet(input.features)
        metrics = compute_zeroshot_baseline_metrics(
            features,
            score_columns=tuple(config.get("baseline_scores", [])),
        )
        metrics["model"] = wildcards.model
        metrics["dataset"] = wildcards.dataset
        metrics["recipe"] = "_zeroshot_"
        metrics["classifier"] = "_zeroshot_"
        metrics["family"] = "baseline"
        metrics.to_parquet(output.metrics, index=False)


rule aggregate_leaderboard:
    """Concatenate all metric parquets into one long-form leaderboard."""
    input:
        supervised=metric_targets(),
        baseline=baseline_metric_targets(),
    output:
        "results/leaderboard.parquet",
    threads: 1
    run:
        parts = []
        for p in input.supervised + input.baseline:
            parts.append(pd.read_parquet(p))
        leaderboard = pd.concat(parts, ignore_index=True)
        leaderboard.to_parquet(output[0], index=False)
        print(
            f"[supervised_vep] leaderboard: n_rows={len(leaderboard)} "
            f"families={sorted(leaderboard['family'].unique().tolist())}"
        )
