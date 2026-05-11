"""Stage 3 (CPU): PairwiseAccuracy per (subset, score) + global pooled + macro.

Three aggregation flavors per (model, window, dataset, score):

- ``per_subset``: 8 rows, one per consequence-group subset (the breakdown).
- ``global_pooled``: single row, PairwiseAccuracy on all matched pairs flat
  (weights subsets by their pair counts).
- ``global_macro``: single row, unweighted mean of per-subset accuracies. SE
  computed from per-subset accuracies as a between-subset standard error.
"""

import math


def _macro_aggregate(per_subset: pd.DataFrame) -> dict:
    """Mean of per-subset PairwiseAccuracies, with between-subset SEM.

    SEM = stdev(values) / sqrt(n_subsets). This is the SE of the mean across
    subsets (a different statistical object than per-subset binomial SE — it
    captures variability of the *score's behavior across subsets*, not within).
    """
    vals = per_subset["value"].values
    n = len(vals)
    mean_val = float(vals.mean())
    sem = float(vals.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {
        "value": mean_val,
        "se": sem,
        "n_pairs": int(per_subset["n_pairs"].sum()),
        "n_ties": int(per_subset["n_ties"].sum()),
    }


rule compute_metrics:
    input:
        "results/scores/{model}__win{window}__{dataset}.parquet",
    output:
        "results/metrics/{model}__win{window}__{dataset}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
        window=r"[0-9]+",
    run:
        df = pd.read_parquet(input[0])
        score_cols = list(config["score_columns"])
        for c in score_cols:
            assert c in df.columns, f"missing score column {c!r}"

        records: list[dict] = []
        for score_col in score_cols:
            # Per-subset.
            per_subset_rows = []
            for subset_name, sub in df.groupby("subset", sort=False):
                res = pairwise_accuracy(
                    label=sub["label"],
                    score=sub[score_col],
                    match_group=sub["match_group"],
                )
                row = dict(
                    aggregation="per_subset",
                    subset=str(subset_name),
                    score=score_col,
                    **res,
                )
                records.append(row)
                per_subset_rows.append(row)

            # Global pooled — every matched_group across the whole dataset.
            # match_group ids are globally unique across subsets per evals_v2's
            # invariant ("no match_group spans subsets"), so passing the full
            # frame to pairwise_accuracy is safe.
            pooled = pairwise_accuracy(
                label=df["label"],
                score=df[score_col],
                match_group=df["match_group"],
            )
            records.append(dict(
                aggregation="global_pooled",
                subset="__global_pooled__",
                score=score_col,
                **pooled,
            ))

            # Global macro — unweighted mean of per-subset accuracies.
            per_subset_df = pd.DataFrame(per_subset_rows)
            macro = _macro_aggregate(per_subset_df)
            records.append(dict(
                aggregation="global_macro",
                subset="__global_macro__",
                score=score_col,
                **macro,
            ))

        out = pd.DataFrame(records)
        out["model"] = wildcards.model
        out["window"] = int(wildcards.window)
        out["dataset"] = wildcards.dataset
        out["split"] = config["split"]

        # Defensive: PairwiseAccuracy must be in [0, 1].
        assert (out["value"] >= 0).all() and (out["value"] <= 1).all(), (
            f"out-of-range PairwiseAccuracy: "
            f"{out[(out['value'] < 0) | (out['value'] > 1)]}"
        )
        out.to_parquet(output[0], index=False)
        print(
            f"[zeroshot_vep/metrics] {wildcards.model} win={wildcards.window} "
            f"{wildcards.dataset}: {len(out)} rows "
            f"({len(score_cols)} scores × (8 subsets + 2 global) = {len(score_cols)*10})",
            flush=True,
        )
