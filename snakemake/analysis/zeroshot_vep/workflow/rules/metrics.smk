"""Stage 2 (CPU): PairwiseAccuracy per (subset, score) + global pooled + macro.

Sign convention: every score is converted to "higher = more pathogenic" via
the ``SCORE_DIRECTIONS`` mapping in :mod:`bolinas.zeroshot_vep.scores` BEFORE
calling ``pairwise_accuracy``. Three sign flips happen here (vs. the raw
features parquet):
- ``entropy`` → ``minus_entropy`` (negate; high entropy = permissive = NOT pathogenic).
- ``minus_logp_ref`` → ``logp_ref`` (negate; high log p[ref] = conserved = pathogenic-prone).
- ``embed_dot_*`` → ``embed_minus_dot_*`` (rename only; the raw value is already ``-⟨ref, alt⟩``).

With the sign locked, p-values are **one-sided** (``H1: acc > 0.5``) which is
2x more powerful than two-sided at the same FDR. A score with the wrong sign
correctly tests as non-significant (one-sided p ~1.0).

Three aggregation flavors per (model, window, dataset, score):
- ``per_subset``: 8 rows, one per consequence-group subset.
- ``global_pooled``: single row, PairwiseAccuracy on all matched pairs flat.
- ``global_macro``: single row, unweighted mean of per-subset accuracies. SE is
  the between-subset SEM. p-value is a one-sided one-sample t-test against
  ``H1: mean > 0.5`` (df = n_subsets - 1).
"""

import math
from scipy.stats import t as student_t

from bolinas.zeroshot_vep.scores import SCORE_DIRECTIONS, apply_score_directions


def _macro_aggregate(per_subset: pd.DataFrame) -> dict:
    """Mean of per-subset PairwiseAccuracies, with between-subset SEM + p-value.

    SEM = stdev(values) / sqrt(n_subsets). ``p_value`` is the closed-form
    **one-sided** one-sample t-test against ``H1: mean > 0.5`` (df = n - 1).
    """
    vals = per_subset["value"].values
    n = len(vals)
    mean_val = float(vals.mean())
    if n > 1:
        std = float(vals.std(ddof=1))
        sem = std / math.sqrt(n)
        if sem > 0:
            t_stat = (mean_val - 0.5) / sem
            # One-sided p-value, df = n - 1, alternative: mean > 0.5
            p_value = float(student_t.sf(t_stat, df=n - 1))
        else:
            # All per-subset values identical → degenerate t-test
            p_value = 0.0 if mean_val > 0.5 else (1.0 if mean_val <= 0.5 else float("nan"))
    else:
        sem = 0.0
        p_value = float("nan")
    return {
        "value": mean_val,
        "se": float(sem),
        "n_pairs": int(per_subset["n_pairs"].sum()),
        "n_ties": int(per_subset["n_ties"].sum()),
        "p_value": p_value,
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

        # Apply locked sign convention: build a DataFrame with final names
        # (minus_entropy / logp_ref / embed_minus_dot_*) and "higher = pathogenic"
        # values. Pass this to pairwise_accuracy with alternative='greater' for
        # a one-sided test.
        signed = apply_score_directions(df)
        score_cols = list(SCORE_DIRECTIONS.keys())

        records: list[dict] = []
        for score_col in score_cols:
            # Per-subset.
            per_subset_rows = []
            for subset_name, sub_idx in df.groupby("subset", sort=False).groups.items():
                sub_label = df.loc[sub_idx, "label"]
                sub_mg = df.loc[sub_idx, "match_group"]
                sub_score = signed.loc[sub_idx, score_col]
                res = pairwise_accuracy(
                    label=sub_label,
                    score=sub_score,
                    match_group=sub_mg,
                    alternative="greater",
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
            pooled = pairwise_accuracy(
                label=df["label"],
                score=signed[score_col],
                match_group=df["match_group"],
                alternative="greater",
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
