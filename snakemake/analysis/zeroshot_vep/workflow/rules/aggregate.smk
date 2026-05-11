"""Stage 3 (CPU): concat all per-(model, window, dataset) metric parquets into
one master table + CSV for the issue body. Adds Benjamini-Hochberg FDR-adjusted
q-values across the per-(dataset, aggregation) test family.

Schema of the aggregated table::

    model, window, dataset, split, aggregation, subset, score,
    value, se, n_pairs, n_ties, p_value, q_value

That's 5 models × 3 windows × 3 datasets × 30 scores × (8 + 2) aggregations =
13,500 rows. Small enough to commit if we ever want to.

Multi-hypothesis correction: ``q_value`` is the Benjamini-Hochberg adjusted
p-value, computed within each (dataset, aggregation) family — controls FDR at
α when one rejects q ≤ α. Families separate per_subset from global_pooled
from global_macro because they test conceptually different hypotheses (a single
subset-level test ≠ a pooled test ≠ a between-subset test). Within a family,
the comparisons being corrected are (model × window × score × subset) — i.e.
"of all the scoring rules I tried in this dataset, which are significant?"
NaN p-values (e.g. degenerate single-subset macro) are passed through unchanged.
"""

import numpy as np


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (Storey & Tibshirani 2003 form).

    Closed-form: sort p ascending, multiply by ``n / rank``, then enforce
    monotonicity by taking the running min from the largest p down to the
    smallest, then cap at 1.

    NaNs are preserved.
    """
    p = np.asarray(p_values, dtype=np.float64)
    mask = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if mask.sum() == 0:
        return out
    p_valid = p[mask]
    order = np.argsort(p_valid)
    n = len(p_valid)
    ranks = np.arange(1, n + 1)
    adjusted_sorted = p_valid[order] * n / ranks
    # Step-up: enforce that q is non-decreasing as p increases.
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    # Unsort.
    adjusted = np.empty_like(p_valid)
    adjusted[order] = adjusted_sorted
    out[mask] = adjusted
    return out


rule aggregate_metrics:
    input:
        all_metric_paths(),
    output:
        parquet="results/metrics_aggregated.parquet",
        csv="results/metrics_aggregated.csv",
    run:
        frames = [pd.read_parquet(p) for p in input]
        agg = pd.concat(frames, ignore_index=True)

        # Benjamini-Hochberg FDR within each (dataset, aggregation) family —
        # corrects across (model × window × score × subset) for that family.
        agg["q_value"] = np.nan
        for (_dataset, _agg), idx in agg.groupby(["dataset", "aggregation"]).groups.items():
            agg.loc[idx, "q_value"] = _bh_adjust(agg.loc[idx, "p_value"].values)

        # Column order for readability.
        cols = [
            "model", "window", "dataset", "split",
            "aggregation", "subset", "score",
            "value", "se", "n_pairs", "n_ties",
            "p_value", "q_value",
        ]
        agg = agg[cols].sort_values(["dataset", "model", "window", "aggregation", "subset", "score"])

        agg.to_parquet(output.parquet, index=False)
        agg.to_csv(output.csv, index=False)

        n_combos = agg.groupby(["model", "window", "dataset"]).ngroups
        n_sig_05 = int((agg["q_value"] < 0.05).sum())
        print(
            f"[zeroshot_vep/aggregate] {len(agg)} metric rows across "
            f"{n_combos} (model, window, dataset) combos; "
            f"{n_sig_05} q < 0.05 (BH-FDR within (dataset, aggregation))",
            flush=True,
        )
