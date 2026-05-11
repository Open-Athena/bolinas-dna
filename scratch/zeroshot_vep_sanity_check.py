"""Compare zeroshot_vep output to evals_v2 reference at window=native.

Validates: my 4-pass LLR collapses to biofoundation's 2-pass CLM LLR when you
take the difference (the softmax normalizer over the 4 candidates cancels).
PairwiseAccuracy values should match within bf16 tolerance.
"""

from __future__ import annotations

import sys

import pandas as pd


# (dataset, score_column) pairs as used by evals_v2.
DATASET_SCORE_MAP = {
    "mendelian_traits": "minus_llr",
    "complex_traits": "abs_llr",
    "eqtl": "abs_llr",
}

# Native window per model (from snakemake/analysis/zeroshot_vep/config/config.yaml).
NATIVE_WINDOW = {
    "exp55-mammals": 256,
    "exp58-mammals": 256,
    "exp58-animals": 256,
    "exp59-mammals": 256,
    "exp136-proj_v30": 255,
}


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <zeroshot_metrics.parquet> <evals_v2_ref.parquet>", file=sys.stderr)
        return 2

    new = pd.read_parquet(sys.argv[1])
    ref = pd.read_parquet(sys.argv[2])

    rows = []
    for (model, dataset), score in (
        (k, DATASET_SCORE_MAP[k[1]]) for k in [(m, d) for m in NATIVE_WINDOW for d in DATASET_SCORE_MAP]
    ):
        w = NATIVE_WINDOW[model]
        new_sub = new[
            (new["model"] == model)
            & (new["dataset"] == dataset)
            & (new["window"] == w)
            & (new["aggregation"] == "per_subset")
            & (new["score"] == score)
        ][["subset", "value", "n_pairs"]].rename(columns={"value": "value_new", "n_pairs": "n_new"})

        ref_sub = ref[
            (ref["model"] == model) & (ref["dataset"] == dataset) & (ref["score_type"] == score)
        ][["subset", "value", "n_pairs"]].rename(columns={"value": "value_ref", "n_pairs": "n_ref"})

        m = new_sub.merge(ref_sub, on="subset", how="outer", indicator=True)
        m["model"] = model
        m["dataset"] = dataset
        m["score"] = score
        m["window"] = w
        m["abs_diff"] = (m["value_new"] - m["value_ref"]).abs()
        rows.append(m)

    cmp = pd.concat(rows, ignore_index=True)

    # Highlight only mismatches.
    bad = cmp[(cmp["abs_diff"] > 0.005) | (cmp["_merge"] != "both")]
    if len(bad):
        print("[sanity] DIFFS or missing rows:")
        print(bad.to_string(index=False))
    else:
        print("[sanity] all matching rows agree within 0.005")

    print()
    print(f"[sanity] max abs_diff = {cmp['abs_diff'].max():.6f}")
    print(f"[sanity] n_compared = {len(cmp)}; n_both = {(cmp['_merge'] == 'both').sum()}")
    print(f"[sanity] pair-count match (n_new == n_ref): {(cmp['n_new'] == cmp['n_ref']).sum()}/{len(cmp)}")
    return 0 if cmp["abs_diff"].max() < 0.005 else 1


if __name__ == "__main__":
    sys.exit(main())
