# Iteration 1 — feature extraction complete (DRAFT)

Pipeline ran on AWS A10G (us-east-2) at commit {COMMIT_SHA}. 45 (model, window, dataset) GPU jobs → 13,500 metric rows = 5 models × 3 windows × 3 datasets × 30 scores × (8 per-subset + 2 global = 10 aggregations).

**Statistics**:
- `value` ∈ [0, 1] — PairwiseAccuracy (ties = 0.5).
- `se` — Wald binomial SE for per_subset / global_pooled; between-subset SEM for global_macro.
- `p_value` — closed-form two-sided sign-test `Binom(n_pairs - n_ties, 0.5)` for per_subset / global_pooled; one-sample two-sided t-test against `mean = 0.5` for global_macro.
- `q_value` — Benjamini-Hochberg FDR-adjusted p, computed **within each (dataset, aggregation) family**, correcting across (model × window × score × subset). Reject `q ≤ α` to control FDR at `α`.

## Sanity check vs. evals_v2

At `window = native`, `minus_llr` on `mendelian_traits` and `abs_llr` on `complex_traits` / `eqtl` per-subset should match `evals_v2`'s [`results/metrics/{model}/{dataset}.parquet`](https://github.com/Open-Athena/bolinas-dna/tree/{COMMIT_SHA}/snakemake/analysis/evals_v2/) within bf16 tolerance.

<details><summary>Sanity check table (PASS — max abs diff {MAX_DIFF})</summary>

{SANITY_CHECK_TABLE}

The 4-pass LLR equals biofoundation's 2-pass CLM LLR algebraically (the softmax normalizer over the 4 candidate nucleotides cancels in the difference). Bf16-level agreement confirms the inlined forward pass is numerically equivalent.

</details>

## Headline findings

{HEADLINE_BULLETS}

## Heatmaps

### `global_pooled` PairwiseAccuracy (rows = score, columns = `model | w{window}`)

- mendelian_traits: {MENDELIAN_HEATMAP_URL}
- complex_traits: {COMPLEX_HEATMAP_URL}
- eqtl: {EQTL_HEATMAP_URL}

<details><summary>Per-subset detail — best score across (model, window) per (subset, dataset)</summary>

- mendelian_traits per-subset: {MENDELIAN_PER_SUBSET_URL}
- complex_traits per-subset: {COMPLEX_PER_SUBSET_URL}
- eqtl per-subset: {EQTL_PER_SUBSET_URL}

</details>

## Top-10 per dataset (global_pooled, ranked by value; only `q < 0.05` shown)

<details><summary>mendelian_traits</summary>

{MENDELIAN_TOP10}

</details>

<details><summary>complex_traits</summary>

{COMPLEX_TOP10}

</details>

<details><summary>eqtl</summary>

{EQTL_TOP10}

</details>

## Significance footprint

<details><summary>Total (model × window × score × subset) hits, per (dataset, aggregation), q &lt; 0.05</summary>

{SIGNIFICANCE_TABLE}

Interpret as "at FDR 0.05, what fraction of the tested score combos beats random within this dataset family?" — high counts mean the dataset distinguishes positives from negatives easily under our scoring rules; low counts mean the dataset is harder OR the rules are mis-tuned.

</details>

## Aggregated results

Full table (13,500 rows): [`metrics_aggregated.csv`]({CSV_GIST_URL}) ({CSV_SIZE_KB} KB).
Parquet form on S3 at `s3://oa-bolinas/snakemake/analysis/zeroshot_vep/results/metrics_aggregated.parquet`.

## Pipeline code @ {COMMIT_SHA}

- [`src/bolinas/zeroshot_vep/features.py`](https://github.com/Open-Athena/bolinas-dna/blob/{COMMIT_SHA}/src/bolinas/zeroshot_vep/features.py) — 4-pass forward + on-the-fly scoring.
- [`src/bolinas/zeroshot_vep/scores.py`](https://github.com/Open-Athena/bolinas-dna/blob/{COMMIT_SHA}/src/bolinas/zeroshot_vep/scores.py) — 30 scoring functions.
- [`src/bolinas/evals/metrics.py`](https://github.com/Open-Athena/bolinas-dna/blob/{COMMIT_SHA}/src/bolinas/evals/metrics.py) — pairwise_accuracy (now returns `p_value`).
- [`snakemake/analysis/zeroshot_vep/workflow/rules/aggregate.smk`](https://github.com/Open-Athena/bolinas-dna/blob/{COMMIT_SHA}/snakemake/analysis/zeroshot_vep/workflow/rules/aggregate.smk) — BH FDR.

## Next iteration

{NEXT_STEPS}
