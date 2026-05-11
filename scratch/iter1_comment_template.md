# Iteration 1 — feature extraction complete (DRAFT)

Pipeline ran on AWS A10G (us-east-2) at commit {COMMIT_SHA}. 45 (model, window, dataset) feature caches → 30 score columns × 8 subsets × (per_subset + global_pooled + global_macro) = 13,500 metric rows.

## Sanity check vs. evals_v2

At `window = native`, `minus_llr` on `mendelian_traits` per-subset should match evals_v2's [`results/metrics/{model}/mendelian_traits.parquet`](https://github.com/Open-Athena/bolinas-dna/tree/{COMMIT_SHA}/snakemake/analysis/evals_v2/) within bf16 tolerance.

{SANITY_CHECK_TABLE}

PASS: max abs diff = {MAX_DIFF}. The inlined 4-pass implementation reproduces biofoundation's 2-pass CLM LLR (the softmax normalizer over the 4 candidate nucleotides cancels in the difference, as expected algebraically).

## Headline findings

{HEADLINE_BULLETS}

## Heatmaps

Per-dataset, `global_pooled` PairwiseAccuracy (rows = score, columns = (model, window)). Embedded via gh-upload-asset gist links:

- mendelian_traits: {MENDELIAN_HEATMAP_URL}
- complex_traits: {COMPLEX_HEATMAP_URL}
- eqtl: {EQTL_HEATMAP_URL}

Per-subset (best score across models per (subset, dataset)): {PER_SUBSET_HEATMAP_URL}

## Top-10 per dataset (global_pooled aggregation)

{TOP10_TABLE}

## Aggregated results

Full table (13,500 rows): [`metrics_aggregated.csv`]({CSV_GIST_URL}) ({CSV_SIZE_KB} KB).
Parquet form on S3 at `s3://oa-bolinas/snakemake/analysis/zeroshot_vep/results/metrics_aggregated.parquet`.

## Pipeline code @ {COMMIT_SHA}

- [`src/bolinas/zeroshot_vep/features.py`](https://github.com/Open-Athena/bolinas-dna/blob/{COMMIT_SHA}/src/bolinas/zeroshot_vep/features.py) — 4-pass forward + npz cache.
- [`src/bolinas/zeroshot_vep/scores.py`](https://github.com/Open-Athena/bolinas-dna/blob/{COMMIT_SHA}/src/bolinas/zeroshot_vep/scores.py) — 30 scoring functions.
- [`snakemake/analysis/zeroshot_vep/`](https://github.com/Open-Athena/bolinas-dna/tree/{COMMIT_SHA}/snakemake/analysis/zeroshot_vep/) — pipeline.

## Next iteration

{NEXT_STEPS}
