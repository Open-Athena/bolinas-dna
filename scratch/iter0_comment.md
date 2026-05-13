🤖 **iter-0: feature cache complete for `exp166-p1B` × 3 datasets.**

Cache schema (per-variant parquet, one row aligned with the HF train split):

| dataset | n | D | dense list cols | scalar cols |
|---|---:|---:|---|---|
| `mendelian_traits` | 9,820 | 1,920 | `mean_ref`, `mean_alt`, `traitgym_innerprod` | `llr`, `minus_llr`, `abs_llr`, `embed_last_l2` |
| `complex_traits` | 1,128 | 1,920 | same | same |
| `eqtl` | 4,612 | 1,920 | same | same |

Total cache size on S3 ≈ 350 MB. Hidden dim D=1,920 (`exp166-p1B`'s width). Files at `s3://oa-bolinas/snakemake/analysis/supervised_vep/results/features/exp166-p1B/{dataset}.parquet`.

### Zero-shot sanity (macro avg + global PairwiseAccuracy from the cache alone)

These are the **no-supervised-head baselines** — what iter-1+ has to beat. Computed by piping the cached scalars through `compute_pairwise_metrics`, same code path as #161 / #162 / #172.

| dataset | score | macro avg | global | n_pairs (global) |
|---|---|---:|---:|---:|
| `mendelian_traits` | `minus_llr` | **0.7175** ± 0.020 | 0.7350 ± 0.006 | 4,910 |
| `mendelian_traits` | `embed_last_l2` | 0.7408 ± 0.019 | 0.7225 ± 0.006 | 4,910 |
| `mendelian_traits` | `abs_llr` | 0.6371 ± 0.021 | 0.7141 ± 0.006 | 4,910 |
| `complex_traits` | `embed_last_l2` | **0.6541** ± 0.032 | 0.5824 ± 0.021 | 564 |
| `complex_traits` | `abs_llr` | 0.5927 ± 0.033 | 0.5355 ± 0.021 | 564 |
| `complex_traits` | `minus_llr` | 0.5693 ± 0.034 | 0.5195 ± 0.021 | 564 |
| `eqtl` | `embed_last_l2` | **0.5360** ± 0.022 | 0.5306 ± 0.010 | 2,306 |
| `eqtl` | `abs_llr` | 0.5225 ± 0.022 | 0.5230 ± 0.010 | 2,306 |
| `eqtl` | `minus_llr` | 0.4946 ± 0.022 | 0.5017 ± 0.010 | 2,306 |

Consistent with #175's `exp166-p1B` cells — `minus_llr` is best on mendelian, `embed_last_l2` leads on the regulatory-heavy datasets, and the eqtl floor is close to chance. The supervised question is whether a classifier on top of `mean_ref`, `mean_alt`, and `traitgym_innerprod` lifts these macros materially.

### Where the code lives

* compute_fn returning `[B, 2 + 3·D]`: [`src/bolinas/supervised/scoring.py`](https://github.com/Open-Athena/bolinas-dna/blob/640b33a9b7c2e190882e438d4d253d21f5bd1bdd/src/bolinas/supervised/scoring.py)
* bolinas wrapper (parquet IO + slicing): [`src/bolinas/supervised/inference.py`](https://github.com/Open-Athena/bolinas-dna/blob/640b33a9b7c2e190882e438d4d253d21f5bd1bdd/src/bolinas/supervised/inference.py)
* pipeline scaffold: [`snakemake/analysis/supervised_vep/`](https://github.com/Open-Athena/bolinas-dna/tree/640b33a9b7c2e190882e438d4d253d21f5bd1bdd/snakemake/analysis/supervised_vep)
* 8 unit tests on scoring: [`tests/supervised/test_scoring.py`](https://github.com/Open-Athena/bolinas-dna/blob/640b33a9b7c2e190882e438d4d253d21f5bd1bdd/tests/supervised/test_scoring.py)

### Resources

A10G:1, g5.xlarge, us-east-2. 23m20s end-to-end (10m genome download + 13m forward passes across 3 datasets). Cluster name `supervised-vep` — staying up for iter-1.

### Next

iter-1 BFS sweep launched via `sky exec` on the same cluster (commit [`640b33a`](https://github.com/Open-Athena/bolinas-dna/commit/640b33a9b7c2e190882e438d4d253d21f5bd1bdd)): 85 (dataset × recipe × classifier) supervised combos through `compute_pairwise_metrics`, 3-fold chrom-grouped CV, `cv.mode: bfs` (1 hparam value per knob). Refinement on top performers in a follow-up iter-1b.
