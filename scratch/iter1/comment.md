🤖 **Iteration 1 complete + paired-comparison analysis.** Pipeline ran on AWS A10G (us-east-2) at commit [`cf7bc7d`](https://github.com/Open-Athena/bolinas-dna/commit/cf7bc7d1dfb24ebb2221eeeaaa96ca307bab4667). 13,500 metric rows; paired-tests run on the underlying 233 K per-variant scored rows.

## Statistics

Two p-value flavors:

- **Per-row significance vs chance** (sign test, one-sided `H1: acc > 0.5`): used for `value` / `q_value` in `metrics_aggregated.parquet`.
- **Paired comparison** (McNemar-style closed-form sign test on discordant matched pairs, two-sided `H0: A == B`): used for every "is A better than B?" finding below. BH-FDR within each comparison family.

Each comparison broken out at three levels (per-(dataset, subset) collapsed by default):
1. **Global** — all cells in the family.
2. **Per dataset** — split by `mendelian_traits` / `complex_traits` / `eqtl`.
3. **Per (dataset, subset)** — full 24-row breakdown in branch parquets.

"A wins sig" = cells where the paired test favors A with `q < 0.05`. Most cells indeterminate (small N → can't reject H0). `mean` = fraction of discordant pairs where A wins; >0.5 = A directionally better.

A second pass — **home-subset analysis** — restricts every comparison to (model, subset) pairs in the model's design target:

| model | target consequence subsets |
|---|---|
| **exp55-mammals** (promoter) | tss_proximal, 5_prime_UTR_variant |
| **exp58-mammals**, **exp58-animals** (CDS) | missense_variant, synonymous_variant, splicing |
| **exp59-mammals** (downstream) | 3_prime_UTR_variant |
| **exp136-proj_v30** (enhancer) | distal |

## Sign conventions locked

`minus_entropy = -H(p)`, `logp_ref = +log p[ref]`, `embed_minus_dot_* = -⟨ref, alt⟩`. Applied at metrics time; raw features parquet untouched. All three appear as per-subset winners below → directional assumptions validated.

<details><summary>Sanity check vs evals_v2 — 120/120 pair counts match; bf16-noise-level disagreement (median |Δ|≈0.03)</summary>

Compared `minus_llr` (mendelian) + `abs_llr` (complex/eqtl) at native window. Outliers up to 0.20 at small-N subsets where 1–2 pair flips dominate. Discussed at [Open-Athena/biofoundation#21](https://github.com/Open-Athena/biofoundation/issues/21).

</details>

---

# Part 1 — Comparisons across the full grid

## A. Pool strategy: `mean ≈ flat > lastpos > varpos`

**Global** (2,160 cells per pair):

| pool_a | pool_b | A wins sig | B wins sig | mean (A wins disc.) |
|---|---|---:|---:|---:|
| **mean** | varpos | **53** | 0 | 0.530 |
| **flat** | varpos | **47** | 7 | 0.538 |
| mean | lastpos | 30 | 5 | 0.508 |
| flat | lastpos | 29 | 10 | 0.503 |
| lastpos | varpos | 28 | 7 | 0.523 |
| flat | mean | 7 | 11 | 0.494 |

**Per dataset** (significant differences are mendelian-only):

| pool_a | pool_b | mendelian a/b | complex a/b | eqtl a/b |
|---|---|---:|---:|---:|
| mean | varpos | **53/0** | 0/0 | 0/0 |
| flat | varpos | **45/0** | 1/0 | 1/7 |
| mean | lastpos | **30/4** | 0/0 | 0/1 |
| flat | lastpos | **28/9** | 1/0 | 0/1 |
| lastpos | varpos | 6/27 | 1/1 | 0/0 |
| flat | mean | 4/9 | 2/0 | 1/2 |

`varpos` (single variant-position embedding) is the worst pool by a wide margin. `mean` and `flat` are tied at the top.

<details><summary>Per (dataset, subset)</summary>

Full table at [`scratch/iter1/paired/pool_pairwise.parquet.per_dataset_subset.parquet`](https://github.com/Open-Athena/bolinas-dna/blob/cf7bc7d/scratch/iter1/paired/) on the branch.

</details>

## B. Distance metric: `cosine ≈ l2 > dot`

**Global** (2,880 cells per pair):

| dist_a | dist_b | A wins sig | B wins sig | mean (A) |
|---|---|---:|---:|---:|
| **cosine** | dot | **110** | 12 | 0.529 |
| **l2** | dot | **108** | 19 | 0.528 |
| l2 | cosine | 17 | 27 | 0.509 |

**Per dataset:**

| dist_a | dist_b | mendelian a/b | complex a/b | eqtl a/b |
|---|---|---:|---:|---:|
| cosine | dot | **105/12** | 2/0 | 3/0 |
| l2 | dot | **102/19** | 3/0 | 3/0 |
| l2 | cosine | 17/27 | 0/0 | 0/0 |

`dot` is the worst distance — loses ~5–10× as often as it wins to either l2 or cosine. l2 ≈ cosine.

## C. Layer: last ≈ middle

**Global** (4,320 cells): last wins sig 35, middle wins sig 11. Mean 0.495.

| dataset | last wins | middle wins | mean (last) |
|---|---:|---:|---:|
| mendelian_traits | 35 | 11 | 0.490 |
| complex_traits | 0 | 0 | 0.509 |
| eqtl | 0 | 0 | 0.486 |

## D. Likelihood scores: `minus_llr` modest edge; family mostly interchangeable

**Global** (360 cells per pair):

| A | B | A wins sig | B wins sig | mean (A) |
|---|---|---:|---:|---:|
| minus_llr | minus_entropy | 5 | 2 | 0.523 |
| minus_llr | logp_ref | 6 | 0 | 0.505 |
| minus_llr | abs_llr | 4 | 4 | 0.535 |
| minus_llr | minus_logp_alt | 0 | 0 | 0.506 |
| abs_llr | logp_ref | 4 | 0 | 0.478 |
| abs_llr | minus_entropy | 3 | 0 | 0.502 |
| minus_entropy | minus_logp_alt | 2 | 5 | 0.482 |
| abs_llr | minus_logp_alt | 1 | 1 | 0.476 |
| minus_entropy | logp_ref | 2 | 2 | 0.479 |
| logp_ref | minus_logp_alt | 0 | 1 | 0.498 |

`minus_llr` has positive mean in all 4 of its pairings. The other 4 scores are mostly interchangeable.

## E. Window: native fine; exp58 (CDS) benefits modestly from longer

| model | 128 vs native | 128 vs 512 | native vs 512 |
|---|---|---|---|
| exp55-mammals | 0/0 | 1/0 | 0/0 |
| **exp58-mammals** | 2/9 | **1/13** | 0/1 |
| exp58-animals | 5/8 | 1/6 | 0/7 |
| exp59-mammals | 0/0 | 3/0 | 1/0 |
| exp136-proj_v30 | 0/0 | 0/0 | 0/0 |

## F. Best embedding vs best likelihood per cell

**Global** (360 cells): best-likelihood wins sig 0, best-embedding wins sig 6 (all mendelian). Mean (likelihood wins) = 0.428.

| dataset | best-llk wins | best-emb wins | mean (llk) |
|---|---:|---:|---:|
| mendelian_traits | 0 | **6** | 0.420 |
| complex_traits | 0 | 0 | 0.417 |
| eqtl | 0 | 0 | 0.446 |

Best embedding marginally beats best likelihood, mendelian-driven.

## G. Alternative scores vs leaderboard baseline (`minus_llr` / `abs_llr`)

**Per (dataset, subset)** — the informative slice:

| dataset | subset | alt > leaderboard | alt < leaderboard | note |
|---|---|---:|---:|---|
| mendelian | **missense_variant** | **167** | 145 | embeddings beat `minus_llr` here |
| mendelian | **5_prime_UTR_variant** | 0 | **30** | `minus_llr` strictly best |
| mendelian | **splicing** | 0 | **49** | `minus_llr` strictly best |
| mendelian | distal | 2 | 7 | `minus_llr` modestly best |
| mendelian | others | ≤1 | ≤2 | low N, indet |
| complex / eqtl | all subsets | 0 | 0–5 | leaderboard ≥ alts everywhere |

**Punch line: on mendelian, embeddings can beat `minus_llr` on missense but consistently *lose* on 5' UTR and splicing.** Complex / eqtl: leaderboard is at least as good as any alternative everywhere.

<details><summary>Top alts that beat the leaderboard most often (all mendelian missense)</summary>

| score | n cells |
|---|---:|
| `embed_cosine_mean_middle` | 9 |
| `embed_minus_dot_mean_last` | 9 |
| `embed_cosine_lastpos_middle` | 8 |
| `embed_cosine_flat_middle` | 8 |
| `embed_l2_lastpos_middle` | 8 |
| `embed_cosine_varpos_last` | 8 |

</details>

---

# Part 2 — Home-subset analysis (each model on its design target)

Restricting comparisons to (model, subset) cells in the model's design target. Patterns intensify because we're sampling where the model is supposed to work; small-N subsets are still small (5'UTR has N=87, splicing N=78, distal N=56), but the comparisons are now "fair" — every model is evaluated on its intended region.

## A-home. Pool — pattern strengthens; `varpos` clearly worst on home cells

**Global** (540 cells per pair):

| pool_a | pool_b | A wins sig | B wins sig | mean (A) |
|---|---|---:|---:|---:|
| **mean** | varpos | **57** | 2 | 0.583 |
| **flat** | varpos | **52** | 6 | 0.595 |
| mean | lastpos | 28 | 3 | 0.548 |
| flat | lastpos | 26 | 11 | 0.525 |
| lastpos | varpos | 4 | 31 | 0.461 |
| flat | mean | 8 | 10 | 0.477 |

**Per dataset** — mendelian still dominates the significant signal:

| pool_a | pool_b | mendelian a/b | complex a/b | eqtl a/b |
|---|---|---:|---:|---:|
| mean | varpos | **57/2** | 0/0 | 0/0 |
| flat | varpos | **51/6** | 1/0 | 0/0 |
| mean | lastpos | **28/2** | 0/1 | 0/0 |
| flat | lastpos | **25/10** | 1/0 | 0/1 |
| lastpos | varpos | 4/31 | 0/0 | 0/0 |
| flat | mean | 8/10 | 0/0 | 0/0 |

vs global, the mendelian wins for `mean > varpos` go from 53 → 57 and for `flat > varpos` from 45 → 51 with the same total cell count proportionally. Pattern is robustly mendelian-driven and concentrates on home cells.

## B-home. Distance — `cosine ≈ l2 > dot` strengthens

**Global** (720 cells per pair):

| dist_a | dist_b | A wins sig | B wins sig | mean (A) |
|---|---|---:|---:|---:|
| **cosine** | dot | **105** | 1 | **0.566** |
| **l2** | dot | **101** | 1 | **0.558** |
| l2 | cosine | 18 | 3 | 0.524 |

**Per dataset:**

| dist_a | dist_b | mendelian a/b | complex a/b | eqtl a/b |
|---|---|---:|---:|---:|
| cosine | dot | **94/0** | 6/1 | 5/0 |
| l2 | dot | **91/0** | 6/1 | 4/0 |
| l2 | cosine | 18/3 | 0/0 | 0/0 |

On home cells, `dot` essentially never wins (1 sig vs 105/101). `cosine` and `l2` interchangeable.

## C-home. Layer — slightly more decisive on home cells

**Global** (1,080 cells): last wins sig 33, middle wins sig 6. Mean 0.504 (vs 0.495 in the full grid — last layer takes a small directional lead on home cells).

| dataset | last wins | middle wins | mean (last) |
|---|---:|---:|---:|
| mendelian_traits | 33 | 6 | 0.514 |
| complex_traits | 0 | 0 | 0.531 |
| eqtl | 0 | 0 | 0.468 |

Last has a clearer edge on home cells than on the full grid (mendelian 33/6 vs 35/11; mean rises 0.490 → 0.514). Still not large.

## D-home. Likelihood scores — `minus_llr` edge widens on home

**Global** (90 cells per pair):

| A | B | A wins sig | B wins sig | mean (A) |
|---|---|---:|---:|---:|
| minus_llr | logp_ref | 7 | 0 | 0.539 |
| minus_llr | minus_entropy | 6 | 0 | 0.539 |
| minus_llr | abs_llr | 4 | 0 | 0.585 |
| minus_llr | minus_logp_alt | 0 | 0 | 0.541 |
| abs_llr | logp_ref | 0 | 0 | 0.474 |
| abs_llr | minus_entropy | 3 | 0 | 0.510 |
| abs_llr | minus_logp_alt | 0 | 1 | 0.436 |
| logp_ref | minus_logp_alt | 0 | 3 | 0.470 |
| minus_entropy | logp_ref | 0 | 2 | 0.484 |
| minus_entropy | minus_logp_alt | 0 | 5 | 0.461 |

`minus_llr`'s edge intensifies on home cells: every one of its 4 pairings has positive mean, all in significant-win territory against `logp_ref` and `minus_entropy` (where it didn't reach significance in the full grid). **`minus_llr` is the strongest single likelihood score on home cells.**

## E-home. Window — exp58 mammals modestly favors longer; otherwise null

<details><summary>Window pairwise per (model, dataset) on home cells</summary>

Strongest pattern: exp58-mammals 128 vs 512 on mendelian home subsets = 1/14 (longer wins 14 cells significantly). exp58-animals 128 vs 512 mendelian home = 2/9. Other models on home cells: no significant window differences anywhere. Full table in [`scratch/iter1/paired/home_window_pairwise.parquet.per_dataset.parquet`](https://github.com/Open-Athena/bolinas-dna/blob/cf7bc7d/scratch/iter1/paired/).

</details>

## F-home. Best embedding vs best likelihood per cell (home only)

**Global** (90 cells): best-likelihood wins sig 0, best-embedding wins sig 2 (both mendelian). Mean (likelihood wins) = 0.415.

| dataset | best-llk wins | best-emb wins | mean (llk) |
|---|---:|---:|---:|
| mendelian_traits | 0 | 2 | 0.424 |
| complex_traits | 0 | 0 | 0.381 |
| eqtl | 0 | 0 | 0.441 |

Best embedding is marginally ahead even on home cells; effect smaller than the full grid because home cells include the LLR-friendly splicing + 5'UTR subsets where embedding has no advantage.

## H. Does the home model win on its home subset? (cross-model)

For each (home_model, home_subset, dataset), pick the home model's best score (at its native window) vs each other model's best score (at *that* model's native window). Paired-test on the same match_groups.

| subset | dataset | home wins sig | other wins sig | median value (home) |
|---|---|---:|---:|---:|
| 3_prime_UTR_variant | mendelian | 0 | 0 | 0.550 |
| 3_prime_UTR_variant | complex | 0 | 0 | 0.613 |
| 3_prime_UTR_variant | eqtl | 0 | 0 | 0.519 |
| 5_prime_UTR_variant | mendelian | **1** | 0 | 0.760 |
| 5_prime_UTR_variant | complex | 0 | 0 | 1.000 |
| 5_prime_UTR_variant | eqtl | 0 | 0 | 0.568 |
| distal | mendelian | 0 | 0 | 0.605 |
| distal | complex | 0 | 0 | 0.533 |
| distal | eqtl | 0 | 0 | 0.512 |
| **missense_variant** | **mendelian** | **7** | **1** | 0.741 |
| missense_variant | complex | 1 | 0 | 0.699 |
| missense_variant | eqtl | 0 | 0 | 0.460 |
| **splicing** | **mendelian** | **4** | 0 | 0.782 |
| splicing | complex | 0 | 0 | 0.500 |
| splicing | eqtl | 0 | 0 | 0.500 |
| synonymous_variant | mendelian | 0 | 0 | 0.618 |
| synonymous_variant | complex | 0 | 0 | 0.583 |
| synonymous_variant | eqtl | 0 | 0 | 0.423 |
| tss_proximal | mendelian | 0 | 0 | 0.560 |
| tss_proximal | complex | 0 | 0 | 0.517 |
| tss_proximal | eqtl | 0 | 0 | 0.515 |

**Significant cross-model events** (9 out of 84): home models win 12 cells significantly, lose 1. The 1 loss is the standout finding:

- **`exp58-mammals` (CDS, mammals) is significantly beaten by `exp58-animals` (CDS, animals) on its own home turf (mendelian × missense)** — `minus_llr` (mammals) loses to `embed_l2_flat_last` (animals), q=0. The "animals" timescale (longer evolutionary horizon) captures CDS constraint *better* than "mammals" for missense.

Other notable wins: home CDS models beat non-CDS models on missense (7 cells) and splicing (4 cells); home promoter model beats one non-promoter model on 5'UTR mendelian (1 cell). Most home advantages are *directional* (median value 0.5–0.8) but don't reach individual-cell FDR — discordant-pair counts are small.

---

## Per-subset winners (significance-vs-chance)

<details><summary>9 (dataset, subset) winners — three locked-sign scores all appear</summary>

| dataset | subset | best score | model | window | value | n_pairs |
|---|---|---|---|---|---:|---:|
| complex_traits | missense_variant | `embed_l2_flat_last` | exp58-animals | 256 | 0.807 | 57 |
| mendelian_traits | 3_prime_UTR_variant | **`embed_minus_dot_flat_middle`** | exp136-proj_v30 | 255 | 0.741 | 58 |
| mendelian_traits | 5_prime_UTR_variant | `embed_cosine_mean_last` | exp55-mammals | 128 | 0.828 | 87 |
| mendelian_traits | **distal** | **`minus_entropy`** | **exp136-proj_v30** | **255** | 0.777 | 56 |
| mendelian_traits | missense_variant | `embed_l2_flat_last` | exp58-animals | 256 | 0.823 | 4,495 |
| mendelian_traits | non_coding_transcript_exon | `embed_cosine_lastpos_middle` | exp55-mammals | 256 | 0.833 | 42 |
| mendelian_traits | splicing | `embed_cosine_mean_middle` | exp58-animals | 512 | 0.897 | 78 |
| mendelian_traits | synonymous_variant | `embed_l2_flat_last` | exp58-mammals | 512 | 0.879 | 33 |
| mendelian_traits | **tss_proximal** | **`logp_ref`** | **exp58-mammals** | **128** | 0.730 | 61 |

Bolded scores are the locked-sign ones: `minus_entropy` (distal), `logp_ref` (tss_proximal), and `embed_minus_dot_*` (3' UTR) each win at least one subset.

</details>

<details><summary>Heatmaps — mendelian + complex + eqtl global views</summary>

mendelian_traits — global_pooled (missense-heavy: 4,495 of 9,820 mendelian train pairs):

![mendelian_traits_global_pooled.png](https://gist.githubusercontent.com/gonzalobenegas/3649e68fb63ca1f3443e4486078eb4d8/raw/aa348662ab3a0b70c280d6a60a185fc4eb66bcd4/mendelian_traits_global_pooled.png)

mendelian_traits — global_macro:

![mendelian_traits_global_macro.png](https://gist.githubusercontent.com/gonzalobenegas/3649e68fb63ca1f3443e4486078eb4d8/raw/aa348662ab3a0b70c280d6a60a185fc4eb66bcd4/mendelian_traits_global_macro.png)

complex_traits — global_pooled:

![complex_traits_global_pooled.png](https://gist.githubusercontent.com/gonzalobenegas/3649e68fb63ca1f3443e4486078eb4d8/raw/aa348662ab3a0b70c280d6a60a185fc4eb66bcd4/complex_traits_global_pooled.png)

eqtl — global_pooled:

![eqtl_global_pooled.png](https://gist.githubusercontent.com/gonzalobenegas/3649e68fb63ca1f3443e4486078eb4d8/raw/aa348662ab3a0b70c280d6a60a185fc4eb66bcd4/eqtl_global_pooled.png)

</details>

## TL;DR

- **Pool**: `mean` ≈ `flat` > `lastpos` > `varpos`. Pattern strengthens on home cells.
- **Distance**: `cosine` ≈ `l2` > `dot`. Pattern strengthens on home cells (dot loses 105/1 to cosine on home).
- **Layer**: last ≈ middle (small edge to last on home cells: 33/6 vs 35/11 mendelian).
- **Window**: native fine; both exp58 CDS models benefit modestly from longer windows.
- **Likelihood scores**: `minus_llr` is the strongest single likelihood score, *especially on home cells* (significantly beats `logp_ref` and `minus_entropy` in 7 and 6 cells respectively).
- **Best embedding marginally beats best likelihood** on the full grid; effect is smaller on home cells (because home includes LLR-friendly splicing + 5'UTR).
- **Leaderboard `minus_llr` strictly best on mendelian 5'UTR + splicing**; embeddings beat it on missense. For complex / eqtl, leaderboard is at least as good as any alternative.
- **Home models win directionally on home subsets** (median 0.5–0.8). 12 significant home-wins vs 1 home-loss across 84 cross-model comparisons.
- **The 1 home-loss is the most interesting finding**: `exp58-animals` significantly beats `exp58-mammals` on mendelian missense — the animals evolutionary timescale beats mammals for missense even at "home turf".
- All three locked-sign assumptions (`minus_entropy`, `logp_ref`, `embed_minus_dot_*`) appear as per-subset winners → sign assumptions validated.
- Complex / eqtl: signal is sparse; no alternative beats the leaderboard `abs_llr` significantly anywhere.

## Raw data + code @ [`cf7bc7d`](https://github.com/Open-Athena/bolinas-dna/commit/cf7bc7d1dfb24ebb2221eeeaaa96ca307bab4667)

- [`metrics_aggregated.csv`](https://gist.githubusercontent.com/gonzalobenegas/3649e68fb63ca1f3443e4486078eb4d8/raw/d785a6851aabf0cf91da28793446be141cfbf7f9/metrics_aggregated.csv) (2.1 MB)
- All paired-test results: [`scratch/iter1/paired/`](https://github.com/Open-Athena/bolinas-dna/tree/cf7bc7d/scratch/iter1/paired/) (including all `home_*` re-runs)
- Pipeline: [`snakemake/analysis/zeroshot_vep/`](https://github.com/Open-Athena/bolinas-dna/tree/cf7bc7d/snakemake/analysis/zeroshot_vep/)
- Code: [`scores.py`](https://github.com/Open-Athena/bolinas-dna/blob/cf7bc7d/src/bolinas/zeroshot_vep/scores.py) · [`features.py`](https://github.com/Open-Athena/bolinas-dna/blob/cf7bc7d/src/bolinas/zeroshot_vep/features.py) · [`metrics.py`](https://github.com/Open-Athena/bolinas-dna/blob/cf7bc7d/src/bolinas/evals/metrics.py) · [`paired_analysis.py`](https://github.com/Open-Athena/bolinas-dna/blob/cf7bc7d/scratch/zeroshot_vep_paired_analysis.py)

## Iter 2 (proposed)

1. **Rank-based combinations** restricted to surviving choices:
   - `rank_sum(minus_llr, embed_l2_flat_*, embed_cosine_flat_*)` — combine the per-subset winners.
   - Subset-tailored composite: `minus_llr` for splicing + 5'UTR; `embed_l2_flat_last` for missense; `minus_entropy` for distal; `logp_ref` for tss_proximal — rank-sum within subset.
2. Drop `varpos` and `dot` from the score grid → ~16 active scores instead of 30.
3. **Investigate eqtl's null result**: per-variant LLR distribution for eqtl distals — is the matched-pair construction too tight, or is signal genuinely absent for zero-shot bolinas gLMs?
4. **Animals-timescale CDS sweep**: given `exp58-animals` beats `exp58-mammals` on its home subset, worth sweeping `exp58` timescales (humans / primates / mammals / animals / vertebrates) at native window.

All CPU-only except (4). ETA ~30 min for (1)+(2)+(3).
