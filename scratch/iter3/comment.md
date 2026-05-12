🤖 **Iteration 3 — downstream-effect scoring scout on exp55-mammals.** Single-model, single-window scout at commit [`0464553`](https://github.com/Open-Athena/bolinas-dna/commit/046455301e290836a79790164b2a345e1395c4a7). Inspired by the per-position-effect approaches in [Brixi et al. 2025](https://www.nature.com/articles/s41588-025-02347-3) and GPN-Star.

## Setup

For each variant, 2 forward passes (REF-context + ALT-context). At each output index `i ∈ [p, T−2]` (the range where the AR model's prediction depends on the swap — `i = p` is the immediate next-token prediction, then long-range), compute a divergence between the REF and ALT predicted 4-nucleotide softmax distributions, then aggregate over positions.

| score | per-position metric | aggregation |
|---|---|---|
| `down_jsd_mean / _max` | Jensen-Shannon divergence | mean / max |
| `down_l1_mean / _max` | Σ_k \|p_ref(k) − p_alt(k)\| | mean / max |
| `down_l2_mean / _max` | √Σ_k (p_ref(k) − p_alt(k))² | mean / max |
| `down_linf_mean / _max` | max_k \|p_ref(k) − p_alt(k)\| | mean / max |

→ 8 candidate scores. All operate on the **renormalized 4-nucleotide softmax** (additive-constant-invariant — log-prob slicing is recomputed against {A,C,G,T} only, then exp'd).

**Scout config**:
- Model: **exp55-mammals** (the promoter model)
- Window: 256 (native)
- All 3 datasets, *every subset* computed; **analysis restricted to `tss_proximal` + `5_prime_UTR_variant`** (exp55's design-targeted subsets)
- 2-pass forward (REF, ALT); no caching
- ~15.5K variants total × 2 passes = ~31K forward passes; completed in 5 min on A10G after fixing a hot-path bottleneck (see below).

**AR limitation explicit**: only the downstream half of the effect footprint is measured. Bidirectional MLMs (GPN/GPN-Star, AlphaGenome) see the variant in both directions and capture upstream perturbations too. Our scout captures the AR-half only.

## Results — solo PairwiseAccuracy per cell, sign-test q-value (BH within iter-3's 6 cells × 8 scores)

<details><summary>Full per-cell table — all 48 (cell × down score) rows</summary>

| dataset | subset | score | value | n_pairs | p_value | q_value |
|---|---|---|---:|---:|---:|---:|
| mendelian | 5_prime_UTR_variant | **down_jsd_mean** | **0.839** | 87 | 0.0000 | **0.0000** |
| mendelian | 5_prime_UTR_variant | down_l1_mean | 0.828 | 87 | 0.0000 | 0.0000 |
| mendelian | 5_prime_UTR_variant | down_l2_mean | 0.828 | 87 | 0.0000 | 0.0000 |
| mendelian | 5_prime_UTR_variant | down_linf_mean | 0.828 | 87 | 0.0000 | 0.0000 |
| mendelian | 5_prime_UTR_variant | down_jsd_max | 0.770 | 87 | 0.0000 | 0.0000 |
| mendelian | 5_prime_UTR_variant | down_l1_max | 0.759 | 87 | 0.0000 | 0.0000 |
| mendelian | 5_prime_UTR_variant | down_linf_max | 0.759 | 87 | 0.0000 | 0.0000 |
| mendelian | 5_prime_UTR_variant | down_l2_max | 0.736 | 87 | 0.0000 | 0.0000 |
| mendelian | tss_proximal | **down_jsd_mean** | **0.705** | 61 | 0.0009 | **0.0041** |
| mendelian | tss_proximal | down_jsd_max | 0.705 | 61 | 0.0009 | 0.0041 |
| mendelian | tss_proximal | down_l1_max | 0.705 | 61 | 0.0009 | 0.0041 |
| mendelian | tss_proximal | down_l2_max | 0.689 | 61 | 0.0022 | 0.0089 |
| mendelian | tss_proximal | down_linf_max | 0.672 | 61 | 0.0049 | 0.0182 |
| mendelian | tss_proximal | down_l1_mean | 0.623 | 61 | 0.0361 | 0.0722 |
| mendelian | tss_proximal | down_l2_mean | 0.623 | 61 | 0.0361 | 0.0722 |
| mendelian | tss_proximal | down_linf_mean | 0.623 | 61 | 0.0361 | 0.0722 |
| eqtl | 5_prime_UTR_variant | **down_linf_mean** | **0.667** | 51 | 0.0120 | **0.0340** |
| eqtl | 5_prime_UTR_variant | down_l2_mean | 0.647 | 51 | 0.0244 | 0.0559 |
| eqtl | 5_prime_UTR_variant | down_jsd_mean | 0.628 | 51 | 0.0460 | 0.0848 |
| eqtl | 5_prime_UTR_variant | down_l1_mean | 0.628 | 51 | 0.0460 | 0.0848 |
| eqtl | 5_prime_UTR_variant | down_l2_max | 0.608 | 51 | 0.0804 | 0.1331 |
| eqtl | 5_prime_UTR_variant | down_l1_max | 0.569 | 51 | 0.2005 | 0.3105 |
| eqtl | 5_prime_UTR_variant | down_linf_max | 0.569 | 51 | 0.2005 | 0.3105 |
| eqtl | 5_prime_UTR_variant | down_jsd_max | 0.549 | 51 | 0.2879 | 0.4065 |
| eqtl | tss_proximal | **down_l1_mean** | **0.573** | 241 | 0.0142 | **0.0358** |
| eqtl | tss_proximal | down_linf_mean | 0.573 | 241 | 0.0142 | 0.0358 |
| eqtl | tss_proximal | down_l2_mean | 0.569 | 241 | 0.0195 | 0.0469 |
| eqtl | tss_proximal | down_jsd_mean | 0.548 | 241 | 0.0781 | 0.1331 |
| eqtl | tss_proximal | down_jsd_max | 0.523 | 241 | 0.2598 | 0.3779 |
| eqtl | tss_proximal | down_linf_max | 0.523 | 241 | 0.2598 | 0.3779 |
| eqtl | tss_proximal | down_l1_max | 0.510 | 241 | 0.3984 | 0.4780 |
| eqtl | tss_proximal | down_l2_max | 0.510 | 241 | 0.3984 | 0.4780 |
| complex | 5_prime_UTR_variant | **down_l2_mean** | **0.900** | 10 | 0.0107 | **0.0322** |
| complex | 5_prime_UTR_variant | down_l1_mean | 0.700 | 10 | 0.1719 | 0.2898 |
| complex | 5_prime_UTR_variant | down_jsd_mean | 0.600 | 10 | 0.3770 | 0.4762 |
| complex | 5_prime_UTR_variant | down_l1_max | 0.600 | 10 | 0.3770 | 0.4762 |
| complex | 5_prime_UTR_variant | down_l2_max | 0.600 | 10 | 0.3770 | 0.4762 |
| complex | 5_prime_UTR_variant | down_linf_max | 0.600 | 10 | 0.3770 | 0.4762 |
| complex | 5_prime_UTR_variant | down_jsd_max | 0.400 | 10 | 0.8281 | 0.8281 |
| complex | 5_prime_UTR_variant | down_linf_mean | 0.500 | 10 | 0.6230 | 0.6864 |
| complex | tss_proximal | down_l1_max | 0.517 | 29 | 0.5000 | 0.5581 |
| complex | tss_proximal | down_l2_max | 0.517 | 29 | 0.5000 | 0.5581 |
| complex | tss_proximal | down_linf_max | 0.517 | 29 | 0.5000 | 0.5581 |
| complex | tss_proximal | down_l1_mean | 0.483 | 29 | 0.6445 | 0.6725 |
| complex | tss_proximal | down_l2_mean | 0.483 | 29 | 0.6445 | 0.6725 |
| complex | tss_proximal | down_linf_mean | 0.483 | 29 | 0.6445 | 0.6725 |
| complex | tss_proximal | down_jsd_mean | 0.448 | 29 | 0.7709 | 0.7709 |
| complex | tss_proximal | down_jsd_max | 0.448 | 29 | 0.7709 | 0.7709 |

</details>

### Headline: best `down_*` per cell vs iter-1 winner on the same cell

| dataset × subset | N | iter-1 winner | iter-1 value (q) | iter-3 winner | iter-3 value (q) |
|---|---:|---|---|---|---|
| mendelian × 5'UTR | 87 | `minus_llr` | 0.828 (q=0.000) | **`down_jsd_mean`** | **0.839 (q=0.000)** |
| mendelian × tss_proximal | 61 | `embed_l2_flat_last` | 0.721 (q=0.003) | `down_jsd_mean` | 0.705 (q=0.004) |
| eqtl × 5'UTR | 51 | `embed_cosine_flat_last` | 0.706 (q=0.487 — **NOT sig**) | **`down_linf_mean`** | **0.667 (q=0.034)** |
| eqtl × tss_proximal | 241 | `embed_l2_mean_last` | 0.577 (q=0.666 — **NOT sig**) | **`down_l1_mean`** | **0.573 (q=0.036)** |
| complex × 5'UTR | 10 | `embed_l2_mean_last` | 1.000 (q=0.206 — N=10 fragile) | `down_l2_mean` | 0.900 (q=0.032) |
| complex × tss_proximal | 29 | `embed_minus_dot_flat_middle` | 0.690 (q=0.624 — NOT sig) | `down_l1_max` | 0.517 (q=0.558) |

**Note on q values**: iter-1 q-values are corrected over the full 30-score × 5-model × 3-window × 8-subset family. Iter-3 q-values are corrected over just iter-3's 8-score × 6-cell family (48 tests). The smaller test family makes it easier for iter-3 to clear FDR — that's why eqtl results "appear" significant for iter-3 but didn't for iter-1, despite very similar underlying PairwiseAccuracy values. **The underlying PairwiseAccuracy values are nearly identical between iter-1 and iter-3 winners** — not new information, just a different correction scope.

### Paired McNemar — best `down_*` vs best iter-1 score on same matched pairs (BH within 6 cells)

| dataset × subset | best down | best iter1 | paired value (down wins disc.) | n_down/n_iter1 | paired_p | paired_q |
|---|---|---|---:|---|---:|---:|
| mendelian × 5'UTR | `down_jsd_mean` | `minus_llr` | 0.538 | 7/6 | 1.00 | 1.00 |
| mendelian × tss_proximal | `down_jsd_mean` | `embed_l2_flat_last` | 0.455 | 5/6 | 1.00 | 1.00 |
| eqtl × tss_proximal | `down_l1_mean` | `embed_l2_mean_last` | 0.488 | 20/21 | 1.00 | 1.00 |
| eqtl × 5'UTR | `down_linf_mean` | `minus_entropy` | 0.438 | 7/9 | 0.80 | 1.00 |
| complex × tss_proximal | `down_l1_max` | `embed_minus_dot_flat_middle` | 0.333 | 5/10 | 0.30 | 1.00 |
| complex × 5'UTR | `down_l1_mean` | `embed_l2_mean_last` | 0.000 | 0/1 | 1.00 | 1.00 |

**No paired test shows down_* significantly different from iter-1 winners.** Where iter-1 had signal (mendelian's two cells), iter-3 ties. Where iter-1 didn't have signal (eqtl, complex), iter-3 also doesn't pull significantly ahead in the paired test — its solo win is a BH-scope artifact, not new information.

## TL;DR

- **`down_jsd_mean` is the strongest downstream-effect variant** (matches `minus_llr` on mendelian × 5'UTR at 0.839 vs 0.828). JSD + mean-over-positions is the right combination.
- **Downstream-effect scoring matches but does NOT meaningfully exceed iter-1 winners** on the exp55-mammals home subsets. AR's downstream-only footprint captures the same signal that LLR + embedding distance already encode.
- **No new information beyond iter-1/iter-2** for these subsets. The paired-test verdict is "tied" everywhere — including cells where iter-3 "looks" newly-significant in solo form (it's just BH scope, same underlying value).
- This is consistent with the hypothesis that **bidirectional formulations (GPN/GPN-Star) have an inherent advantage from seeing both upstream and downstream perturbations**, which AR can't replicate from a single forward pass.

## Practical note: Genome lookup was a major bottleneck

`biofoundation.data.Genome` stores chromosome sequences in a pandas Series with pyarrow-backed strings; per-variant lookup via `self._genome[chrom]` is slow when the Series is arrow-backed (~5 var/sec on the scout). Monkey-patching `genome._genome` to a plain `dict[str, str]` at startup speeds the pipeline ~20×. May be worth filing upstream in biofoundation — it'd benefit every snakemake pipeline that uses Genome with the current pandas/pyarrow versions.

## Code @ [`0464553`](https://github.com/Open-Athena/bolinas-dna/commit/046455301e290836a79790164b2a345e1395c4a7)

- [`scratch/zeroshot_vep_iter3_scout.py`](https://github.com/Open-Athena/bolinas-dna/blob/0464553/scratch/zeroshot_vep_iter3_scout.py) — forward pass + 8 down_* scores
- [`scratch/zeroshot_vep_iter3_analyze.py`](https://github.com/Open-Athena/bolinas-dna/blob/0464553/scratch/zeroshot_vep_iter3_analyze.py) — solo + paired analysis
- [`scratch/iter3/`](https://github.com/Open-Athena/bolinas-dna/tree/0464553/scratch/iter3/) — parquets + analysis log

## Where this leaves us

- **Best zero-shot scoring rules (overall, from iter 1–3)**:
  - mendelian: `rk_minus_llr_plus_top_emb` (iter-2 composite, macro 0.610)
  - For tss_proximal specifically: `logp_ref` (iter-1)
  - For 5'UTR: `minus_llr` ≈ `down_jsd_mean` (iter-1 / iter-3 tied)
  - For missense: `embed_l2_flat_last` (iter-1)
  - For distal: `minus_entropy` (iter-1)
  - For splicing: `embed_cosine_mean_middle` (iter-1)
  - For complex/eqtl: no zero-shot bolinas gLM score reliably beats chance (per the selection-strength argument from iter-2 follow-up)
- **Don't bother with**: AR downstream-effect scoring at this model scale + window. Wasn't worse than existing scores but didn't add information; bidirectional methods would be the path if we wanted more.

I'd recommend stopping the zero-shot exploration here unless a specific question opens up.
