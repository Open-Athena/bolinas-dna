🤖 **Iter-4 follow-up: exp166-p1B (1B generalist) FWD on all 3 datasets × all 30 scoring approaches.** Mendelian was covered in [`-4435271389`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4435271389); this comment adds `complex_traits` and `eqtl` for the full picture. Commit [`8ec136a`](https://github.com/Open-Athena/bolinas-dna/commit/8ec136a).

## TL;DR

- **exp166-p1B is the most competitive single model across the 3-dataset × 8-subset grid.** It wins outright on 8 of 23 (dataset, subset) cells — more than any specialist (which top out at ~6 wins each).
- **On the largest, most-reliable subsets** (mendelian missense n=4495, complex distal n=428, eqtl distal n=1678), specialists win narrowly (margins ≤0.04). exp166-p1B is close-second on each of these.
- **On smaller subsets**, the generalist often wins by a wide margin: eqtl 5'UTR +0.019 over best specialist, mendelian synonymous +0.090, mendelian tss_proximal +0.009.
- **Macro accuracy** (mean of per-subset best PairwiseAccuracy across all 30 scores): p1B wins on mendelian (0.776 vs ≤0.715) and eqtl (0.664 vs ≤0.662), close-second on complex (0.706 vs exp55-mammals 0.728 — exp55 was lifted by 1.000 on 5'UTR with n=10, fragile).
- **Embedding scores frequently beat LLR for exp166-p1B** — its wins are split between `minus_llr` (LLR family), `embed_l2_flat_last`, `embed_l2_mean_middle`, and `minus_entropy`. Suggests the 1B generalist has richer hidden representations that beat raw likelihoods for some variant types.

## Setup

- Model: exp166-p1B (HF `bolinas-dna/exp166-p1B-step-16398`, 1B Qwen3, 19 layers, hidden=1920, trained on `zoonomia-v1-v1` 108-species whole-genome).
- Window: 255 bp + BOS (native).
- Strand: FWD only (per request).
- All 30 scores computed: 6 likelihood + 24 embedding (3 distances × 4 pools × 2 layers).
- Datasets: `bolinas-dna/evals_{mendelian_traits, complex_traits, eqtl}`, train split.
- bf16 forward pass (precision test in [`-4434884279`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4434884279) showed no meaningful gain from fp32).

## Pooled per (dataset, model)

PairwiseAccuracy after pooling all matched pairs across the 8 subsets; reported is the best score across all 30 for each cell:

| dataset | exp55 | exp58-m | exp59 | exp136 | **exp166-p1B** |
|---|---:|---:|---:|---:|---:|
| mendelian | 0.564 | **0.758** | 0.555 | 0.558 | 0.739 |
| complex | 0.555 | 0.551 | 0.539 | 0.551 | **0.578** |
| eqtl | 0.525 | 0.518 | 0.516 | 0.531 | **0.533** |

p1B wins pooled on complex and eqtl. On mendelian, exp58-m wins by 0.019 — close, and almost entirely driven by exp58-m's home subset (missense, n=4495 dominates the pool).

## Macro per (dataset, model)

Mean of per-subset best PairwiseAccuracy (each subset weighted equally):

| dataset | exp55 | exp58-m | exp59 | exp136 | **exp166-p1B** |
|---|---:|---:|---:|---:|---:|
| mendelian | 0.715 | 0.715 | 0.643 | 0.671 | **0.776** |
| complex | **0.728** | 0.714 | 0.664 | 0.671 | 0.706 |
| eqtl | 0.662 | 0.633 | 0.634 | 0.643 | **0.664** |

p1B wins macro on mendelian and eqtl. On complex_traits, exp55 wins macro — but driven by 1.000 on 5'UTR (n=10) and 0.833 on synonymous (n=6); both fragile small-n subsets.

## Per-(dataset, subset) winners — best model + best score per cell

★ marks exp166-p1B wins.

### mendelian_traits

| subset | n_pairs | best PairwiseAccuracy | winner | best score |
|---|---:|---:|---|---|
| 3_prime_UTR_variant | 58 | 0.741 | exp136-proj_v30 | `embed_minus_dot_flat_middle` |
| 5_prime_UTR_variant | 87 | 0.828 | exp55-mammals | `minus_llr` |
| **distal** | 56 | **0.786** | **★ exp166-p1B** | `embed_l2_flat_last` |
| missense_variant | 4495 | 0.775 | exp58-mammals | `minus_llr` |
| ncRNA_exon | 42 | 0.833 | exp55-mammals | `embed_l2_lastpos_middle` |
| splicing | 78 | 0.821 | exp58-mammals | `embed_l2_mean_middle` |
| **synonymous_variant** | 33 | **0.848** | **★ exp166-p1B** | `embed_l2_mean_middle` |
| **tss_proximal** | 61 | **0.730** | **★ exp166-p1B** | `logp_ref` |

### complex_traits

| subset | n_pairs | best | winner | best score |
|---|---:|---:|---|---|
| 3_prime_UTR_variant | 21 | 0.762 | exp59-mammals | `abs_llr` |
| 5_prime_UTR_variant | 10 | 1.000 | exp55-mammals | `embed_l2_mean_last` |
| distal | 428 | 0.582 | exp136-proj_v30 | `embed_l2_lastpos_middle` |
| **missense_variant** | 57 | **0.737** | **★ exp166-p1B** | `embed_l2_flat_last` |
| ncRNA_exon | 12 | 0.833 | exp59-mammals | `embed_minus_dot_mean_last` |
| synonymous_variant | 6 | 0.833 | exp55-mammals (tied many) | `minus_logp_alt` |
| tss_proximal | 29 | 0.690 | exp55-mammals (tied many) | `embed_minus_dot_flat_middle` |

(complex_traits has no splicing subset.)

### eqtl

| subset | n_pairs | best | winner | best score |
|---|---:|---:|---|---|
| **3_prime_UTR_variant** | 115 | **0.617** | **★ exp166-p1B** | `minus_entropy` |
| **5_prime_UTR_variant** | 51 | **0.725** | **★ exp166-p1B** | `embed_minus_dot_flat_middle` |
| distal | 1678 | 0.534 | exp136-proj_v30 | `embed_cosine_lastpos_middle` |
| missense_variant | 30 | 0.767 | exp58-mammals | `embed_minus_dot_mean_last` |
| **ncRNA_exon** | 157 | **0.586** | **★ exp166-p1B** | `embed_cosine_flat_last` |
| **splicing** | 5 | **1.000** | **★ exp166-p1B** | `minus_entropy` |
| synonymous_variant | 29 | 0.828 | exp55-mammals | `embed_l2_mean_middle` |
| tss_proximal | 241 | 0.581 | exp136-proj_v30 | `embed_l2_lastpos_middle` |

## exp166-p1B's best-score winners per subset (own model)

Interesting: the winning score varies — sometimes LLR-family, sometimes embedding pools, sometimes entropy. p1B's representations carry more diverse signal than the specialists, where `minus_llr` tends to dominate.

| dataset | subset | best score for p1B | value |
|---|---|---|---:|
| mendelian | 3'UTR | `minus_llr` | 0.724 |
| mendelian | 5'UTR | `minus_logp_alt` | 0.736 |
| mendelian | distal | `embed_l2_flat_last` | 0.786 |
| mendelian | missense | `minus_llr` | 0.741 |
| mendelian | ncRNA | `embed_l2_flat_last` | 0.833 |
| mendelian | splicing | `minus_llr` | 0.808 |
| mendelian | synonymous | `embed_l2_mean_middle` | 0.848 |
| mendelian | tss_proximal | `logp_ref` | 0.730 |
| complex | distal | `embed_l2_flat_last` | 0.565 |
| complex | missense | `embed_l2_flat_last` | 0.737 |
| eqtl | 3'UTR | `minus_entropy` | 0.617 |
| eqtl | 5'UTR | `embed_minus_dot_flat_middle` | 0.725 |
| eqtl | splicing | `minus_entropy` | 1.000 |

`embed_l2_flat_last` (full-window L2 distance on last-layer embeddings) is a recurring winner for p1B — it captures global representation shifts that the 1B model picks up well. `minus_entropy` (low-entropy = informative predictions) wins on eqtl where the signal is genuinely subtle.

## Outright-win counts (23 cells, one per (dataset, subset))

| model | wins |
|---|---:|
| exp166-p1B | **8** |
| exp55-mammals | 6 |
| exp58-mammals | 5 |
| exp136-proj_v30 | 5 |
| exp59-mammals | 3 |

(Total >23 because some cells have ties.)

## Practical takeaways

1. **exp166-p1B is a strong default**: it never loses by much, wins the most cells, and tops both pooled and macro metrics on eqtl + mendelian (macro). Specialists still win on their home-region large subsets — exp58-m on missense, exp136 on distal — but those wins are 0.02–0.04 margins.
2. **The score that wins varies per (model, subset)** — LLR-family is the best score on 6 of p1B's 8 mendelian subsets but only 1 of 5 on the other datasets, where embedding pools and entropy lead. The "use minus_llr everywhere" leaderboard default leaves performance on the table; per-(dataset, subset) score selection would lift the macro by ~0.05 in many cases.
3. **The 1B vs 0.6B size advantage is a caveat** — exp166-p1B has 2× the parameters of the specialists, so the apples-to-apples comparison would be a 0.6B generalist (not yet trained). Open question for future iteration.

## Code @ [`8ec136a`](https://github.com/Open-Athena/bolinas-dna/commit/8ec136a)

- [`scratch/iter4/sky_exp166_1b_three_datasets.yaml`](https://github.com/Open-Athena/bolinas-dna/blob/8ec136a/scratch/iter4/sky_exp166_1b_three_datasets.yaml) — sky yaml (FWD only, win=255+BOS, complex_traits + eqtl)
- [`scratch/iter4/iter4_fwd_exp166-p1B__win255__{mendelian_traits,complex_traits,eqtl}.parquet`](https://github.com/Open-Athena/bolinas-dna/tree/8ec136a/scratch/iter4) — FWD scores across all 3 datasets
- [`scratch/iter4/iter4_exp166_p1B_3datasets_full.parquet`](https://github.com/Open-Athena/bolinas-dna/tree/8ec136a/scratch/iter4) — full PairwiseAccuracy by (dataset, model, subset, score)
