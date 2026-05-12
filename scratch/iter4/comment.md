🤖 **Iteration 4 — strand-handling exploration.** Scout: same 30 iter-1 scores but computed on the **reverse-complement** strand. Then compared FWD / RC / AVG (per-variant mean) per (score, subset). Commit [`5a2f4b6`](https://github.com/Open-Athena/bolinas-dna/commit/5a2f4b67865b6a7effb2c860a9a5045976a93b58).

## Setup

- exp55-mammals × native window 256 × mendelian (train, 9820 variants, all 8 subsets)
- 4-pass forward on the **RC of the centered window**:
  - DNA-space variant index: `v_rc = W − 1 − (W // 2)` — 127 for W=256 (vs 128 FWD); 127 for W=255 (= FWD; symmetric for odd W).
  - Tokenized position: `v_rc + n_prefix`. exp55 has no BOS, so `tok_v_rc = 127`.
  - Allele indices: `NUC_TO_IDX[complement(ref)]` / `NUC_TO_IDX[complement(alt)]`.
  - Asserts `rc_seq[v_rc] == complement(ref)` per variant.
- The forward-strand scores were already on S3 from iter-1; this run only computed the RC parquet.
- Sign convention applied AFTER averaging (equivalent since averaging is linear).
- Same `Genome._genome` → dict speedup from [iter-3](https://github.com/Open-Athena/biofoundation/issues/23); ~3 min A10G total for the 9820 × 4 = ~40K forward passes.

## Sanity check: LLR-family is strand-consistent — no bug

I initially wrote a stronger "exp55 is not strand-invariant" headline; checking the canonical score (LLR) on its own walks that back. Reporting the bug-check first since it changes the framing:

| subset | n | Pearson(llr_fwd, llr_rc) | same-sign rate | mean(llr_rc − llr_fwd) |
|---|---:|---:|---:|---:|
| 3_prime_UTR_variant | 116 | 0.873 | 0.853 | +0.093 |
| 5_prime_UTR_variant | 174 | 0.843 | 0.897 | −0.974 |
| distal | 112 | 0.847 | 0.804 | +0.028 |
| missense_variant | 8990 | 0.825 | 0.783 | +0.069 |
| non_coding_transcript_exon | 84 | 0.892 | 0.786 | −0.738 |
| splicing | 156 | 0.837 | 0.827 | +0.081 |
| synonymous_variant | 66 | 0.712 | 0.833 | −0.224 |
| tss_proximal | 122 | 0.767 | 0.820 | +0.259 |

- **Pearson 0.71–0.89, ~0.83 typical.** Both sides are looking at the same biology with complementary context (FWD-LLR conditions on the upstream half; RC-LLR conditions on the downstream half complemented). 0.83 is consistent with that — strong shared signal, real per-variant differences from the asymmetric context.
- **Same-sign rate 78–90%** — sign convention propagates correctly.
- **Bias ≈ 0** for most subsets; the larger biases on 5'UTR / ncRNA are driven by individual high-magnitude pathogenic outliers (e.g. APC 5'UTR at chr5:112707526 with FWD LLR=−60.6, RC LLR=−62.6 — extreme but agreeing on both strands).
- **Label discrimination preserved**: every subset shows pathogenic (label=1) with more-negative mean LLR than benign in *both* FWD and RC, with similar magnitudes (e.g. 5'UTR: FWD label=1=−12.4 vs label=0=−3.7; RC: −13.3 vs −4.7).
- **Quantile-bin monotonicity is clean**: FWD-LLR quantile bins map to monotonically increasing RC-LLR means (most-negative FWD bin: mean −9.96 / RC mean −9.05; positive FWD bin: 2.45 / RC 1.82).

So the RC implementation is correct, and exp55-mammals does have **strong, but not identical**, per-variant agreement between the two strands' LLR signals.

## Finding 1: Strand-invariance varies enormously by score class

Pooled Pearson(FWD, RC) across **all 9820 variants**, per score, sorted high → low:

| score | Pearson | Spearman |
|---|---:|---:|
| `embed_minus_dot_flat_last` | **0.938** | 0.836 |
| `embed_minus_dot_mean_last` | **0.919** | 0.869 |
| `llr` / `minus_llr` | **0.840** | 0.755 |
| `minus_logp_alt` | 0.828 | 0.660 |
| `abs_llr` | 0.828 | 0.574 |
| `embed_l2_flat_last` | 0.815 | 0.639 |
| `embed_l2_flat_middle` | 0.806 | 0.836 |
| `minus_entropy` | 0.797 | 0.609 |
| `embed_cosine_flat_middle` | 0.764 | 0.830 |
| `logp_ref` | 0.721 | 0.730 |
| `embed_l2_mean_last` | 0.696 | 0.570 |
| `embed_cosine_flat_last` | 0.602 | 0.566 |
| `embed_minus_dot_mean_middle` | 0.597 | 0.638 |
| `embed_l2_mean_middle` | 0.534 | 0.724 |
| `embed_minus_dot_varpos_last` | 0.512 | 0.425 |
| `embed_cosine_mean_last` | 0.491 | 0.585 |
| `embed_minus_dot_lastpos_last` | 0.402 | 0.289 |
| `embed_cosine_mean_middle` | 0.398 | 0.723 |
| `embed_minus_dot_flat_middle` | 0.361 | 0.374 |
| `embed_l2_lastpos_last` | 0.327 | 0.362 |
| `embed_cosine_varpos_middle` | 0.251 | 0.241 |
| `embed_l2_varpos_middle` | 0.241 | 0.261 |
| `embed_l2_varpos_last` | 0.225 | −0.089 |
| `embed_minus_dot_varpos_middle` | 0.137 | 0.153 |
| `embed_l2_lastpos_middle` | 0.103 | 0.295 |
| `embed_cosine_lastpos_last` | 0.067 | 0.278 |
| `embed_minus_dot_lastpos_middle` | 0.038 | 0.028 |
| `embed_cosine_lastpos_middle` | 0.020 | 0.297 |
| `embed_cosine_varpos_last` | **−0.257** | −0.301 |

Patterns:

- **`*_flat_*` pools dominate the top.** The full-window representation is the most strand-invariant — `flat_last` ≥ 0.6 for all three distances, with `minus_dot_flat_last` at 0.938.
- **Likelihood family clusters at ~0.83.** `llr`, `minus_llr`, `abs_llr`, `minus_logp_alt`, `minus_entropy`, `logp_ref` are all 0.72–0.84. (`llr` and `minus_llr` are identical in Pearson since sign is irrelevant.)
- **`*_mean_*` pools intermediate (0.4–0.7).** Mean-pooling preserves some strand structure but loses more than flat (since the per-position context differs and gets averaged differently across strands).
- **`*_lastpos_*` and `*_varpos_*` near zero or negative.** These pick out specific tokens whose left-context windows are entirely different in FWD vs RC. There's no construction-level reason for these to correlate, and they don't.
- **Only one score is genuinely anti-correlated**: `embed_cosine_varpos_last` at −0.257. Averaging it across strands gives essentially noise.

## Finding 2: AVG doesn't reliably beat best-single-strand on a same-score basis

Per-subset same-score comparison for the most relevant scores (this is the right way to read AVG benefit — comparing different scores between modes is misleading):

| score | subset | n | FWD | RC | AVG | Δ(AVG − best strand) |
|---|---|---:|---:|---:|---:|---:|
| `minus_llr` | 5_prime_UTR_variant | 87 | **0.828** | 0.713 | 0.805 | −0.023 |
| `minus_llr` | tss_proximal | 61 | **0.672** | 0.639 | 0.656 | −0.016 |
| `minus_llr` | missense | 4495 | 0.500 | 0.505 | 0.505 | +0.000 |
| `embed_cosine_mean_last` | 5_prime_UTR_variant | 87 | 0.816 | 0.805 | **0.851** | **+0.034** |
| `embed_cosine_lastpos_last` | splicing | 78 | 0.590 | 0.667 | **0.705** | **+0.038** |
| `embed_cosine_lastpos_last` | non_coding_transcript_exon | 42 | 0.738 | 0.714 | **0.810** | **+0.071** |
| `embed_minus_dot_flat_last` | distal | 56 | 0.750 | 0.696 | 0.750 | 0.000 |
| `embed_minus_dot_flat_last` | non_coding_transcript_exon | 42 | 0.357 | 0.357 | **0.405** | **+0.048** |
| `logp_ref` | tss_proximal | 61 | 0.648 | **0.713** | 0.680 | −0.033 |
| `logp_ref` | 5_prime_UTR_variant | 87 | 0.592 | 0.632 | **0.655** | +0.023 |

It goes both ways. AVG helps on some (score, subset) cells (+0.03 to +0.07) and hurts on others (−0.02 to −0.07). Across the full grid of (30 scores × 8 subsets) = 240 cells per comparison, the paired McNemar test gives:

| comparison | n_a_wins (q<0.05) | n_b_wins (q<0.05) | mean discordant-AVG win rate | median |
|---|---:|---:|---:|---:|
| AVG vs FWD | 1 | 0 | 0.514 | 0.502 |
| AVG vs RC  | 0 | 0 | 0.558 | 0.515 |
| FWD vs RC  | 0 | 0 | 0.521 | 0.500 |

**No comparison shows a strand mode is significantly better at the (score, subset) level after BH-FDR.** AVG has a small directional edge (mean discordant-pair win rate of 0.558 vs RC, 0.514 vs FWD) but nothing reaches significance.

## TL;DR

- **No bug.** LLR Pearson FWD↔RC is 0.71–0.89 (~0.83 typical), same-sign rate 78–90%, label discrimination preserved on both strands, quantile-bin relationship is clean and monotonic.
- **Strand-invariance depends on the score class, not on the model** (pooled across 9820 variants):
  - `*_flat_last` (full-window last-layer representation): Pearson 0.6–0.94 — most invariant. `minus_dot_flat_last` tops the list at 0.938.
  - Likelihood family (`llr`/`minus_llr`/`minus_logp_alt`/`logp_ref`/`minus_entropy`/`abs_llr`): Pearson 0.72–0.84 — strong, complementary-context disagreement.
  - `*_mean_*` pools: Pearson 0.40–0.70 — averaging over positions partially loses strand structure.
  - `*_varpos_*` and `*_lastpos_*` pools: Pearson 0.02–0.51, with `embed_cosine_varpos_last` actually **anti-correlated** (−0.26). These pools pick a single token whose left-context window is entirely different between strands — no construction-level reason for them to correlate.
- **AVG is not a free win.** On a same-score basis, AVG helps on ~30% of (score, subset) cells (≤ +0.07) and hurts on a comparable fraction (≤ −0.07). The cells where AVG helps cluster in scores with intermediate strand-correlation (e.g. `embed_cosine_lastpos_last`), where each strand adds independent signal. Where one strand strongly dominates (e.g. `minus_llr` on 5'UTR: FWD=0.828 vs RC=0.713), AVG just drags toward the worse strand.
- **No paired test reaches significance after BH-FDR** across 240 cells × 3 comparisons; AVG has a small directional advantage in the discordant-pair means (0.514–0.558).
- **Practical recommendation**: AVG is a safe but **conditional** improvement — when both strands provide independent useful signal (most embedding scores) it can help; when one strand is clearly dominant it can hurt. Skip AVG entirely for `_varpos`/`_lastpos_middle` scores where the two strands are not comparable.

## Code @ [`5a2f4b6`](https://github.com/Open-Athena/bolinas-dna/commit/5a2f4b67865b6a7effb2c860a9a5045976a93b58)

- [`scratch/zeroshot_vep_iter4_rc_scout.py`](https://github.com/Open-Athena/bolinas-dna/blob/5a2f4b6/scratch/zeroshot_vep_iter4_rc_scout.py) — RC forward pass + 30 scores
- [`scratch/zeroshot_vep_iter4_rc_analyze.py`](https://github.com/Open-Athena/bolinas-dna/blob/5a2f4b6/scratch/zeroshot_vep_iter4_rc_analyze.py) — FWD/RC/AVG comparison
- [`scratch/iter4/`](https://github.com/Open-Athena/bolinas-dna/tree/5a2f4b6/scratch/iter4/) — parquets + analysis log
