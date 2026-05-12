🤖 **Iteration 4 follow-up — bug-check on RC implementation, expanded to all 4 published models.**

User confirms all 4 models (exp55-mammals, exp58-mammals, exp59-mammals, exp136-proj_v30) were trained with RC augmentation, so the FWD-vs-RC LLR PairwiseAccuracy gap on exp55 5'UTR (0.828 vs 0.713) and the very low joint-LLR FWD↔RC Pearson on exp58 mendelian subsets warranted a bug-check. Ran the RC scout on all 4 models × mendelian and computed joint-LLR Pearson side-by-side. Commit [`9a98b14`](https://github.com/Open-Athena/bolinas-dna/commit/9a98b14d5c51a78f38431b43281194c6bcf32b2c).

## TL;DR

- **No bug in my RC implementation.** Three of four models (exp55, exp59, exp136) cluster at pooled joint-LLR Pearson **0.72–0.84**. A code bug would affect all 4 similarly; the model-specific variance points at training-recipe differences.
- **exp58-mammals is the outlier**: pooled LLR Pearson 0.617, with subset values down to **0.019 on 3'UTR** and **0.053 on splicing** (essentially uncorrelated FWD↔RC LLR). Worth checking exp58's RC-augmentation parameters specifically.
- Splicing-is-most-directional intuition holds best for **exp58** (its weakest subset) and partly for **exp59** (splicing 0.43 vs 3'UTR 0.85). For exp55 and exp136, splicing is on par with other subsets.
- Despite weak joint-LLR Pearson, **exp58 PairwiseAccuracy on home domains is similar between FWD and RC** (missense FWD=0.775 RC=0.776) — the two strands use different cues but each ranks pairs well. So **AVG can aggregate near-orthogonal evidence**: best case exp58 splicing FWD=0.782, RC=0.731, **AVG=0.846 (+0.064)**.
- **AVG-by-default in leaderboard protocols looks justified.** Where joint-LLR Pearson is high (exp55), AVG ≈ best-single-strand; where Pearson is low (exp58), AVG adds genuine independent-signal value. Low downside, real upside.

## Sanity checks (all pass)

- **Tokenizer is character-level** (vocab=6, A/C/G/T + UNK/PAD); no BPE merges → strand-symmetric tokenization.
- **DNA variant indices** match GPN-SS reference: `v_fwd = W // 2`, `v_rc = W - 1 - W // 2`. For even W=256: v_fwd=128, v_rc=127. For odd W=255 (exp136): v_fwd=v_rc=127 (symmetric).
- **Token-space variant pos**: `v_dna + n_prefix`. exp55/58/59 have n_prefix=0, exp136 has n_prefix=1. All paths exercised.
- `rc_seq[v_rc] == complement(ref)` asserts pass for all 9820 × 4 = ~40K variants across the 4 models.
- Allele complement mapping `NUC_TO_IDX[complement(ref/alt)]` for RC, matching GPN-SS.
- FWD label discrimination preserved in RC (pathogenic → more-negative mean LLR on both strands, every subset, every model).
- Quantile-bin monotonicity FWD-LLR → RC-LLR is clean for exp55 (most-neg FWD bin: mean −9.96 / RC mean −9.05; positive FWD bin: 2.45 / RC 1.82).

## Per-position logit was NOT the right test

I initially ran a per-position-logit diagnostic that compared the model's 4-vector P(A,C,G,T at variant pos) under FWD and RC. That was the wrong test for a CLM:

- RC equivariance is a property of the **joint distribution** P(window) = P(rc(window)) ⟹ joint LLR is symmetric.
- **Per-position conditionals are direction-dependent** under autoregressive factorization, even with perfect joint-distribution RC equivariance. FWD conditions on upstream context; RC conditions on RC of downstream context. These are genuinely different conditionals, so low per-position Pearson does NOT indicate a bug.

The right test is **joint LLR Pearson**, since `LLR = log P(window_alt) − log P(window_ref)`. With perfect joint-level RC equivariance this equals exactly across strands. Pearson < 1 measures the equivariance gap.

## Joint-LLR Pearson FWD↔RC across all 4 models

**Pooled across all 9820 mendelian variants:**

| score | exp55-mammals | exp58-mammals | exp59-mammals | exp136-proj_v30 |
|---|---:|---:|---:|---:|
| `llr` / `minus_llr` | **0.840** | **0.617** | 0.723 | 0.802 |
| `minus_logp_alt` | 0.828 | 0.585 | 0.733 | 0.753 |
| `abs_llr` | 0.828 | 0.585 | 0.715 | 0.705 |
| `minus_entropy` | 0.797 | 0.737 | 0.427 | 0.540 |
| `logp_ref` | 0.721 | 0.419 | 0.514 | 0.763 |

**Per-subset LLR Pearson:**

| subset | exp55 | exp58 | exp59 | exp136 |
|---|---:|---:|---:|---:|
| 3_prime_UTR_variant | 0.873 | **0.019** | 0.850 | 0.913 |
| 5_prime_UTR_variant | 0.843 | **0.180** | 0.595 | 0.767 |
| distal | 0.847 | 0.345 | 0.658 | 0.802 |
| missense_variant | 0.825 | 0.615 | 0.646 | 0.786 |
| non_coding_transcript_exon | 0.891 | 0.453 | 0.598 | 0.896 |
| splicing | 0.837 | **0.053** | **0.435** | 0.751 |
| synonymous_variant | 0.712 | 0.309 | 0.542 | 0.848 |
| tss_proximal | 0.767 | 0.656 | **0.410** | 0.685 |

Observations:
- **exp55** and **exp136** are reasonably uniform across subsets (0.71–0.89 and 0.69–0.91 respectively).
- **exp59** has visible subset structure: tss_proximal (its non-home domain) is 0.41 vs 3'UTR (its home) at 0.85. Splicing is also weak at 0.43.
- **exp58** has extreme subset structure: most subsets below 0.5, with 3'UTR and splicing essentially uncorrelated. exp58 also has the weakest pooled Pearson by a wide margin.

The "splicing is the most directional context" pattern shows up strongest in exp58 (0.053) and exp59 (0.435), consistent with splice donor/acceptor signals being inherently strand-aware and the FWD-vs-RC CLM seeing fundamentally different cues. exp55 and exp136 don't show splicing as notably weaker than other subsets — possibly because those models' training data was less splice-heavy.

## PairwiseAccuracy: similar FWD/RC despite low Pearson

Even where joint-LLR Pearson is near zero, PairwiseAccuracy on the home-domain subsets is often similar across FWD and RC:

`minus_llr` PairwiseAccuracy, exp58 specifically:

| subset | n_pairs | FWD | RC | AVG | Δ(AVG − best strand) |
|---|---:|---:|---:|---:|---:|
| missense_variant | 4495 | 0.775 | 0.776 | **0.795** | +0.019 |
| splicing | 78 | 0.782 | 0.731 | **0.846** | **+0.064** |
| synonymous_variant | 33 | 0.697 | 0.667 | 0.697 | 0.000 |

For splicing exp58: FWD and RC give **near-orthogonal evidence** (Pearson 0.053) but **AVG=0.846 beats either strand by ≥0.064**. That's the strongest AVG benefit anywhere in iter-4 and lines up exactly with the "averaging independent signals helps most" intuition.

## Recommendations for downstream leaderboard protocol

1. **Default to AVG of FWD + RC** for likelihood-family scores. Upside is highest when the model has weak joint-LLR symmetry (exp58 splicing +0.064); downside is small when it has strong symmetry (exp55 5'UTR −0.023). Low-risk, real-upside.
2. **Skip AVG for `*_varpos_*` and most `*_lastpos_*` pools** — these refer to different tokens in FWD vs RC token space and AVG over them is meaningless. Listed in iter-4 main comment.
3. **Probe exp58's training recipe** — Pearson 0.617 pooled (vs 0.72–0.84 for the other 3 models), and 0.019/0.053 on 3'UTR/splicing, is much weaker RC equivariance than the others. If RC augmentation was supposed to be on, something about exp58's training may have suppressed it.

## Code @ [`9a98b14`](https://github.com/Open-Athena/bolinas-dna/commit/9a98b14d5c51a78f38431b43281194c6bcf32b2c)

- [`scratch/zeroshot_vep_iter4_rc_scout.py`](https://github.com/Open-Athena/bolinas-dna/blob/9a98b14/scratch/zeroshot_vep_iter4_rc_scout.py) — RC forward pass + 30 scores (works for all 4 models)
- [`scratch/iter4_compare_all_models.py`](https://github.com/Open-Athena/bolinas-dna/blob/9a98b14/scratch/iter4_compare_all_models.py) — joint-LLR Pearson side-by-side
- [`scratch/iter4_bugcheck.py`](https://github.com/Open-Athena/bolinas-dna/blob/9a98b14/scratch/iter4_bugcheck.py) — per-position diagnostic (NOT the right RC-eq test, kept for completeness)
- [`scratch/iter4/iter4_rc_{exp55,exp58,exp59,exp136}*.parquet`](https://github.com/Open-Athena/bolinas-dna/tree/9a98b14/scratch/iter4) — RC scores for all 4 models
