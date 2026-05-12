🤖 **Iteration 4 follow-up — bug-check on RC implementation.** Verifying whether the FWD-vs-RC LLR asymmetry on 5'UTR (exp55-mammals: FWD=0.828 vs RC=0.713) reflects a bug in my RC code or genuine model behaviour. Commit [`2101169`](https://github.com/Open-Athena/bolinas-dna/commit/2101169cf0147603fcdb3f13195506b3c192fe79).

## Methodology

Two complementary diagnostics:

**(a) Per-position logit RC-symmetry check** (`scratch/iter4_bugcheck.py`): for each variant, run one FWD forward pass and one RC pass and extract the logit at the variant position (the prediction *for* the variant token, given the model's left context). Softmax over A/C/G/T, complement-permute the RC vector, and compare the resulting 4-vectors.

→ **This was the wrong test** — the variant-position logit is a *per-position conditional* under one specific autoregressive factorization. Even with perfect joint-distribution RC equivariance, per-position conditionals are direction-dependent (FWD uses upstream context, RC uses RC of downstream context). So low per-position Pearson does NOT indicate a bug.

**(b) Joint-LLR FWD↔RC Pearson** (the right RC-equivariance test for a CLM): if `P(window) = P(rc(window))` then `LLR_FWD = LLR_RC` exactly. Pearson < 1 measures the equivariance gap.

I also ran the full RC scout for **exp58-mammals** on mendelian (~3 min A10G), so I have joint-LLR for both models side-by-side.

## Sanity checks passed

- Tokenizer is **character-level** (vocab size 6 = A/C/G/T + UNK/PAD); no BPE merges, so tokenization is strand-symmetric by construction.
- DNA variant index in RC space: `v_rc = W - 1 - (W // 2)` — matches GPN-SS's `pos_rev = pos_fwd - 1 if W % 2 == 0 else pos_fwd` reference.
- `rc_seq[v_rc] == complement(ref)` asserts pass for all 9820 variants.
- Allele complement: `NUC_TO_IDX[complement(ref/alt)]` for RC, matching GPN-SS.
- FWD label discrimination preserved in RC: pathogenic (label=1) has more-negative mean LLR on **both strands** for every subset.

## Finding 1: Joint-LLR FWD↔RC Pearson varies strongly between models

Per-subset Pearson(llr_fwd, llr_rc), exp55 vs exp58:

| subset | n | exp55 LLR Pearson | exp58 LLR Pearson |
|---|---:|---:|---:|
| 5_prime_UTR_variant | 174 | 0.843 | **0.180** |
| 3_prime_UTR_variant | 116 | 0.873 | **0.019** |
| missense_variant | 8990 | 0.825 | 0.615 |
| splicing | 156 | 0.837 | **0.053** |
| synonymous_variant | 66 | 0.712 | 0.309 |
| distal | 112 | 0.847 | 0.345 |
| tss_proximal | 122 | 0.767 | 0.656 |
| non_coding_transcript_exon | 84 | 0.891 | 0.453 |

Pooled across all 9820 variants:
- exp55-mammals: `llr` Pearson = **0.840**
- exp58-mammals: `llr` Pearson = **0.617**

If both models were perfectly RC-equivariant at the joint distribution level, both would be 1.0. exp55 is markedly more RC-equivariant than exp58. **This suggests the RC augmentation behavior differs between the two models** (or wasn't applied to exp58 in the way expected) — worth checking the training recipes.

## Finding 2: Despite weak joint-LLR Pearson, FWD/RC PairwiseAccuracy is similar on home domains

The seemingly-paradoxical detail: exp58's joint-LLR FWD↔RC Pearson is ~0.05 on splicing, yet both strands give similar PairwiseAccuracy. The two strands use **different cues** but each independently ranks pairs well.

`minus_llr` PairwiseAccuracy by mode, exp55 vs exp58 on relevant subsets:

| subset | n_pairs | model | FWD | RC | AVG |
|---|---:|---|---:|---:|---:|
| 5_prime_UTR_variant | 87 | exp55-mammals | **0.828** | 0.713 | 0.805 |
| 5_prime_UTR_variant | 87 | exp58-mammals | 0.471 | 0.586 | 0.529 |
| missense_variant | 4495 | exp55-mammals | 0.500 | 0.505 | 0.505 |
| missense_variant | 4495 | exp58-mammals | 0.775 | 0.776 | **0.795** |
| splicing | 78 | exp55-mammals | 0.564 | 0.577 | 0.526 |
| splicing | 78 | exp58-mammals | 0.782 | 0.731 | **0.846** |
| synonymous_variant | 33 | exp55-mammals | 0.545 | 0.485 | 0.485 |
| synonymous_variant | 33 | exp58-mammals | 0.697 | 0.667 | 0.697 |
| tss_proximal | 61 | exp55-mammals | 0.672 | 0.639 | 0.656 |
| tss_proximal | 61 | exp58-mammals | 0.607 | 0.525 | 0.590 |

The really interesting cell: **exp58 splicing — AVG=0.846 vs FWD=0.782 vs RC=0.731 (Δ +0.064 over best strand)**. This is the strongest AVG benefit observed anywhere in iter-4. Because FWD and RC give near-orthogonal evidence (Pearson 0.05), averaging them aggregates two largely-independent informative signals. Where FWD and RC are highly correlated (exp55 5'UTR Pearson 0.84), AVG just drags the best toward the worst.

## Reframed conclusion

- **No bug in my RC implementation.** Multiple sanity checks (tokenization, variant position, complement-allele mapping, label-direction preservation, quantile-bin monotonicity) all pass on both models.
- **The CLM's joint distribution is imperfectly RC-equivariant.** Pooled LLR Pearson FWD↔RC is 0.84 (exp55) and 0.62 (exp58). Neither hits 1.0, so neither is strongly RC-equivariant at the joint level.
- **exp58 is much less RC-equivariant than exp55.** Worth checking training recipes — if both were supposed to have RC augmentation, exp58's near-zero Pearson on 3'UTR / splicing is unexpected.
- **AVG's value is highest where FWD and RC give independent evidence.** exp58 splicing (Pearson 0.05) → AVG +0.064 over best strand. exp55 5'UTR (Pearson 0.84) → AVG underperforms FWD by 0.023.
- **Per-position logit symmetry is not a meaningful RC-equivariance test for a CLM.** The right test is joint-distribution equivariance, which equals joint-LLR equality. My bug-check (a) was directionally informative but conceptually a probe of the wrong quantity; the joint-LLR comparison (b) is what should be reported.

## Code @ [`2101169`](https://github.com/Open-Athena/bolinas-dna/commit/2101169cf0147603fcdb3f13195506b3c192fe79)

- [`scratch/iter4_bugcheck.py`](https://github.com/Open-Athena/bolinas-dna/blob/2101169/scratch/iter4_bugcheck.py) — per-position logit diagnostic (don't read too much into; per-position is not the right RC-eq test)
- [`scratch/iter4_compare_exp58.py`](https://github.com/Open-Athena/bolinas-dna/blob/2101169/scratch/iter4_compare_exp58.py) — joint-LLR Pearson exp55 vs exp58
- [`scratch/iter4/iter4_rc_exp58-mammals__win256__mendelian_traits.parquet`](https://github.com/Open-Athena/bolinas-dna/blob/2101169/scratch/iter4/iter4_rc_exp58-mammals__win256__mendelian_traits.parquet) — exp58 RC scores
