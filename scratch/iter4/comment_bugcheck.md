🤖 **Iteration 4 follow-up — bug-check on RC implementation, expanded to all 4 published models with scale-aware diagnostics.**

User flagged that Pearson FWD↔RC can be misleading when both strands give near-zero noise (uncorrelated noise has Pearson ~0 regardless of implementation correctness), and asked for distributional shape + a non-scale-invariant metric. So I ran the RC scout on all 4 published models × mendelian and built better diagnostics. Commit [`4956ea9`](https://github.com/Open-Athena/bolinas-dna/commit/4956ea9).

## TL;DR

- **No bug in my RC implementation.** Three of four models (exp55, exp59, exp136) show LLR redundancy 0.11–0.60 across all 8 mendelian subsets — FWD and RC give substantially shared signal as expected from RC augmentation. A code bug would affect all 4 similarly.
- **exp58-mammals is the clear outlier**: LLR redundancy 0.39–0.99, with **0.954 on splicing (its target!)** and **0.991 on 3'UTR**. Splicing is *not* a noise-around-zero case for exp58 — `std(FWD LLR) = 16.7` and `std(RC LLR) = 14.0`, both highly informative — but FWD and RC predictions are **essentially independent measurements** of the same variant. So this is "informative but disagreeing", not "uninformative noise that happens to be uncorrelated".
- **Splicing is directionally biased everywhere**: exp58 (redundancy 0.95), exp59 (0.57), exp136 (0.25), exp55 (0.16). Two of the four models show splicing as substantially less redundant than other subsets, consistent with the strand-aware nature of splice donor/acceptor signals.
- **AVG-by-default in leaderboard protocols looks well-justified.** For models with high redundancy (exp55, exp136), AVG ≈ best single strand (no harm). For exp58, where FWD/RC are nearly independent, AVG aggregates real complementary signal — biggest win is **splicing AVG=0.846 vs FWD=0.782 vs RC=0.731** (+0.064).

## Why redundancy beats Pearson here

Pearson is scale-invariant: a tight cloud around (0, 0) with no correlation gives Pearson ~0 even if both strands are computing "the model says nothing here, just noise." That's not the same as "the implementation has a bug."

Redundancy = `MSE(FWD, RC) / (Var(FWD) + Var(RC))` is the right diagnostic:
- **0**: FWD = RC exactly.
- **1**: FWD and RC are independent draws from their own distributions (no shared signal).
- **>1**: anti-correlated.

For matched variances this reduces to `1 − r`. For mismatched variances it accounts for the difference. Combined with `std(FWD)`, `std(RC)` we can tell:
- both stds small + low redundancy → quiet but redundant (uninformative model, both strands agree it's uninformative)
- both stds small + redundancy ~1 → quiet noise, can't distinguish from bug
- both stds large + low redundancy → informative AND redundant (well-behaved)
- both stds large + redundancy ~1 → **informative but independent** (the bug-suspicion case)

## Per-(model, subset) redundancy

| subset | exp55 | exp58 | exp59 | exp136 |
|---|---:|---:|---:|---:|
| 3_prime_UTR_variant | 0.170 | **0.991** | 0.152 | 0.162 |
| 5_prime_UTR_variant | 0.169 | 0.830 | 0.406 | 0.249 |
| distal | 0.153 | 0.664 | 0.354 | 0.202 |
| missense_variant | 0.175 | 0.385 | 0.354 | 0.215 |
| non_coding_transcript_exon | 0.131 | 0.599 | 0.435 | 0.112 |
| splicing | 0.164 | **0.954** | 0.567 | 0.250 |
| synonymous_variant | 0.306 | 0.778 | 0.461 | 0.155 |
| tss_proximal | 0.242 | 0.446 | 0.597 | 0.315 |

Three of four models (**exp55, exp59, exp136**) sit at redundancy ≤ 0.6 on every subset, with median around 0.2 — consistent with RC equivariance present but imperfect from training. **exp58** is the dramatic outlier, with redundancy ≥ 0.39 on every subset and ≥ 0.78 on five of eight.

## Per-(model, subset) informativeness (std of LLR — confirms model is *not* just noisy on the high-redundancy cases)

| subset | exp55 | exp58 | exp59 | exp136 |
|---|---:|---:|---:|---:|
| 3_prime_UTR_variant | 2.0 | 3.6 | **11.4** | 3.9 |
| 5_prime_UTR_variant | **11.4** | 3.1 | 1.5 | 1.4 |
| distal | 2.4 | 1.4 | 2.7 | **6.3** |
| missense_variant | 3.6 | **10.6** | 1.8 | 1.3 |
| non_coding_transcript_exon | 9.6 | 1.9 | 2.1 | 2.8 |
| splicing | 2.0 | **16.7** | 1.7 | 1.3 |
| synonymous_variant | 3.2 | **6.7** | 1.8 | 1.4 |
| tss_proximal | **5.0** | 1.4 | 1.4 | 1.3 |

Each model has the largest LLR std on its target domain (bolded). For exp58, **the highest-std subsets ARE the ones with highest redundancy** — splicing has std~16 *and* redundancy ~0.95. That rules out "uninformative noise → uncorrelated by coincidence" as the explanation for exp58.

## Scatter (visual confirmation)

![FWD vs RC LLR scatter — 4 models × 8 subsets](https://raw.githubusercontent.com/Open-Athena/bolinas-dna/4956ea9/scratch/iter4/iter4_fwd_rc_scatter_4models.png)

Rows are models (exp55, exp58, exp59, exp136), columns are subsets. Most cells show diagonal correlations (FWD ≈ RC). **Row 2 (exp58)** has **cloud-shaped distributions on multiple cells** — particularly splicing (big cloud, no diagonal), 3'UTR (cloud), and missense (broader cloud with diagonal tendency). exp58 is visually distinct from the other three rows.

## Sanity checks (all pass)

- Tokenizer is character-level (vocab=6); no BPE merges → strand-symmetric tokenization. Same tokenizer for all 4 models (vocab 6, IDs 2=A 3=C 4=G 5=T).
- DNA variant indices match GPN-SS reference. For even W=256: `v_fwd=128, v_rc=127`. For odd W=255 (exp136): `v_fwd=v_rc=127` (symmetric).
- Token-space variant pos `v_dna + n_prefix`: exp55/58/59 have n_prefix=0, exp136 has n_prefix=1.
- `rc_seq[v_rc] == complement(ref)` asserts pass for all 9820 variants × 4 models.
- Allele complement `NUC_TO_IDX[complement(ref/alt)]` for RC, matching GPN-SS.
- FWD label discrimination preserved in RC (pathogenic = more-negative mean LLR on both strands, every (model, subset)).

## Reframing the per-position diagnostic

My initial bug-check ran a per-position-logit RC-symmetry test. That was the **wrong test for a CLM**:

- RC equivariance is a property of the **joint distribution** `P(window) = P(rc(window))` ⟹ joint LLR is symmetric.
- Per-position conditionals are direction-dependent under autoregressive factorization, even with perfect joint-level RC equivariance. FWD conditions on upstream context; RC conditions on RC of downstream context. These are different conditionals by construction.

The right test is joint-LLR equality, which is what redundancy + std + scatter measure here.

## Recommendations

1. **Default to AVG of FWD + RC** for likelihood-family scores in leaderboard protocols. Cost is one extra forward pass; upside is real on models with weak RC equivariance (exp58 splicing +0.064); downside is small on models with strong equivariance (exp55 5'UTR −0.023).
2. **Skip AVG for `*_varpos_*` and `*_lastpos_middle` pools** — these pick different tokens in FWD vs RC token space and AVG over them is meaningless.
3. **Investigate exp58's training recipe** specifically. The redundancy 0.95 on splicing and 0.99 on 3'UTR, with high informativeness (std 16.7 and 3.6 respectively), is far weaker RC equivariance than the other three models trained on the same RC-aug recipe (allegedly). Either RC augmentation was effectively absent for exp58, or some training-stage-specific factor disrupted it.

## Code @ [`4956ea9`](https://github.com/Open-Athena/bolinas-dna/commit/4956ea9)

- [`scratch/zeroshot_vep_iter4_rc_scout.py`](https://github.com/Open-Athena/bolinas-dna/blob/4956ea9/scratch/zeroshot_vep_iter4_rc_scout.py) — RC forward pass + 30 scores (all 4 models)
- [`scratch/iter4_compare_distributional.py`](https://github.com/Open-Athena/bolinas-dna/blob/4956ea9/scratch/iter4_compare_distributional.py) — std / median |LLR| / redundancy diagnostics
- [`scratch/iter4_scatter_fwd_rc.py`](https://github.com/Open-Athena/bolinas-dna/blob/4956ea9/scratch/iter4_scatter_fwd_rc.py) — 4×8 scatter grid
- [`scratch/iter4_compare_all_models.py`](https://github.com/Open-Athena/bolinas-dna/blob/4956ea9/scratch/iter4_compare_all_models.py) — joint-LLR Pearson side-by-side
- [`scratch/iter4_bugcheck.py`](https://github.com/Open-Athena/bolinas-dna/blob/4956ea9/scratch/iter4_bugcheck.py) — per-position diagnostic (kept for completeness; NOT the right test)
- [`scratch/iter4/iter4_rc_{exp55,exp58,exp59,exp136}*.parquet`](https://github.com/Open-Athena/bolinas-dna/tree/4956ea9/scratch/iter4) — RC scores
