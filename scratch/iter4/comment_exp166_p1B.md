🤖 **Iter-4 follow-up: exp166-p1B (1B generalist) on mendelian.** Ran the same FWD + RC scout on the 1B zoonomia-trained generalist model ([HF: `bolinas-dna/exp166-p1B-step-16398`](https://huggingface.co/bolinas-dna/exp166-p1B-step-16398), 19-layer Qwen3, 255 bp + BOS, trained on whole-genome cross-mammal data). 1B params vs 0.6B for the 4 specialists, so not a like-for-like size comparison — but the across-region uniformity of the generalist is striking. Commit [`9cb4135`](https://github.com/Open-Athena/bolinas-dna/commit/9cb4135).

## TL;DR

- **The 1B generalist is competitive or better than the 0.6B specialists on every subset, and wins outright on 4 of 8.** Most striking wins: synonymous +0.15 over exp58-mammals, 3'UTR +0.09 over exp59-mammals (which was supposed to be the 3'UTR specialist).
- **Closest losses are ≤0.05** on splicing / missense / distal — within a few pair flips on small-n subsets except for missense (n=4495 pairs where 0.04 is real). 5'UTR is the only meaningful gap at +0.13 to exp55-mammals.
- **AVG wins on 3/8 subsets for exp166-p1B** (5'UTR, missense, synonymous) with biggest gain on synonymous (+0.060 vs best single strand). Tied with FWD on splicing.
- **The CDS-boundary effect reproduces in the generalist** — splicing redundancy 0.553 is real but much milder than exp58-mammals (0.954). So this isn't a training-recipe artifact specific to specialists; it's a general property of causal LMs at CDS-vs-non-CDS boundaries.

## 5-model best-mode comparison (`minus_llr` PairwiseAccuracy, max of FWD/RC/AVG per cell)

| subset | n_pairs | exp55-m | exp58-m | exp59-m | exp136 | **exp166-p1B** | winner | margin |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 3_prime_UTR_variant | 58 | 0.586 | 0.621 | 0.638 | 0.500 | **0.724** | **p1B** | — |
| 5_prime_UTR_variant | 87 | **0.828** | 0.586 | 0.563 | 0.540 | 0.701 | exp55 | +0.126 |
| distal | 56 | 0.464 | 0.518 | 0.571 | **0.768** | 0.714 | exp136 | +0.054 |
| missense_variant | 4495 | 0.505 | **0.795** | 0.486 | 0.473 | 0.755 | exp58-m | +0.040 |
| non_coding_transcript_exon | 42 | 0.619 | 0.500 | 0.548 | 0.524 | **0.690** | **p1B** | — |
| splicing | 78 | 0.577 | **0.846** | 0.538 | 0.538 | 0.808 | exp58-m | +0.038 |
| synonymous_variant | 33 | 0.545 | 0.697 | 0.394 | 0.576 | **0.848** | **p1B** | +0.151 |
| tss_proximal | 61 | 0.672 | 0.607 | 0.557 | 0.508 | **0.689** | **p1B** | +0.017 |

"Winner margin" is the gap between the specialist winner and exp166-p1B; positive = specialist wins by that much.

## exp166-p1B per-mode breakdown

| subset | n_pairs | FWD | RC | AVG | best | Δ(AVG − best strand) |
|---|---:|---:|---:|---:|---|---:|
| 3_prime_UTR_variant | 58 | **0.724** | 0.638 | 0.655 | FWD | −0.069 |
| 5_prime_UTR_variant | 87 | 0.678 | 0.678 | **0.701** | AVG | +0.023 |
| distal | 56 | **0.714** | 0.661 | 0.696 | FWD | −0.018 |
| missense_variant | 4495 | 0.741 | 0.733 | **0.755** | AVG | +0.014 |
| non_coding_transcript_exon | 42 | **0.690** | 0.690 | 0.643 | FWD/RC | −0.047 |
| splicing | 78 | **0.808** | 0.705 | **0.808** | FWD/AVG | 0.000 |
| synonymous_variant | 33 | 0.758 | 0.788 | **0.848** | AVG | **+0.060** |
| tss_proximal | 61 | **0.689** | 0.590 | 0.656 | FWD | −0.033 |

AVG is the best mode on 5'UTR / missense / synonymous, tied with FWD on splicing. The biggest win is synonymous (+0.060) where the model gives confident predictions on both strands but with partly-independent signals.

## CDS-boundary effect reproduces in the generalist

Per-subset redundancy and informativeness (std of LLR):

| subset | std FWD | std RC | Pearson | redundancy |
|---|---:|---:|---:|---:|
| 3_prime_UTR_variant | 5.66 | 6.44 | 0.850 | 0.159 |
| 5_prime_UTR_variant | 6.04 | 6.35 | 0.864 | 0.138 |
| distal | 5.53 | 5.04 | 0.900 | 0.105 |
| missense_variant | 8.55 | 8.62 | 0.737 | 0.263 |
| non_coding_transcript_exon | 8.04 | 6.75 | 0.922 | 0.092 |
| **splicing** | **6.96** | **6.97** | **0.447** | **0.553** |
| synonymous_variant | 7.87 | 7.42 | 0.697 | 0.304 |
| tss_proximal | 2.10 | 3.06 | 0.907 | 0.155 |

Splicing redundancy 0.553 is the clear outlier — milder than exp58-mammals (0.954) but still substantially elevated above the 0.10–0.30 baseline of the other subsets. **The CDS-vs-intron boundary effect is a general property of causal-LM autoregressive factorization, not a CDS-specialist quirk.** Both strands give equally large LLR magnitudes on splicing variants (std ≈ 7 both sides) but disagree on per-variant magnitudes — confirming the "one strand sees CDS, the other sees intron" mechanism from earlier in the thread.

The generalist model has visibly weaker boundary effect than the CDS-specialists (exp58 0.954, exp58-animals 0.962) — likely because being trained on the entire genome makes the model more uniformly capable across regions, so neither strand has a dramatic advantage on a given variant.

## Cross-region uniformity

exp166-p1B's PairwiseAccuracy range across the 8 subsets is **0.690–0.848** (Δ=0.158). For comparison:

- exp55-mammals: range 0.464–0.828, Δ=0.364 (best on 5'UTR, worst on distal)
- exp58-mammals: range 0.500–0.846, Δ=0.346 (best on splicing, worst on ncRNA)
- exp59-mammals: range 0.394–0.638, Δ=0.244 (uniformly mediocre)
- exp136-proj_v30: range 0.473–0.768, Δ=0.295 (best on distal, weak elsewhere)
- **exp166-p1B: range 0.690–0.848, Δ=0.158** — much narrower spread, **never below 0.69** on any subset.

The generalist is the only model whose worst-case subset accuracy is above 0.66 (vs ~0.50 for the specialists outside their home domain). This is exactly the trade-off pattern you'd hope for: specialists peak higher on their target region; generalist gives strong, uniform predictions everywhere.

## Practical recommendation

For a leaderboard protocol that runs one model across all subsets (the natural way), **exp166-p1B should be a strong default** — it's never the worst on any subset, and its 4 outright wins suggest its broad training is paying off. Specialist ensembles (best-of-models per subset) would beat it overall, but the simpler "one generalist with AVG-strand-mode" recipe is competitive with much less infrastructure.

The biggest open question is **whether a same-sized (0.6B) generalist would still beat the specialists** — the 1B advantage here might be inflated by the 2× parameter budget. A direct 0.6B-generalist vs 0.6B-specialist comparison would settle it.

## Code @ [`9cb4135`](https://github.com/Open-Athena/bolinas-dna/commit/9cb4135)

- [`scratch/iter4/sky_exp166_1b.yaml`](https://github.com/Open-Athena/bolinas-dna/blob/9cb4135/scratch/iter4/sky_exp166_1b.yaml) — sky launch yaml (HF-direct, win=255+BOS, batch=4)
- [`scratch/zeroshot_vep_iter4_precision_scout.py`](https://github.com/Open-Athena/bolinas-dna/blob/9cb4135/scratch/zeroshot_vep_iter4_precision_scout.py) — strand-aware scout (used at --dtype bf16)
- [`scratch/iter4/iter4_{fwd,rc}_exp166-p1B__win255__mendelian_traits.parquet`](https://github.com/Open-Athena/bolinas-dna/tree/9cb4135/scratch/iter4) — scores
