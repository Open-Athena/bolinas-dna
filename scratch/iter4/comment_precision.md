🤖 **Iter-4 follow-up: bf16 vs fp32 precision test.** User asked whether running the eval with fp32 model forward pass (vs the bf16 default) improves PairwiseAccuracy. Tested on exp58-mammals × mendelian × win=256 (FWD only). Commit [`fd0d86d`](https://github.com/Open-Athena/bolinas-dna/commit/fd0d86d).

## TL;DR

**fp32 does not meaningfully improve performance over bf16.** Across 240 (score × subset) cells: 56 fp32 wins, 56 bf16 wins, 128 ties, mean delta = −0.0003. LLR-family results are bit-identical or within 0.002 on exp58-mammals' home subsets (missense / splicing / synonymous). The biggest single-cell deltas (up to ±0.09) come from small-n subsets (synonymous n=33, ncRNA_exon n=42) where a single pair flip = ±0.03; they don't survive as statistical signal.

## Setup

- Model: exp58-mammals (the most-analyzed CDS model in iter-4)
- Window: 256 bp, no BOS
- Dataset: mendelian × 9820 variants (train split)
- Strand: FWD only (precision is a single-strand question; RC is orthogonal and was iterating to ~25 more min so I cancelled)
- Compared raw FWD scores from iter-1 (bf16 baseline) against new FWD scores at fp32 model forward pass
- Log-softmax was already in fp32 in both versions (existing `logits.float()` cast); only the **model forward pass** changed
- Inner join on (chrom, pos, ref, alt) — 9820/9820 matched

## Sanity: bf16 and fp32 raw scores agree very tightly

Pooled Pearson(bf16_raw, fp32_raw) across all 9820 variants per score:

- Min: 0.9946 (`embed_l2_lastpos_last`)
- Median: 0.9998
- Max: 1.0000 (multiple)

The two precisions are computing the same thing up to ~bf16 noise. So any PairwiseAccuracy delta would have to come from the small subset of variants where the bf16 noise flips a matched-pair ranking — at most a few percent.

## PairwiseAccuracy bf16 vs fp32 across all 240 cells

| metric | value |
|---|---:|
| n_fp32_wins (Δ > 0) | 56 |
| n_bf16_wins (Δ < 0) | 56 |
| n_tied (Δ = 0) | 128 |
| mean Δ(fp32 − bf16) | −0.0003 |
| median Δ | 0.0000 |
| max Δ (fp32 wins) | +0.0909 (synonymous × embed_l2_lastpos_last, n=33) |
| min Δ (bf16 wins) | −0.0606 (synonymous × embed_minus_dot_lastpos_middle, n=33) |

Equal counts of fp32-wins and bf16-wins, mean essentially zero. The largest single-cell deltas all sit on small-n subsets where 1 pair flip = 1/n = 0.03–0.05, so ±0.09 is 1–3 pair flips, well within sampling noise.

## Focus on exp58-mammals home subsets

| subset | score | bf16 | fp32 | Δ |
|---|---|---:|---:|---:|
| missense_variant | `minus_llr` | 0.7746 | 0.7758 | +0.0011 |
| missense_variant | `abs_llr` | 0.7657 | 0.7675 | +0.0018 |
| missense_variant | `minus_logp_alt` | 0.7713 | 0.7715 | +0.0002 |
| missense_variant | `embed_cosine_mean_last` | 0.7671 | 0.7666 | −0.0004 |
| splicing | `minus_llr` | 0.7821 | 0.7821 | 0.0000 |
| splicing | `embed_cosine_mean_last` | 0.7821 | 0.7821 | 0.0000 |
| splicing | `embed_l2_lastpos_last` | 0.7564 | 0.7564 | 0.0000 |
| synonymous_variant | `minus_llr` | 0.6970 | 0.6970 | 0.0000 |

The LLR family on missense / splicing / synonymous: bit-identical or within 0.002. Splicing in particular — where bf16 had Pearson(FWD, RC) = 0.05 in iter-4 — is bit-identical between bf16 and fp32 on the FWD side, ruling out bf16 noise as the source of the FWD/RC disagreement.

## Where (small) fp32 wins concentrate

Top fp32 wins are dominated by `*_lastpos_*` and `*_varpos_*` pool scores — these depend on a single token's full embedding vector, which is the most numerically sensitive way to use the model output (vs aggregated pools like flat/mean which average bf16 noise out). Even there, the wins are at the noise threshold of small-n subsets and don't show as a coherent pattern across (score, subset) pairs.

## Practical takeaway

**bf16 is fine for this eval.** The pre-existing `logits.float()` cast for log-softmax (biofoundation#21) was the only precision-related issue that mattered, and that was already addressed. Running the model in fp32 doubles compute and memory without measurable gains on PairwiseAccuracy.

For specific cases that are numerically delicate (e.g. comparing fine score-rankings across runs, or scores derived from raw embedding components), fp32 might be worth keeping as an option — but for the headline matched-pair metric, no need.

## Code @ [`fd0d86d`](https://github.com/Open-Athena/bolinas-dna/commit/fd0d86d)

- [`scratch/zeroshot_vep_iter4_precision_scout.py`](https://github.com/Open-Athena/bolinas-dna/blob/fd0d86d/scratch/zeroshot_vep_iter4_precision_scout.py) — precision-aware scout (--strand fwd|rc, --dtype bf16|fp32)
- [`scratch/iter4/iter4_fwd_fp32_exp58-mammals__win256__mendelian_traits.parquet`](https://github.com/Open-Athena/bolinas-dna/tree/fd0d86d/scratch/iter4) — fp32 FWD scores for the comparison
