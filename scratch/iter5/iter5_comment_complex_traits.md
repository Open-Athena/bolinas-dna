🤖 **exp166-p1B (FWD+RC AVG) — complex_traits.** LLR-family default vs `embed_l2_flat_last` vs 7 ensembles of the two. Global + Macro Avg PairwiseAccuracy (leaderboard convention: Macro Avg = unweighted mean of per-subset PAs over subsets with n_pairs ≥ 30).

## Global + Macro Avg (2 subsets qualify)

| score | Global | Macro Avg (2 subsets, n≥30) |
|---|---:|---:|
| `abs_llr` | 0.5372 ± 0.0210 | 0.6085 ± 0.0331 |
| `embed_l2_flat_last` | 0.5674 ± 0.0209 | 0.6289 ± 0.0326 |
| `mean_rank` | 0.5674 ± 0.0209 | 0.6257 ± 0.0328 |
| `min_rank` | 0.5399 ± 0.0210 | 0.6134 ± 0.0328 |
| `max_rank` | 0.5700 ± 0.0208 | 0.6242 ± 0.0330 |
| `geomean_rank` | 0.5567 ± 0.0209 | 0.6178 ± 0.0330 |
| `harmonic_rank` | 0.5514 ± 0.0209 | 0.6166 ± 0.0330 |
| `rrf_k60` | 0.5603 ± 0.0209 | 0.6254 ± 0.0326 |
| `zscore_mean` | 0.5603 ± 0.0209 | 0.6266 ± 0.0326 |

- Best ensemble (Global): `max_rank` = **0.5700** vs `abs_llr` 0.5372 (Δ +0.0328), vs `embed_l2_flat_last` 0.5674 (Δ +0.0027).
- Best ensemble (Macro Avg): `zscore_mean` = **0.6266** vs `abs_llr` 0.6085 (Δ +0.0181), vs `embed_l2_flat_last` 0.6289 (Δ -0.0023).

## Per-subset PA

| subset | abs_llr | embed_l2_flat_last | mean_rank | min_rank | max_rank | geomean_rank | harmonic_rank | rrf_k60 | zscore_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3_prime_UTR_variant | 0.5238 | 0.5238 | 0.6190 | 0.3810 | 0.6190 | 0.4762 | 0.4286 | 0.6190 | 0.5714 |
| 5_prime_UTR_variant | 0.5000 | 0.6000 | 0.5000 | 0.5000 | 0.6000 | 0.5000 | 0.5000 | 0.5000 | 0.6000 |
| distal | 0.5327 | 0.5561 | 0.5584 | 0.5339 | 0.5643 | 0.5514 | 0.5491 | 0.5491 | 0.5514 |
| missense_variant | 0.6842 | 0.7018 | 0.6930 | 0.6930 | 0.6842 | 0.6842 | 0.6842 | 0.7018 | 0.7018 |
| non_coding_transcript_exon_variant | 0.4167 | 0.5000 | 0.6250 | 0.5000 | 0.4583 | 0.5833 | 0.5000 | 0.5833 | 0.4167 |
| splicing | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| synonymous_variant | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| tss_proximal | 0.4483 | 0.5862 | 0.4828 | 0.5345 | 0.5000 | 0.5172 | 0.5172 | 0.4828 | 0.5172 |
