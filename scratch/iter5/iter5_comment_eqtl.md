🤖 **exp166-p1B (FWD+RC AVG) — eqtl.** LLR-family default vs `embed_l2_flat_last` vs 7 ensembles of the two. Global + Macro Avg PairwiseAccuracy (leaderboard convention: Macro Avg = unweighted mean of per-subset PAs over subsets with n_pairs ≥ 30).

## Global + Macro Avg (6 subsets qualify)

| score | Global | Macro Avg (6 subsets, n≥30) |
|---|---:|---:|
| `abs_llr` | 0.5106 ± 0.0104 | 0.5072 ± 0.0224 |
| `embed_l2_flat_last` | 0.5225 ± 0.0104 | 0.5255 ± 0.0224 |
| `mean_rank` | 0.5147 ± 0.0104 | 0.5283 ± 0.0224 |
| `min_rank` | 0.5089 ± 0.0104 | 0.5138 ± 0.0224 |
| `max_rank` | 0.5197 ± 0.0104 | 0.5215 ± 0.0224 |
| `geomean_rank` | 0.5137 ± 0.0104 | 0.5228 ± 0.0224 |
| `harmonic_rank` | 0.5095 ± 0.0104 | 0.5113 ± 0.0224 |
| `rrf_k60` | 0.5234 ± 0.0104 | 0.5240 ± 0.0224 |
| `zscore_mean` | 0.5212 ± 0.0104 | 0.5232 ± 0.0224 |

- Best ensemble (Global): `rrf_k60` = **0.5234** vs `abs_llr` 0.5106 (Δ +0.0128), vs `embed_l2_flat_last` 0.5225 (Δ +0.0009).
- Best ensemble (Macro Avg): `mean_rank` = **0.5283** vs `abs_llr` 0.5072 (Δ +0.0211), vs `embed_l2_flat_last` 0.5255 (Δ +0.0028).

## Per-subset PA

| subset | abs_llr | embed_l2_flat_last | mean_rank | min_rank | max_rank | geomean_rank | harmonic_rank | rrf_k60 | zscore_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3_prime_UTR_variant | 0.5652 | 0.4870 | 0.5130 | 0.5217 | 0.5652 | 0.5217 | 0.4957 | 0.5478 | 0.5304 |
| 5_prime_UTR_variant | 0.4706 | 0.5490 | 0.5294 | 0.5098 | 0.5098 | 0.5490 | 0.5098 | 0.5098 | 0.5098 |
| distal | 0.5051 | 0.5173 | 0.4979 | 0.4991 | 0.5098 | 0.5024 | 0.5006 | 0.5125 | 0.5101 |
| missense_variant | 0.4667 | 0.5000 | 0.4667 | 0.4500 | 0.4667 | 0.4667 | 0.4667 | 0.4667 | 0.4667 |
| non_coding_transcript_exon_variant | 0.4841 | 0.5605 | 0.5860 | 0.5605 | 0.4904 | 0.5287 | 0.5350 | 0.5223 | 0.5414 |
| splicing | 0.6000 | 0.2000 | 0.4000 | 0.2000 | 0.4000 | 0.3000 | 0.2000 | 0.4000 | 0.2000 |
| synonymous_variant | 0.5172 | 0.6552 | 0.6379 | 0.5862 | 0.6034 | 0.6207 | 0.6207 | 0.6552 | 0.6552 |
| tss_proximal | 0.5519 | 0.5394 | 0.5768 | 0.5415 | 0.5871 | 0.5685 | 0.5602 | 0.5851 | 0.5809 |
