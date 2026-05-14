🤖 **exp166-p1B (FWD+RC AVG) — mendelian_traits.** LLR-family default vs `embed_l2_flat_last` vs 7 ensembles of the two. Global + Macro Avg PairwiseAccuracy (leaderboard convention: Macro Avg = unweighted mean of per-subset PAs over subsets with n_pairs ≥ 30).

## Global + Macro Avg (8 subsets qualify)

| score | Global | Macro Avg (8 subsets, n≥30) |
|---|---:|---:|
| `minus_llr` | 0.7513 ± 0.0062 | 0.7203 ± 0.0198 |
| `embed_l2_flat_last` | 0.7424 ± 0.0062 | 0.7461 ± 0.0193 |
| `mean_rank` | 0.7588 ± 0.0061 | 0.7579 ± 0.0187 |
| `min_rank` | 0.7609 ± 0.0061 | 0.7534 ± 0.0190 |
| `max_rank` | 0.7445 ± 0.0062 | 0.7538 ± 0.0191 |
| `geomean_rank` | 0.7617 ± 0.0061 | 0.7500 ± 0.0190 |
| `harmonic_rank` | 0.7617 ± 0.0061 | 0.7538 ± 0.0189 |
| `rrf_k60` | 0.7546 ± 0.0061 | 0.7625 ± 0.0185 |
| `zscore_mean` | 0.7613 ± 0.0061 | 0.7587 ± 0.0188 |

- Best ensemble (Global): `geomean_rank` = **0.7617** vs `minus_llr` 0.7513 (Δ +0.0104), vs `embed_l2_flat_last` 0.7424 (Δ +0.0193).
- Best ensemble (Macro Avg): `rrf_k60` = **0.7625** vs `minus_llr` 0.7203 (Δ +0.0422), vs `embed_l2_flat_last` 0.7461 (Δ +0.0164).

## Per-subset PA

| subset | minus_llr | embed_l2_flat_last | mean_rank | min_rank | max_rank | geomean_rank | harmonic_rank | rrf_k60 | zscore_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3_prime_UTR_variant | 0.6552 | 0.7241 | 0.7500 | 0.6897 | 0.6897 | 0.6897 | 0.6897 | 0.7241 | 0.7069 |
| 5_prime_UTR_variant | 0.7011 | 0.6897 | 0.6954 | 0.7241 | 0.7299 | 0.7241 | 0.7241 | 0.7011 | 0.7241 |
| distal | 0.6964 | 0.8214 | 0.7500 | 0.7500 | 0.8036 | 0.7143 | 0.7321 | 0.8036 | 0.7679 |
| missense_variant | 0.7548 | 0.7419 | 0.7594 | 0.7620 | 0.7435 | 0.7633 | 0.7628 | 0.7542 | 0.7620 |
| non_coding_transcript_exon_variant | 0.6429 | 0.7857 | 0.7500 | 0.7381 | 0.7143 | 0.7619 | 0.7619 | 0.7143 | 0.7619 |
| splicing | 0.8077 | 0.8718 | 0.8718 | 0.8590 | 0.8590 | 0.8590 | 0.8718 | 0.8846 | 0.8590 |
| synonymous_variant | 0.8485 | 0.7273 | 0.8636 | 0.8485 | 0.8182 | 0.8485 | 0.8485 | 0.8788 | 0.8485 |
| tss_proximal | 0.6557 | 0.6066 | 0.6230 | 0.6557 | 0.6721 | 0.6393 | 0.6393 | 0.6393 | 0.6393 |
