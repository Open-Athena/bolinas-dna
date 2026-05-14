🤖 **Iter-5 summary — exp166-p1B FWD+RC AVG, 3 datasets × 9 scores.**

Columns: Global / Macro Avg PairwiseAccuracy per dataset (canonical leaderboard convention: Macro Avg over subsets with n_pairs ≥ 30). **Bold** marks the best ensemble per (dataset, metric) cell.

| score | mendelian_traits Global | mendelian_traits Macro Avg (k=8) | complex_traits Global | complex_traits Macro Avg (k=2) | eqtl Global | eqtl Macro Avg (k=6) |
|---|---:|---:|---:|---:|---:|---:|
| `minus_llr` / `abs_llr`* | 0.7513 | 0.7203 | 0.5372 | 0.6085 | 0.5106 | 0.5072 |
| `embed_l2_flat_last` | 0.7424 | 0.7461 | 0.5674 | 0.6289 | 0.5225 | 0.5255 |
| `mean_rank` | 0.7588 | 0.7579 | 0.5674 | 0.6257 | 0.5147 | **0.5283** |
| `min_rank` | 0.7609 | 0.7534 | 0.5399 | 0.6134 | 0.5089 | 0.5138 |
| `max_rank` | 0.7445 | 0.7538 | **0.5700** | 0.6242 | 0.5197 | 0.5215 |
| `geomean_rank` | **0.7617** | 0.7500 | 0.5567 | 0.6178 | 0.5137 | 0.5228 |
| `harmonic_rank` | 0.7617 | 0.7538 | 0.5514 | 0.6166 | 0.5095 | 0.5113 |
| `rrf_k60` | 0.7546 | **0.7625** | 0.5603 | 0.6254 | **0.5234** | 0.5240 |
| `zscore_mean` | 0.7613 | 0.7587 | 0.5603 | **0.6266** | 0.5212 | 0.5232 |

*`minus_llr` for mendelian; `abs_llr` for complex_traits & eqtl.*

**Conventions**:
- **Global** = PA across ALL match-pairs concatenated (no n filter). **Macro Avg** = unweighted mean of per-subset PAs over subsets with n_pairs ≥ 30 — matches `bolinas.evals.metrics.compute_pairwise_metrics` on main (PR #178).
- Ranks computed within `(dataset, subset)` (matched pairs are subset-local).
- `mean_rank` ≡ iter-2's `rk_minus_llr_plus_l2flat_last` formula (sum is monotone-equivalent to mean).
- `rrf_k60` uses rank-from-top (`n+1-r`) so standard "rank 1 = best" RRF semantics apply; PA direction reproduced via `PA(x).value + PA(-x).value == 1` cross-check.