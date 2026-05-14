🤖 **Q1: paired sign-test — is any of {LLR, L2, ensemble} better than the others?**

Test: `paired_score_comparison` (McNemar-style sign test, two-sided) at the Global level (all match-pairs pooled per dataset). `frac_A_wins` = fraction of pairs where A correctly orders (pos>neg) but B does not, with halves counted at 0.5 (note: A only "wins" a pair when it disagrees with B, so concordant pairs aren't counted). BH-adjusted q within each dataset (3 tests).

For "ensemble" we pick the best ensemble per dataset by Global PairwiseAccuracy:

- **mendelian_traits**: best ensemble = `geomean_rank`
- **complex_traits**: best ensemble = `max_rank`
- **eqtl**: best ensemble = `rrf_k60`

### mendelian_traits

| comparison (A vs B) | PA(A) | PA(B) | frac A wins | n_pairs | n_A>B | n_B>A | n_tied | p (two-sided) | q (BH) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 vs LLR | 0.7424 | 0.7513 | 0.4702 | 4910 | 347 | 391 | 4172 | 0.1134 | 0.1134 |
| best_ens=geomean_rank vs LLR | 0.7617 | 0.7513 | 0.5722 | 4910 | 202 | 151 | 4557 | 0.0077 | 0.0115 ★ |
| best_ens=geomean_rank vs L2 | 0.7617 | 0.7424 | 0.6234 | 4910 | 240 | 145 | 4525 | 1.5e-06 | 4.4e-06 ★ |

### complex_traits

| comparison (A vs B) | PA(A) | PA(B) | frac A wins | n_pairs | n_A>B | n_B>A | n_tied | p (two-sided) | q (BH) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 vs LLR | 0.5674 | 0.5372 | 0.5359 | 564 | 127 | 110 | 327 | 0.2986 | 0.4480 |
| best_ens=max_rank vs LLR | 0.5700 | 0.5372 | 0.5822 | 564 | 65 | 46 | 453 | 0.0871 | 0.2613 |
| best_ens=max_rank vs L2 | 0.5700 | 0.5674 | 0.5060 | 564 | 62 | 61 | 441 | 1.0000 | 1.0000 |

### eqtl

| comparison (A vs B) | PA(A) | PA(B) | frac A wins | n_pairs | n_A>B | n_B>A | n_tied | p (two-sided) | q (BH) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 vs LLR | 0.5225 | 0.5106 | 0.5118 | 2306 | 595 | 568 | 1143 | 0.4458 | 0.6688 |
| best_ens=rrf_k60 vs LLR | 0.5234 | 0.5106 | 0.5265 | 2306 | 293 | 264 | 1749 | 0.2354 | 0.6688 |
| best_ens=rrf_k60 vs L2 | 0.5234 | 0.5225 | 0.5017 | 2306 | 304 | 302 | 1700 | 0.9676 | 0.9676 |

★ = q < 0.05.

**Reading**: `PA(A) > PA(B)` alone is descriptive; the paired test is the rigorous check that the per-pair advantage is non-random. Concordant pairs (both A and B order correctly, or both wrong) carry no signal for the test.