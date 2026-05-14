🤖 **Q2: among the 7 ensembles, does any beat the standard `mean_rank` baseline?**

Test: `paired_score_comparison` (McNemar-style, two-sided) at the Global level. BH-adjusted q within each dataset (6 tests per dataset). Δ-PA = PA(candidate) − PA(mean_rank). `frac cand wins` is the fraction of *discordant* pairs where the candidate orders correctly and mean_rank does not (halves = 0.5).

### mendelian_traits  (mean_rank Global PA = 0.7588)

| candidate | PA(cand) | Δ vs mean_rank | frac cand wins | n_pairs | n_cand>base | n_base>cand | n_tied | p (two-sided) | q (BH) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `max_rank` | 0.7445 | -0.0143 | 0.3664 | 4910 | 95 | 163 | 4652 | 2.7e-05 | 0.0002 ★ |
| `rrf_k60` | 0.7546 | -0.0042 | 0.3660 | 4910 | 27 | 47 | 4836 | 0.0265 | 0.0796 |
| `zscore_mean` | 0.7613 | +0.0025 | 0.6190 | 4910 | 32 | 18 | 4860 | 0.0649 | 0.1298 |
| `geomean_rank` | 0.7617 | +0.0030 | 0.6090 | 4910 | 39 | 25 | 4846 | 0.1034 | 0.1551 |
| `harmonic_rank` | 0.7617 | +0.0030 | 0.5707 | 4910 | 57 | 43 | 4810 | 0.1933 | 0.2320 |
| `min_rank` | 0.7609 | +0.0021 | 0.5245 | 4910 | 111 | 101 | 4698 | 0.5366 | 0.5366 |

### complex_traits  (mean_rank Global PA = 0.5674)

| candidate | PA(cand) | Δ vs mean_rank | frac cand wins | n_pairs | n_cand>base | n_base>cand | n_tied | p (two-sided) | q (BH) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `min_rank` | 0.5399 | -0.0275 | 0.3960 | 564 | 27 | 44 | 493 | 0.0568 | 0.3409 |
| `harmonic_rank` | 0.5514 | -0.0160 | 0.4196 | 564 | 23 | 32 | 509 | 0.2806 | 0.7905 |
| `geomean_rank` | 0.5567 | -0.0106 | 0.4302 | 564 | 18 | 24 | 522 | 0.4408 | 0.7905 |
| `zscore_mean` | 0.5603 | -0.0071 | 0.4512 | 564 | 18 | 22 | 524 | 0.6358 | 0.7905 |
| `rrf_k60` | 0.5603 | -0.0071 | 0.4574 | 564 | 21 | 25 | 518 | 0.6587 | 0.7905 |
| `max_rank` | 0.5700 | +0.0027 | 0.5090 | 564 | 41 | 40 | 483 | 1.0000 | 1.0000 |

### eqtl  (mean_rank Global PA = 0.5147)

| candidate | PA(cand) | Δ vs mean_rank | frac cand wins | n_pairs | n_cand>base | n_base>cand | n_tied | p (two-sided) | q (BH) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rrf_k60` | 0.5234 | +0.0087 | 0.5431 | 2306 | 125 | 106 | 2075 | 0.2362 | 0.7098 |
| `zscore_mean` | 0.5212 | +0.0065 | 0.5444 | 2306 | 91 | 77 | 2138 | 0.3159 | 0.7098 |
| `max_rank` | 0.5197 | +0.0050 | 0.5164 | 2306 | 180 | 168 | 1958 | 0.5555 | 0.7098 |
| `harmonic_rank` | 0.5095 | -0.0052 | 0.4815 | 2306 | 156 | 167 | 1983 | 0.5780 | 0.7098 |
| `min_rank` | 0.5089 | -0.0059 | 0.4841 | 2306 | 204 | 216 | 1886 | 0.5915 | 0.7098 |
| `geomean_rank` | 0.5137 | -0.0011 | 0.4940 | 2306 | 103 | 104 | 2099 | 1.0000 | 1.0000 |

★ = q < 0.05.