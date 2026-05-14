🤖 **Q (systematic) — is any specific nuc-dep approach systematically better?**

Two paired-McNemar analyses on the iter-6 exp166-p1B FWD+RC AVG nuc-dep scores:

## (a) `_mean` vs `_max` aggregation — paired test per metric per dataset

For each metric (`jsd`/`l1`/`l2`/`linf`), test whether the `_mean` aggregation across positions differs from `_max`. Δ = PA(mean) − PA(max); BH within dataset (4 tests each).

| dataset | metric | mean PA | max PA | Δ | n_pairs | n_mean>max | n_max>mean | p | q |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mendelian_traits | `jsd` | 0.7320 | 0.7165 | +0.0155 | 4910 | 353 | 277 | 0.0028 | 0.0111 ★ |
| mendelian_traits | `l1` | 0.7138 | 0.7173 | -0.0035 | 4910 | 428 | 445 | 0.5882 | 0.8915 |
| mendelian_traits | `l2` | 0.7163 | 0.7151 | +0.0012 | 4910 | 453 | 447 | 0.8676 | 0.8915 |
| mendelian_traits | `linf` | 0.7157 | 0.7167 | -0.0010 | 4910 | 427 | 432 | 0.8915 | 0.8915 |
| complex_traits | `jsd` | 0.5709 | 0.5514 | +0.0195 | 564 | 65 | 54 | 0.3594 | 0.7114 |
| complex_traits | `l1` | 0.5727 | 0.5638 | +0.0089 | 564 | 87 | 82 | 0.7584 | 0.7584 |
| complex_traits | `l2` | 0.5745 | 0.5514 | +0.0230 | 564 | 89 | 76 | 0.3502 | 0.7114 |
| complex_traits | `linf` | 0.5762 | 0.5603 | +0.0160 | 564 | 87 | 78 | 0.5335 | 0.7114 |
| eqtl | `jsd` | 0.5243 | 0.4935 | +0.0308 | 2306 | 326 | 255 | 0.0036 | 0.0146 ★ |
| eqtl | `l1` | 0.5295 | 0.5095 | +0.0199 | 2306 | 398 | 352 | 0.1003 | 0.1337 |
| eqtl | `l2` | 0.5312 | 0.5130 | +0.0182 | 2306 | 404 | 362 | 0.1385 | 0.1385 |
| eqtl | `linf` | 0.5278 | 0.5048 | +0.0230 | 2306 | 393 | 340 | 0.0547 | 0.1094 |

★ = q < 0.05.

**Summary (a)**: across all 12 (metric × dataset) cells, mean wins in 10/12 cells; 2 are significant after BH.

## (b) Each score vs the per-dataset Global-PA-best nuc-dep score

Per-dataset best (by Global PA on FWD+RC AVG):
- **mendelian_traits**: `down_jsd_mean`
- **complex_traits**: `down_linf_mean`
- **eqtl**: `down_l2_mean`

Paired McNemar of each other nuc-dep vs the per-dataset best; BH within dataset (7 tests each). Δ = PA(candidate) − PA(best).

### mendelian_traits  (best = `down_jsd_mean`)

| candidate | PA(cand) | Δ vs best | n_pairs | n_cand>best | n_best>cand | p | q |
|---|---:|---:|---:|---:|---:|---:|---:|
| `down_l1_max` | 0.7173 | -0.0147 | 4910 | 257 | 329 | 0.0033 | 0.0033 ★ |
| `down_linf_max` | 0.7167 | -0.0153 | 4910 | 262 | 337 | 0.0025 | 0.0032 ★ |
| `down_jsd_max` | 0.7165 | -0.0155 | 4910 | 277 | 353 | 0.0028 | 0.0032 ★ |
| `down_l2_mean` | 0.7163 | -0.0157 | 4910 | 139 | 216 | 5.2e-05 | 0.0001 ★ |
| `down_linf_mean` | 0.7157 | -0.0163 | 4910 | 138 | 218 | 2.6e-05 | 9.2e-05 ★ |
| `down_l2_max` | 0.7151 | -0.0169 | 4910 | 275 | 358 | 0.0011 | 0.0019 ★ |
| `down_l1_mean` | 0.7138 | -0.0181 | 4910 | 144 | 233 | 5.3e-06 | 3.7e-05 ★ |

### complex_traits  (best = `down_linf_mean`)

| candidate | PA(cand) | Δ vs best | n_pairs | n_cand>best | n_best>cand | p | q |
|---|---:|---:|---:|---:|---:|---:|---:|
| `down_l2_mean` | 0.5745 | -0.0018 | 564 | 2 | 3 | 1.0000 | 1.0000 |
| `down_l1_mean` | 0.5727 | -0.0035 | 564 | 6 | 8 | 0.7905 | 0.9665 |
| `down_jsd_mean` | 0.5709 | -0.0053 | 564 | 41 | 44 | 0.8284 | 0.9665 |
| `down_l1_max` | 0.5638 | -0.0124 | 564 | 80 | 87 | 0.6426 | 0.9665 |
| `down_linf_max` | 0.5603 | -0.0160 | 564 | 78 | 87 | 0.5335 | 0.9665 |
| `down_jsd_max` | 0.5514 | -0.0248 | 564 | 91 | 105 | 0.3531 | 0.9665 |
| `down_l2_max` | 0.5514 | -0.0248 | 564 | 77 | 91 | 0.3159 | 0.9665 |

### eqtl  (best = `down_l2_mean`)

| candidate | PA(cand) | Δ vs best | n_pairs | n_cand>best | n_best>cand | p | q |
|---|---:|---:|---:|---:|---:|---:|---:|
| `down_l1_mean` | 0.5295 | -0.0017 | 2306 | 17 | 21 | 0.6271 | 0.6271 |
| `down_linf_mean` | 0.5278 | -0.0035 | 2306 | 20 | 28 | 0.3123 | 0.4373 |
| `down_jsd_mean` | 0.5243 | -0.0069 | 2306 | 181 | 197 | 0.4404 | 0.5138 |
| `down_l2_max` | 0.5130 | -0.0182 | 2306 | 362 | 404 | 0.1385 | 0.2423 |
| `down_l1_max` | 0.5095 | -0.0217 | 2306 | 354 | 404 | 0.0750 | 0.1751 |
| `down_linf_max` | 0.5048 | -0.0265 | 2306 | 345 | 406 | 0.0285 | 0.0997 |
| `down_jsd_max` | 0.4935 | -0.0377 | 2306 | 398 | 485 | 0.0038 | 0.0264 ★ |

**Cross-dataset summary**: status per nuc-dep score in each dataset (BEST = per-dataset Global winner; tied = not significantly worse than best at q<0.05; WORSE = significantly worse).

| score | mendelian_traits | complex_traits | eqtl |
|---|---|---|---|
| `down_jsd_mean` | BEST | tied | tied |
| `down_jsd_max` | WORSE | tied | WORSE |
| `down_l1_mean` | WORSE | tied | tied |
| `down_l1_max` | WORSE | tied | tied |
| `down_l2_mean` | WORSE | tied | BEST |
| `down_l2_max` | WORSE | tied | tied |
| `down_linf_mean` | WORSE | BEST | tied |
| `down_linf_max` | WORSE | tied | tied |

A score is **systematically a safe choice** if it's BEST or tied in all 3 datasets — i.e., never significantly worse than the per-dataset winner.