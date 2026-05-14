🤖 **Design discussion — evaluation protocol implications from iter-5 + iter-6.**

## Question 1: is "mendelian → LLR, others → `down_jsd_mean`" a reasonable protocol?

The general framing — **strongly negatively selected → LLR, otherwise → downstream-perturbation score** — has clean mechanistic backing:

- When a variant directly disrupts a coding signal under strong purifying selection, the model's conditional `P(alt | context)` at the variant position carries the dominant signal. LLR is sharper because the "right answer" for the model lives at the variant token itself.
- When the variant doesn't materially affect the maximum-likelihood call at its position (e.g., a TFBS variant that lightly modulates expression without changing what nucleotide "fits" there), the alt allele still perturbs the model's predictions at *nearby* positions through the context it provides. `down_jsd_mean` captures this directly.

**Statistical strength is asymmetric**:

- "LLR for strongly selected" is well-established: mendelian Δ(LLR − `down_jsd_mean`) = +0.019, paired-McNemar q = 6e-4 ★ ([c-4453103460](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4453103460)).
- "`down_jsd_mean` for not-strongly-selected" is only descriptive: complex Δ = +0.034 (q=0.41), eqtl Δ = +0.014 (q=0.71). Direction is consistent across datasets but n_pairs constrains power. A larger eqtl set (test split + cross-cohort) would settle it.

So we *know* LLR wins for coding; we *suspect* `down_jsd_mean` wins for regulatory but haven't shown it rigorously at this dataset size.

## Question 2: cross-CLM comparability

A strong reason to favor logit-space scores over embedding-based ones for *cross-model* evaluation:

| concern | LLR / `down_jsd_mean` | `embed_l2_flat_last` |
|---|---|---|
| pooling choice | n/a | `flat`/`mean`/`varpos`/`lastpos` |
| layer choice | n/a (output head) | `last`/`middle`/... |
| distance metric | n/a (softmax / KL family) | `l2`/`cosine`/`dot` |
| hidden-dim dependence | none (softmax probability) | yes |
| comparable across CLM architectures | yes — just need the 4 nucleotide token IDs | only if pooling/layer/dim are matched |

For training-time evals (rapid checkpoint comparison) or cross-architecture leaderboards, this matters: comparing `embed_l2_flat_last` across two CLMs with different hidden dims or different "best layer" conventions is apples-to-oranges. Comparing LLR + `down_jsd_mean` is apples-to-apples.

## The logit-space-only ensemble: `mean_rank(LLR, down_jsd_mean)`

Tested in [c-4453302231](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4453302231). Global PA across all 4 candidates:

| dataset | LLR | down_jsd_mean | mean_rank(LLR, L2) | **mean_rank(LLR, jsd_mean)** |
|---|---:|---:|---:|---:|
| mendelian | 0.7513 | 0.7320 | **0.7588** | 0.7515 |
| complex | 0.5372 | 0.5709 | 0.5674 | **0.5771** |
| eqtl | 0.5106 | 0.5243 | 0.5147 | **0.5278** |

Paired tests (BH within dataset, 3 each):

- **mendelian**: logit-space ensemble ≈ LLR (q=1.0, Δ=+0.0002); **significantly worse than `mean_rank(LLR, L2)`** by Δ=-0.007 (q=0.024 ★).
- **complex / eqtl**: logit-space ensemble is the descriptive winner in both; all 4 are statistically tied.

**The tradeoff is concrete**: ~0.007 PA on mendelian (in exchange for full cross-model portability + no embedding extraction in scoring). For training-time evals, this seems like a strong trade. For one-shot definitive Mendelian variant scoring, the embedding-based ensemble is still better by ~0.007.

## Recommended protocols

Three plausible options depending on use case:

1. **Per-dataset policy** (current zeroshot recipes, embedding-allowed):
   - mendelian → `mean_rank(LLR, L2)` (or 3-way with AlphaGenome per #175 final recipe)
   - complex / eqtl → `embed_l2_flat_last` (or per-recipe with AlphaGenome)

2. **Selection-aware policy** (strong selection split, logit-space):
   - strongly-selected subsets (coding, splicing) → `LLR`
   - regulatory / weakly-selected subsets → `down_jsd_mean`
   - Requires a prior coding-vs-regulatory split per variant. The 8-subset categorization is a clean operationalization.

3. **Universal logit-space score** (cross-CLM comparability prioritized):
   - everywhere → `mean_rank(LLR, down_jsd_mean)`
   - No prior split needed; portable across CLM architectures; ~0.007 PA cost on mendelian vs option 1.

For matched-pair leaderboards #161 / #162 / #172 (currently single-score per dataset), option 3 looks attractive as a single rule that's competitive everywhere and survives an architecture change without redefining the score column. Option 1 keeps the per-dataset specificity and slightly better mendelian.

## Caveats

- The selection-aware split (option 2) maps "coding" to "strongly selected" — not perfectly true: synonymous variants are also weakly selected; some intronic / regulatory variants in essential genes can be highly selected.
- The complex/eqtl improvements from `down_jsd_mean` (and from the logit-space ensemble) are descriptive only; rigorously establishing them would need more pairs.
- All numbers here are exp166-p1B (1B zoonomia generalist) only. Verifying this protocol on other CLMs (especially smaller specialist models) is open work.
