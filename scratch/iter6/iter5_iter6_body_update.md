## Question

Which zero-shot scoring rules — and which window sizes — make the bolinas gLMs most informative for variant-effect prediction across mendelian, complex-trait, and eQTL settings? The matched-pair leaderboards in #161 / #162 / #172 are frozen on a single fixed scoring rule per dataset (`minus_llr` for mendelian, `abs_llr` for complex / eqtl, via `biofoundation.inference.run_llr_and_embedding_distance`). We want **insights**, not a single winner — different scores plausibly favor different (model, dataset, consequence-subset) combos.

## Final conclusions

**Status: closed.** Investigation complete across 6 iterations and ~30 GitHub comments. Headline conclusions:

### 1. Per-dataset best recipe (zero-shot, no learned weights)

| dataset | best zero-shot recipe | macro PA | uplift vs default |
|---|---|---:|---:|
| **mendelian** | `rank-mean(minus_llr, embed_l2_flat_last, alphagenome_max_l2)` — 3-way of exp166-p1B's LLR + embedding distance + AlphaGenome | **0.782** | +0.057 over `minus_llr` |
| **complex_traits** | `rank-mean(embed_l2_flat_last, alphagenome_max_l2)` — 2-way of exp166-p1B's embedding distance + AlphaGenome | **0.618** | +0.103 over `abs_llr` |
| **eqtl** | `alphagenome_max_l2` alone | **0.706** | +0.171 over `abs_llr` |

The right policy is **per-dataset** — there's no universal recipe across all 3. Regulatory-heavy datasets (complex, eqtl) want AlphaGenome dominance; coding-heavy mendelian wants a multi-source ensemble.

### 2. RC-strand averaging is justified by default

Across the 5 published gLMs tested (exp55/exp58-mammals/exp58-animals/exp59/exp136), averaging FWD + RC scores beats single-strand on most home-domain (model, subset) cells. Biggest gain: **exp58-mammals splicing AVG=0.846 vs FWD=0.782** (+0.064). Skip AVG for `*_varpos_*` and `*_lastpos_middle` pools — different tokens between strands.

Filed implementation issues:
- **biofoundation#24** — offline batched VEP path (merged via [PR #26](https://github.com/Open-Athena/biofoundation/pull/26), commit `834dd4c`).
- **#179** — online lm_eval path (LLR-only, drives training-time eval + leaderboards #161/#162/#172).

### 3. exp58-mammals had a non-bug-driven RC asymmetry, explained by the CDS-vs-flanking boundary effect

Investigation triggered by FWD ≠ RC LLR on exp58 splicing (joint Pearson = 0.05). Confirmed via 4-model + redundancy analysis: a real biological property of CLM autoregressive factorization at CDS-vs-non-CDS boundaries, not an implementation bug. Reproduces in exp58-animals (splicing redundancy 0.962 vs 0.954 in exp58-mammals). The 1B generalist exp166-p1B shows a much milder boundary effect (splicing redundancy 0.553), suggesting whole-genome training partly normalizes the asymmetry.

### 4. The 1B generalist exp166-p1B is competitive with 0.6B specialists

Trained on `zoonomia-v1-v1` (108-species whole-genome). On the 23-cell (3 datasets × 8 subsets) grid:
- Outright wins on **8 cells** (more than any specialist; exp55/58/136 each get 5–6).
- Wins macro on mendelian (0.776 best-individual) and eqtl (0.664).
- Cross-region uniformity: range 0.69–0.85 across mendelian subsets (vs ≤0.40 floor for specialists outside their home domain).
- Embedding scores (`embed_l2_flat_last`) frequently match or beat LLR for the 1B model — richer learned representations.

### 5. bf16 model forward pass is fine; fp32 doesn't help

Tested fp32 model forward pass on exp58-mammals × mendelian: across 240 (score × subset) cells, 56 fp32 wins / 56 bf16 wins / 128 ties; mean Δ −0.0003. LLR-family results are bit-identical or within 0.002 on home subsets. The pre-existing `logits.float()` cast for log-softmax (biofoundation#21) was the only meaningful precision concern.

### 6. Score-class effects

- **Pool**: `flat` (full window) and `mean` dominate; `varpos` and `lastpos` pools are noisier and strand-asymmetric.
- **Distance**: `cosine` ≈ `l2` > `dot` (raw inner product) overall.
- **Layer**: last ≈ middle, with small directional edge to `last`.
- **Window**: native (255/256 bp) is competitive; longer windows (512) help exp58 CDS models modestly.

---

## Iter-5 / iter-6 follow-ups (added after reopening 2026-05-14)

### 7. Rigor on exp166-p1B FWD+RC AVG with leaderboard-aligned metrics

Re-ran exp166-p1B on all 3 datasets with **proper FWD+RC averaging** and the canonical leaderboard convention from `bolinas.evals.metrics.compute_pairwise_metrics` on main (PR #178): **Global = PA across all match-pairs concatenated; Macro Avg = unweighted mean of per-subset PAs over subsets with n_pairs ≥ 30** ([c-4452344147](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452344147)).

Headline Global / Macro Avg PA:

| dataset | k | LLR (default) | `embed_l2_flat_last` | best ensemble Global | best ensemble Macro |
|---|---:|---|---|---|---|
| mendelian | 8 | 0.7513 / 0.7203 | 0.7424 / 0.7461 | `geomean_rank` 0.7617 | `rrf_k60` 0.7625 |
| complex | 2 | 0.5372 / 0.6085 | 0.5674 / 0.6289 | `max_rank` 0.5700 | `zscore_mean` 0.6266 |
| eqtl | 6 | 0.5106 / 0.5072 | 0.5225 / 0.5255 | `rrf_k60` 0.5234 | `mean_rank` 0.5283 |

Sanity-checked against iter-4: mendelian `minus_llr` AVG per-subset PA reproduces iter-4 comment [`-4435271389`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4435271389) bit-exact (e.g., synonymous 0.8485, splicing 0.8077, missense 0.7548).

### 8. Paired McNemar significance testing on the iter-5 candidates

Two question families, both at Global level with two-sided paired sign-test ([c-4452418017](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452418017), [c-4452418485](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452418485)):

- **Q1 (LLR vs L2 vs best ensemble per dataset)**: ensembling significantly beats both base scores on **mendelian** (`geomean_rank` vs LLR q=0.012, vs L2 q=4e-6). Complex / eqtl show no significant differences after BH — under-powered (complex n_pairs=564, eqtl=2306) despite descriptive gaps of +0.030–+0.034 over `abs_llr`.
- **Q2 (ensembles vs `mean_rank` baseline)**: nothing significantly beats `mean_rank` on any dataset. On mendelian, **`max_rank` significantly UNDERPERFORMS** `mean_rank` (Δ=-0.014, q=2e-4), and `rrf_k60` is marginally worse. `mean_rank` is the safe default among the rank-based ensembles.

Per-subset paired tests ([c-4452475796](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452475796), [c-4452478662](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452478662)) localize the action: on mendelian, `missense` (n_pairs=4495) drives the global ensemble win; eqtl ncRNA (n=157) shows `max_rank` significantly underperforms `mean_rank` (q=0.033). Other subsets are descriptively interesting but don't survive multiple-testing correction.

### 9. Nucleotide-dependency ("nuc-dep") downstream-effect scores — exp166-p1B FWD+RC AVG

Implemented downstream-effect scoring with strand control for exp166-p1B ([c-4452713609](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452713609)). 8 candidate scores (`down_{jsd,l1,l2,linf}_{mean,max}`) — per-position 4-nuc softmax divergence between REF-context and ALT-context, aggregated downstream. **FWD captures the genomic-downstream half of the variant's effect footprint; RC captures the upstream half** (the AR mask runs in token order, which reverses under RC); AVG combines both → bidirectional nuc-dep despite the unidirectional AR model.

Key findings:

- **nuc-dep is structurally a logit-space `embed_l2_flat_last`**: Spearman ρ(`down_jsd_mean`, L2) within-subset is **0.90 / 0.84 / 0.78** across mendelian / complex / eqtl — much higher than ρ(`down_jsd_mean`, LLR) of 0.61 / 0.51 / 0.38.
- On **regulatory datasets** (complex, eqtl) `down_jsd_mean` descriptively beats both LLR and L2 (Δ+0.034 over LLR on complex; +0.014 on eqtl) — but not significantly after BH (n_pairs-limited).
- On **mendelian**, `down_jsd_mean` significantly *underperforms* both LLR (Δ=-0.019, q=8e-4) and L2 (Δ=-0.010, q=0.008): when the variant directly disrupts coding signal, LLR at the variant position is sharper than its downstream echo.
- **`down_jsd_mean` is the systematically safe single nuc-dep score** ([c-4453058670](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4453058670)): it's BEST or statistically tied with the best in all 3 datasets. Every other nuc-dep score is significantly worse than `down_jsd_mean` on mendelian. `_mean` aggregations beat `_max` everywhere descriptively (10/12 cells, 2 significant after BH).

### 10. Logit-space ensemble `mean_rank(LLR, down_jsd_mean)` and protocol recommendations

The natural follow-up: a rank-mean ensemble that stays entirely in logit space ([c-4453302231](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4453302231)). Global PA:

| dataset | LLR | `down_jsd_mean` | `mean_rank(LLR, L2)` | **`mean_rank(LLR, jsd_mean)`** |
|---|---:|---:|---:|---:|
| mendelian | 0.7513 | 0.7320 | **0.7588** | 0.7515 |
| complex | 0.5372 | 0.5709 | 0.5674 | **0.5771** |
| eqtl | 0.5106 | 0.5243 | 0.5147 | **0.5278** |

The logit-space ensemble is the descriptive winner on complex + eqtl and matches LLR on mendelian. It only loses meaningfully to `mean_rank(LLR, L2)` on mendelian by Δ=-0.007 (q=0.024 ★).

**Three protocol options**, design discussion in [c-4453331086](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4453331086):

1. **Per-dataset policy** (current best, allows embeddings): per-dataset recipes from conclusion #1 above — strongest absolute performance.
2. **Selection-aware split** (logit-space): strongly-selected subsets (coding, splicing) → `LLR`; regulatory / weakly-selected → `down_jsd_mean`. Statistically grounded on the LLR leg; only descriptive on the regulatory leg.
3. **Universal logit-space score** (cross-CLM comparability prioritized): `mean_rank(LLR, down_jsd_mean)` everywhere. Costs ~0.007 PA on mendelian vs option 1; eliminates pooling/layer/distance choices and is portable across CLM architectures.

For training-time evals (many checkpoints, cross-architecture leaderboards), option 3 is attractive: a single score that's competitive on all 3 datasets and survives architecture changes without redefining scoring.

## Comment index

| iter | topic | comment |
|---|---|---|
| 1 | Pipeline + 30 baseline scores × 5 models × 3 datasets × 3 windows | [`-4431776701`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4431776701) |
| 2 | Rank-based composites; eqtl null discussion | [`-4431868008`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4431868008) |
| 3 | Downstream-effect AR scout (exp55-mammals) | [`-4432893316`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4432893316) |
| 4 | RC strand exploration (exp55-mammals) | [`-4433397209`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4433397209) |
| 4-bug | Bug-check: implementation correct, exp58 outlier | [`-4434022140`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4434022140) |
| 4-boundary | CDS-vs-flanking boundary effect confirmed | [`-4434107699`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4434107699) |
| 4-exp58-anim | exp58-animals reproduces splicing boundary | [`-4434639842`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4434639842) |
| 4-precision | bf16 vs fp32 (no gain) | [`-4434884279`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4434884279) |
| 4-1B-mendelian | exp166-p1B 1B generalist vs specialists on mendelian | [`-4435271389`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4435271389) |
| 4-1B-3datasets | exp166-p1B FWD on all 3 datasets × all 30 scores | [`-4435512869`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4435512869) |
| 4-1B-ens | Rank-mean composites within exp166-p1B's 30 scores | [`-4435545805`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4435545805) |
| 4-AG | Multi-composite ensembles with AlphaGenome | [`-4435574128`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4435574128) |
| 4-2way-LLR | Simple 2-way: LLR + AlphaGenome | [`-4435597295`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4435597295) |
| 4-2way-emb | Simple 2-way: embedding-distance + AlphaGenome | [`-4435609030`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4435609030) |
| 4-3way | 3-way: LLR + embed + AG; final recipe table | [`-4435614659`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4435614659) |
| 5 | Iter-5 kickoff (reopen) | [`-4452099593`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452099593) |
| 5-mend | Iter-5 exp166-p1B FWD+RC AVG — mendelian | [`-4452338816`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452338816) |
| 5-cmplx | Iter-5 exp166-p1B FWD+RC AVG — complex_traits | [`-4452340644`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452340644) |
| 5-eqtl | Iter-5 exp166-p1B FWD+RC AVG — eqtl | [`-4452342063`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452342063) |
| 5-sum | Iter-5 cross-dataset summary | [`-4452344147`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452344147) |
| 5-Q1 | Q1: paired McNemar LLR / L2 / best-ensemble | [`-4452418017`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452418017) |
| 5-Q2 | Q2: paired McNemar ensembles vs `mean_rank` | [`-4452418485`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452418485) |
| 5-Q1ps | Q1 per-subset paired McNemar | [`-4452475796`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452475796) |
| 5-Q2ps | Q2 per-subset paired McNemar | [`-4452478662`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452478662) |
| 6 | Iter-6 nuc-dep main: FWD+RC AVG, 8 scores | [`-4452713609`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4452713609) |
| 6-sys | Iter-6 systematic comparison of 8 nuc-dep scores | [`-4453058670`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4453058670) |
| 6-jsd | Iter-6 `down_jsd_mean` fixed vs LLR + L2 | [`-4453103460`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4453103460) |
| 6-3way | Iter-6 3-way: LLR / jsd_mean / mean_rank(LLR,L2) | [`-4453139255`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4453139255) |
| 6-logit | Iter-6 logit-space ensemble `mean_rank(LLR, jsd_mean)` | [`-4453302231`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4453302231) |
| 6-design | Design discussion + protocol options | [`-4453331086`](https://github.com/Open-Athena/bolinas-dna/issues/175#issuecomment-4453331086) |

## Original scope (preserved for context)

<details><summary>Question, scope, approach (collapsed)</summary>

### Scope

**In:**

- **Models (5+)** — `exp55-mammals` (promoter), `exp58-mammals` (CDS), `exp58-animals` (CDS), `exp59-mammals` (downstream), `exp136-proj_v30` (enhancer), and later **`exp166-p1B`** (1B zoonomia whole-genome generalist).
- **Windows (3 per specialist)** — `{128, native, 512}`. Native = 256 for exp55/58/59 (no BOS), 255 for exp136/exp166-p1B (BOS-prepended → 256 tokens).
- **Datasets (3)** — `bolinas-dna/evals_mendelian_traits`, `bolinas-dna/evals_complex_traits`, `bolinas-dna/evals_eqtl`, **train split only** (test held out for the eventual final-eval pass).
- **Subsets per dataset (8)** — `missense_variant`, `tss_proximal`, `5_prime_UTR_variant`, `3_prime_UTR_variant`, `non_coding_transcript_exon_variant`, `splicing`, `distal`, `synonymous_variant`.
- **Scoring rules (30 base)**: 6 likelihood (`llr`, `minus_llr`, `abs_llr`, `minus_logp_ref`, `minus_logp_alt`, `entropy`) + 24 embedding (3 distances × 4 pools × 2 layers).
- **Global aggregations**: per-subset, `pooled`, `macro`.

**Out (deferred / explicitly excluded):**

- **No leaderboard updates** — #161 / #162 / #172 stay frozen on their fixed scoring rules. (Recommendations for switching the leaderboard default are in #179.)
- **Test split** held out — train-split-only convention for development.

### Approach

Two-stage pipeline at [`snakemake/analysis/zeroshot_vep/`](https://github.com/Open-Athena/bolinas-dna/tree/main/snakemake/analysis/zeroshot_vep):

1. **GPU stage** (`extract_features`, one cache per (model, window, dataset)) — 4 forward passes per variant (one per candidate nucleotide at center). Inlined from biofoundation's `transform_llr_clm` / `_logits_to_logprobs` so we can change layer indices / output extras without going through biofoundation's Trainer wrapper.
2. **CPU stage** (`compute_scores` → `compute_metrics` → `aggregate_metrics`) — pure pandas/numpy from the cache. **Adding a new scoring rule = re-run stage 2 only**, no GPU.

For iter-4 specifically, the on-the-fly scoring approach (no embedding cache; 30 scores computed per-batch) avoided the ~17 GB-per-(model, window, dataset) cache budget.

Iter-5 / iter-6 (FWD+RC AVG protocol + nuc-dep scout) used ad-hoc scratch scripts on the same branch; the iter-5 RC scout reuses iter-4's `zeroshot_vep_iter4_precision_scout.py --strand rc`, iter-6 adds `zeroshot_vep_iter6_nucdep_scout.py` for downstream-effect scoring with strand control.

Branch: [`claude/great-herschel-9204ed`](https://github.com/Open-Athena/bolinas-dna/tree/claude/great-herschel-9204ed). Sanity-checked against `evals_v2` PairwiseAccuracy at native window with default scoring rules.

</details>
