## Question

Which zero-shot scoring rules — and which window sizes — make the bolinas gLMs most informative for variant-effect prediction across mendelian, complex-trait, and eQTL settings? The matched-pair leaderboards in #161 / #162 / #172 are frozen on a single fixed scoring rule per dataset (`minus_llr` for mendelian, `abs_llr` for complex / eqtl, via `biofoundation.inference.run_llr_and_embedding_distance`). We want **insights**, not a single winner — different scores plausibly favor different (model, dataset, consequence-subset) combos.

## Final conclusions

**Status: closed.** Investigation complete across 4 iterations and ~16 GitHub comments. Headline conclusions:

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
- **biofoundation#24** — offline batched VEP path.
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

Branch: [`claude/great-herschel-9204ed`](https://github.com/Open-Athena/bolinas-dna/tree/claude/great-herschel-9204ed). Sanity-checked against `evals_v2` PairwiseAccuracy at native window with default scoring rules.

</details>
