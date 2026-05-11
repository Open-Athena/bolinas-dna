## Question

Which zero-shot scoring rules — and which window sizes — make the bolinas gLMs most informative for variant-effect prediction across mendelian, complex-trait, and eQTL settings? The matched-pair leaderboards in #161 / #162 / #172 are frozen on a single fixed scoring rule per dataset (`minus_llr` for mendelian, `abs_llr` for complex / eqtl, via `biofoundation.inference.run_llr_and_embedding_distance`). We want **insights**, not a single winner — different scores plausibly favor different (model, dataset, consequence-subset) combos (e.g. embedding distance on enhancers, LLR on missense, entropy on regulatory variants). Documenting *which scoring rule helps which slice* is the headline goal.

## Scope

**In:**

- **Models (5)** — `exp55-mammals` (promoter), `exp58-mammals` (CDS), `exp58-animals` (CDS), `exp59-mammals` (downstream), `exp136-proj_v30` (enhancer). Same checkpoints used by the leaderboards' "best per region type" rows.
- **Windows (3 per model)** — `{128, native, 512}`. Native = 256 for exp55/58/59 (no BOS), 255 for exp136 (BOS-prepended → 256 tokens).
- **Datasets (3)** — `bolinas-dna/evals_mendelian_traits`, `bolinas-dna/evals_complex_traits`, `bolinas-dna/evals_eqtl`, **train split only** (test held out for the eventual final-eval pass).
- **Subsets per dataset (8)** — `missense_variant`, `tss_proximal`, `5_prime_UTR_variant`, `3_prime_UTR_variant`, `non_coding_transcript_exon_variant`, `splicing`, `distal`, `synonymous_variant`. Every model evaluated on every subset (cross-region transfer visible).
- **Scoring rules (30 base)**:
  - **Likelihood (6)** from the bidirectional 4-pass conditional `P(center | left, right)` — softmax over the joint sequence log-prob of each of the 4 candidate sequences (A/C/G/T at the variant center):
    `llr`, `minus_llr`, `abs_llr`, `minus_logp_ref`, `minus_logp_alt`, `entropy`.
  - **Embedding (24)** = 3 distances × 4 pool strategies × 2 layers:
    - distance ∈ {`l2`, `cosine`, `dot`}
    - pool ∈ {`flat` (no pool / flatten LD then distance — biofoundation default), `mean`, `varpos`, `lastpos`}
    - layer ∈ {`last`, `middle` = index `n_hidden_states // 2`}
- **Global aggregations (3)** — per-subset (8 rows), `global_pooled` (PairwiseAccuracy over all matched pairs flat — weighted by pair counts), `global_macro` (unweighted mean of per-subset accuracies with between-subset SEM).

**Out (deferred / explicitly excluded):**

- **No leaderboard updates** — #161 / #162 / #172 stay frozen on their fixed scoring rules.
- **Combinations** (rank-sum of base scores within a subset, etc.) deferred to iteration 2 once we see which singles carry signal.
- **Test split** held out — train-split-only convention for development.
- **Non-bolinas models** (Evo2, AlphaGenome, etc.) — separate study.

## Approach

Two-stage caching:

1. **GPU stage** (`extract_features`, one cache per (model, window, dataset)) — 4 forward passes per variant (one per candidate nucleotide at center). Writes:
   - `seq_logprob` `(N, 4)` fp32 — joint sequence log-prob under each candidate.
   - `pos_logprob` `(N, 4, T-1)` fp32 — per-position log-prob (cheap; for future per-position scoring).
   - `emb_{ref,alt}_{last,middle}` `(N, T, D)` fp16 — per-position hidden states from REF/ALT passes only (the other 2 passes' embeddings are discarded).
   - Plus indices: `ref_idx`, `alt_idx`, `var_pos`, `window_size`, `n_prefix`, `row_idx`.
   - ~17 GB per (model, window, dataset) at native windows; rotated per-model to stay within 200 GB SkyPilot disk.
2. **CPU stage** (`compute_scores` → `compute_metrics` → `aggregate_metrics`) — pure pandas/numpy from the cache. **Adding a new scoring rule = re-run stage 2 only**, no GPU.

The forward-pass logic is **inlined from biofoundation** (`transform_llr_clm`, `_logits_to_logprobs`, `HFCausalLMWithEmbeddings.forward`) into [`src/bolinas/zeroshot_vep/features.py`](https://github.com/Open-Athena/bolinas-dna/blob/ef1f01c9b9369c9784886065c4fe3e0897b3cab8/src/bolinas/zeroshot_vep/features.py) so we can change layer indices / output extras without going through biofoundation's Trainer wrapper. Memory: `F.cross_entropy(..., reduction='none')` instead of `log_softmax(logits).gather(...)` to avoid materializing the (B*4, T, V) softmax tensor at vocab Qwen3-sized.

**Sanity check:** at window=native + `minus_llr` (mendelian) / `abs_llr` (complex/eqtl), this pipeline must reproduce the `evals_v2` PairwiseAccuracy values within bf16 tolerance. The 4-pass LLR equals biofoundation's 2-pass CLM LLR algebraically (the softmax normalizer cancels in the difference). Will be cross-checked against [`s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/`](https://github.com/Open-Athena/bolinas-dna/blob/ef1f01c9b9369c9784886065c4fe3e0897b3cab8/snakemake/analysis/evals_v2/) on first run.

Pipeline: [`snakemake/analysis/zeroshot_vep/`](https://github.com/Open-Athena/bolinas-dna/tree/ef1f01c9b9369c9784886065c4fe3e0897b3cab8/snakemake/analysis/zeroshot_vep/).
Branch: [`claude/great-herschel-9204ed`](https://github.com/Open-Athena/bolinas-dna/tree/claude/great-herschel-9204ed).
Initial scaffolding commit: [`ef1f01c`](https://github.com/Open-Athena/bolinas-dna/commit/ef1f01c9b9369c9784886065c4fe3e0897b3cab8).

## Current findings

(To be filled in once the SkyPilot run completes. Expected shape: per-(dataset, model, window) heatmap with score columns on the y-axis and consequence subsets on the x-axis, color = PairwiseAccuracy. Plus the 30 × 3 (dataset) ranking of `global_macro` PairwiseAccuracy to surface the headline patterns.)

## Open questions / next steps

- **Iter 1 (in progress)** — run the 45 (model, window, dataset) feature-extraction jobs on SkyPilot; populate this issue body with the results table + heatmaps.
- **Iter 2 (after Iter 1 results)** — pick 3–5 rank-sum combinations informed by Pareto front of single scores; re-run stage 2 only (no GPU).
- **Possible Iter 3** — if "no pool" embedding distance loses badly to single-position pooling for some subsets, investigate whether the `pos_logprob` cache supports a "per-position effect trajectory" score (max L2 distance across positions, integrated area, etc.).
- **Open: window-size scaling** — does the `512` window help or hurt for distal vs missense subsets? Hypothesis: helps for distal eQTLs, hurts for missense (added context = noise for a local effect).

## Tracking

Description = current state. Comments = append-only iteration log (`🤖` prefix, commit-pinned permalinks). Pipeline README = how-to-run only.

Leaderboard policy: this issue's results table **does not propagate** to #161 / #162 / #172.
