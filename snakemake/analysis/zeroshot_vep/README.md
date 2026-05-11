# zeroshot_vep — VEP scoring rule comparison (experimental)

Exploration of zero-shot variant-effect scoring rules on the three matched-pair
eval datasets (mendelian_traits, complex_traits, eqtl). Results live on the
branch + a tracking GitHub issue — **the public matched-pair leaderboards
(#161, #162, #172) are NOT updated by this pipeline**.

## What this evaluates

For each (model, window, dataset, score, subset) combo, computes
`PairwiseAccuracy ± SE` on the train split only.

- **Models (5)**: `exp55-mammals` (promoter), `exp58-mammals` (CDS),
  `exp58-animals` (CDS), `exp59-mammals` (downstream), `exp136-proj_v30`
  (enhancer). Each model is evaluated on every dataset and every consequence
  subset — we want cross-region transfer, not just home-region performance.
- **Windows (3 per model)**: `{128, native, 512}` where `native` is the
  training-time window (256 for exp55/58/59, 255 for exp136). Half × / native /
  ~2× to see how shrink/expand changes scoring.
- **Datasets (3)**: mendelian_traits / complex_traits / eqtl — all share the
  same 8 consequence-group subsets and matched-pair schema.
- **Scores (30 base)**:
  - **Likelihood (6)** from the bidirectional 4-pass conditional
    `P(center | left, right)`:
    `llr`, `minus_llr`, `abs_llr`, `minus_logp_ref`, `minus_logp_alt`, `entropy`.
  - **Embedding (24)** = 3 distances × 4 pool strategies × 2 layers:
    distance ∈ {l2, cosine, dot} × pool ∈ {flat, mean, varpos, lastpos} ×
    layer ∈ {last, middle}.

Combinations (rank-sum, etc.) are deferred to iteration 2 once we see which
single scores carry signal.

## How

Two-stage caching: a GPU-bound feature-extraction step writes per-variant
features (joint seq log-probs over the 4 candidate sequences + per-position
embeddings for REF/ALT × {last, middle}) to one npz per (model, window,
dataset). The scoring stage is pure pandas/numpy on the cache; adding a new
scoring rule = re-run stage 2 only, no GPU needed.

```
results/
├── genome.fa.gz
├── checkpoints/{model}/                    # GCS pull
├── cache/{model}__win{W}__{dataset}.npz   # GPU stage (Stage 1)
├── scores/{model}__win{W}__{dataset}.parquet
├── metrics/{model}__win{W}__{dataset}.parquet
├── metrics_aggregated.parquet              # master table
└── metrics_aggregated.csv
```

## Running

```bash
# Local dry-run / planning:
cd snakemake/analysis/zeroshot_vep
uv run snakemake --profile workflow/profiles/default -n

# Real run on SkyPilot:
sky launch sky/run.yaml -c zeroshot-vep

# Iterate on scoring rules only (skip GPU re-extraction):
sky exec zeroshot-vep sky/run.yaml \
    --env SNAKEMAKE_ARGS='--forcerun compute_scores compute_metrics aggregate_metrics'

# Bring down at end of session:
sky down zeroshot-vep
```

## Where the code lives

- `src/bolinas/zeroshot_vep/features.py` — 4-pass forward inference + cache writer.
  Inlines biofoundation's `transform_llr_clm` + `_logits_to_logprobs` +
  `HFCausalLMWithEmbeddings.forward` so the loop is a single straight-line
  PyTorch function — no Trainer wrapper.
- `src/bolinas/zeroshot_vep/scores.py` — 30 scoring functions. Pure numpy.
  Iterate here.
- `tests/zeroshot_vep/test_scores.py` — unit tests with hand-computed expected
  values for each score family.

## Tracking

Living results doc: the tracking GitHub issue (linked from the PR / branch).
Issue body holds the latest aggregated table + heatmaps; per-iteration findings
go in append-only comments with commit-pinned permalinks.

## Important caveat

The score sign convention follows the existing leaderboards (higher = more
pathogenic) only **where the natural sign is unambiguous**:

- `minus_llr`, `abs_llr`, `minus_logp_ref`, `minus_logp_alt`, all embedding
  distances — natural sign is "higher = more impactful". Use directly.
- `llr` — natural sign is "alt favored over ref"; for pathogenic positives this
  is typically negative. Included for sanity; use `minus_llr` for matched-pair.
- `entropy` — natural sign is "high entropy = ambiguous position". For
  mendelian pathogenic variants (which tend to be at constrained positions)
  this points the wrong way; `pairwise_accuracy(entropy)` may end up below 0.5.
  That's informative, not a bug — interpret as evidence the inverted score is
  the useful one.
