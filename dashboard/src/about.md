---
title: About
---

# About this leaderboard

This site is the public face of the `bolinas-dna` matched-pair variant-effect evaluations: how well each gLM / conservation track / external baseline ranks pathogenic vs. benign variants within gene-matched pairs.

It replaces the hand-curated tables on [#161](https://github.com/Open-Athena/bolinas-dna/issues/161) (Mendelian), [#162](https://github.com/Open-Athena/bolinas-dna/issues/162) (Complex), [#172](https://github.com/Open-Athena/bolinas-dna/issues/172) (eQTL). v1 covers Mendelian only; the other two follow.

## Methodology

**PairwiseAccuracy** = fraction of `(positive, negative)` pairs (matched by `match_group` ⊇ gene + consequence) where the positive scores strictly higher than the negative. Ties count as 0.5. Standard error is the Wald binomial form: `√(p(1−p)/n)`. Implemented in [`src/bolinas/pipelines/evals/metrics.py`](https://github.com/Open-Athena/bolinas-dna/blob/main/src/bolinas/pipelines/evals/metrics.py).

Each method × dataset emits two aggregate rows alongside the per-subset cells:

- **Global** — PA across **all** match groups, regardless of per-subset size.
- **Macro Avg** — unweighted mean of per-subset PAs over subsets with `n_pairs ≥ 30`. SE is `√(Σ SE²) / K`, with K the number of qualifying subsets.

**Sort axis.** Mendelian sorts by Macro Avg (the variant composition is ~92% missense — a ClinVar annotator-history artifact, not pathogenicity reality — so Global PA over-weights protein-coding-specialist methods). Complex traits and eQTL stay on Global.

**Subset threshold.** Per-subset columns exclude subsets with `n_pairs < 30`; those still contribute to Global but not to Macro Avg.

**Train split only.** Test is held out for the final-eval pass. All numbers here reflect train development.

## Agent-readable data

The dashboard is a presentation layer over plain-text source files. To consume the data programmatically, fetch one of:

- **`dashboard/models.yaml`** in the repo — canonical metadata for every method. `gh api repos/Open-Athena/bolinas-dna/contents/dashboard/models.yaml` or `git show main:dashboard/models.yaml`.
- **`/data/models.json`** under this site — models.yaml normalized to JSON. Same fields as the YAML.
- **`/data/leaderboard.parquet`** under this site — long-form `(method × dataset × subset)` PA + SE + n_pairs + n_ties. Readable from Python (`pl.read_parquet(URL)`) or DuckDB (`SELECT * FROM read_parquet('URL')`).
- **`/data/datasets.json`** under this site — per-dataset metadata (HF commit, score type, etc.).

Every field shown in a table or tooltip is present in those files; the rendered HTML never hides information behind a click.

## Adding a new method

1. Append a YAML block to [`dashboard/models.yaml`](https://github.com/Open-Athena/bolinas-dna/blob/main/dashboard/models.yaml) (registry order; tag the appropriate `datasets`).
2. For `family: bolinas`, also add the model to [`snakemake/analysis/evals_v2/config/config.yaml`](https://github.com/Open-Athena/bolinas-dna/blob/main/snakemake/analysis/evals_v2/config/config.yaml).
3. Run the evals_v2 pipeline → parquet written to S3.
4. Open a PR; CI rebuilds this site and the new row appears.

The schema is documented at the top of models.yaml.
