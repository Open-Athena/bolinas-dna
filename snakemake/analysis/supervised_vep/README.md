# supervised_vep

Supervised variant-effect prediction on top of `exp166-p1B` embeddings — a
follow-up to the zero-shot study in [#175](https://github.com/Open-Athena/bolinas-dna/issues/175),
tracked in [#180](https://github.com/Open-Athena/bolinas-dna/issues/180).

## Goal

Cache mean-pooled embeddings + zero-shot scalars + TraitGym innerprod for
`exp166-p1B` × {`mendelian_traits`, `complex_traits`, `eqtl`} train splits, then
(in subsequent iterations) train classifiers on top with 3-fold chrom-grouped
CV and compare to the zero-shot recipes from #175.

## Layout

```
config/config.yaml          # models (exp166-p1B), datasets (3), inference knobs
workflow/Snakefile          # rule all → feature cache per (model, dataset)
workflow/rules/             # download_genome, compute_features
workflow/profiles/default/  # S3 storage backend (oa-bolinas)
sky/run.yaml                # SkyPilot launch task (A10G, us-east-2)
```

## Output cache schema

Each `results/features/{model}/{dataset}.parquet` has one row per variant
(input order preserved) with:

* **variant metadata** (carried from the HF dataset): `chrom, pos, ref, alt,
  label, subset, match_group, consequence_final, tss_closest_gene_id, ...`
* **zero-shot scalars**: `llr, minus_llr, abs_llr, embed_last_l2`
* **dense feature blocks** (parquet list columns, D-dim each):
  * `mean_ref`: mean of last hidden state over token positions, ref context
  * `mean_alt`: same, alt context
  * `traitgym_innerprod`: `(last_ref ⊙ last_alt).sum(seq_axis)` per channel

D is `exp166-p1B`'s hidden width (printed at the end of each rule).

## Running

Dry-run first (CLAUDE.md):

```bash
uv run snakemake --profile workflow/profiles/default --dry-run
```

Local invocation is not recommended — the 1B forward pass needs a GPU. Use
SkyPilot:

```bash
sky launch snakemake/analysis/supervised_vep/sky/run.yaml -c supervised-vep
```

Re-run on the same cluster after a code change:

```bash
sky exec supervised-vep snakemake/analysis/supervised_vep/sky/run.yaml
```

## Status

* iter-0 (this commit): pipeline scaffold + `compute_features` rule. Library
  code in `src/bolinas/supervised/`. Tests in `tests/supervised/`.
* iter-1+: classifier-fit + scoring + metrics rules (chrom-grouped CV, OOF
  predictions through `compute_pairwise_metrics`).
