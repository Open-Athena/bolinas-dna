# Conservation scores baseline (issue #146)

Per-variant conservation scores on the matched-pair eval datasets produced by
`snakemake/evals/` (PR #159):

- `bolinas-dna/evals_mendelian_traits` — Mendelian-disease pathogenic SNVs vs
  gnomAD common variants, 1:1 gene-matched.
- `bolinas-dna/evals_complex_traits` — UKBB fine-mapped (PIP > 0.9) vs low-PIP
  (< 0.01), 1:1 gene-matched.

A classical baseline alongside model-based evals (e.g. Evo 2 in
`scripts/evo2_eval/`).

## What it does

For each `dataset` in `config["datasets"]`, each `split` in `config["splits"]`
(default `train` and `test`), and each `score` in `config["scores"]`:

1. **Download** the bigWig from UCSC / Zoonomia (`results/conservation/{score}.bw`).
2. **Score** each variant by single-base lookup at its 1-based `pos` (converted
   to pyBigWig's 0-based half-open `[pos-1, pos)`). NaN is preserved where the
   bigWig has no aligned data.
3. **Aggregate** the scored parquets into one PairwiseAccuracy table per
   `(dataset, split)` (`results/{dataset}/results_table_{split}.md`) plus a
   long-form `metrics_{split}.parquet`.

## Metric: PairwiseAccuracy ± SE

The eval datasets are matched pairs: every positive variant has exactly one
matched negative variant in the same `match_group`. We score this as a
within-group classification:

- For each `match_group`: +1 if the positive scores higher than the negative,
  +0.5 on a tie, +0 otherwise.
- Reported value = mean across pairs in the subset.
- SE = `sqrt(value * (1 - value) / n_pairs)` (Wald binomial — adequate for the
  hundreds-to-thousands of pairs per subset; ties contribute to `value` but
  the SE form is unchanged).

The markdown table reports per-subset rows (consequence groups: missense,
distal, splicing, promoter, …) — no `global` or `mean` aggregate row, so each
subset is read on its own terms.

## Tracks

| Name | Source URL | Notes |
| --- | --- | --- |
| `phyloP_100v` | `hgdownload.soe.ucsc.edu/.../hg38.phyloP100way.bw` | 100-vertebrate phyloP (UCSC multiz) |
| `phastCons_100v` | `hgdownload.soe.ucsc.edu/.../hg38.phastCons100way.bw` | 100-vertebrate phastCons (UCSC multiz) |
| `phyloP_241m` | `hgdownload.soe.ucsc.edu/.../cactus241way.phyloP.bw` | Zoonomia 241-mammal Cactus phyloP |
| `phyloP_447m` | `hgdownload.soe.ucsc.edu/.../hg38.phyloP447way.bw` | UCSC 447-way phyloP (Zoonomia + densely-sampled primates, Kuderna et al. 2023) |
| `phyloP_470m` | `hgdownload.soe.ucsc.edu/.../hg38.phyloP470way.bw` | UCSC 470-way phyloP (multiz; parallel work to 447-way Cactus, not a successor) |
| `phastCons_470m` | `hgdownload.soe.ucsc.edu/.../hg38.phastCons470way.bw` | UCSC 470-way phastCons (multiz; parallel work to 447-way Cactus, not a successor) |
| `phastCons_43p` | `cgl.gi.ucsc.edu/.../phyloPPrimates.bigWig` | Zoonomia 43-primate. Name follows TraitGym; underlying file is phyloP-over-primates. |

URLs are owned by `bolinas.evals.conservation.CONSERVATION_TRACKS` (single source of truth).

## NaN handling

NaN values arise when the bigWig has no alignment at a given locus (e.g.
patches, alt contigs, or genuinely unalignable bases). Before computing
PairwiseAccuracy we **`fillna(0)`**:

- For phyloP, **0 is semantically meaningful**: neither conserved nor accelerated.
- For phastCons, **0 is also semantically meaningful**: non-conserved.

This matches the choice in TraitGym's own `eval/workflow/rules/conservation.smk`.

The per-variant parquets (`{score}_{split}.parquet`) preserve raw NaNs so you
can apply a different policy without re-running scoring. Filling happens only
at the aggregation stage.

NaN counts are surfaced in `results_table_{split}.md` (per `(score, subset)`)
so the size of the choice is visible.

## Usage

```bash
cd snakemake/conservation_eval

# Dry-run to inspect the DAG
uv run snakemake -n

# Run locally (CPU-only — wall time is dominated by bigWig downloads;
# the full 7-track set is ~50 GB total). Use --cores to cap parallelism if
# you're on a shared node.
uv run snakemake
```

The default profile (`workflow/profiles/default/config.yaml`) uses S3 storage
at `s3://oa-bolinas/snakemake/conservation_eval/`. AWS credentials need S3
access — same setup as `snakemake/evals/`.

## Outputs

```
results/
├── conservation/
│   └── {score}.bw                          # one per track in config["scores"]
└── {dataset}/
    ├── {score}_{split}.parquet             # per-variant score (NaN preserved)
    ├── metrics_{split}.parquet             # long-form metrics table
    └── results_table_{split}.md            # markdown report (PairwiseAccuracy ± SE)
```

Each `metrics_{split}.parquet` has columns `[score_type, score_name, subset,
value, se, n_pairs, n_ties, n_nan, n_total, split, dataset]`.

## Library

Snakemake rules are thin glue around `bolinas.evals.conservation`:

- `score_variants_at_positions(df, bw_path)` — bigWig lookup, NaN preserved.
- `aggregate_conservation_metrics(parquet_paths)` — `(metrics_df, markdown)`.

Tests live at `tests/evals/test_conservation.py` and
`tests/evals/test_pairwise_accuracy.py` (CPU-only, no real bigWig download).
