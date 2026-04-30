# Conservation scores baseline (issue #146)

Per-variant conservation scores on TraitGym Mendelian v2 (`bolinas-dna/evals-traitgym_mendelian_v2`) as a classical baseline alongside model-based evals (e.g. Evo 2 in `scripts/evo2_eval/`, issue #131).

## What it does

For each split in `config["splits"]` (default `train` and `test`) and each track in `config["scores"]`:

1. **Download** the bigWig from UCSC / Zoonomia (`results/conservation/{score}.bw`).
2. **Score** each variant by single-base lookup at its 1-based `pos` (converted to pyBigWig's 0-based half-open `[pos-1, pos)`). NaN is preserved where the bigWig has no aligned data.
3. **Aggregate** the three scored parquets into one AUPRC table per split (`results/conservation_traitgym_v2/results_table_{split}.md`) plus a long-form `metrics_{split}.parquet`.

## Tracks

| Name | Source URL | Notes |
| --- | --- | --- |
| `phyloP_100v` | `hgdownload.soe.ucsc.edu/.../hg38.phyloP100way.bw` | 100-vertebrate phyloP |
| `phyloP_241m` | `hgdownload.soe.ucsc.edu/.../cactus241way.phyloP.bw` | Zoonomia 241-mammal Cactus phyloP |
| `phastCons_43p` | `cgl.gi.ucsc.edu/.../phyloPPrimates.bigWig` | Zoonomia 43-primate. Name follows TraitGym; underlying file is phyloP-over-primates. |

URLs are owned by `bolinas.evals.conservation.CONSERVATION_TRACKS` (single source of truth).

## NaN handling

NaN values arise when the bigWig has no alignment at a given locus (e.g. patches, alt contigs, or genuinely unalignable bases). We make a deliberate choice to **`fillna(0)`** before computing AUPRC:

- For phyloP, **0 is semantically meaningful**: neither conserved nor accelerated.
- For phastCons, **0 is also semantically meaningful**: non-conserved.

This matches the choice in TraitGym's own `eval/workflow/rules/conservation.smk`. Some other benchmarks instead drop NaN variants from evaluation; we don't, for simplicity.

The per-variant parquets (`{score}_{split}.parquet`) preserve raw NaNs so you can apply a different policy without re-running scoring. Filling happens only at the aggregation stage.

NaN counts are surfaced in `results_table_{split}.md` (per `(score, subset)`) so the size of the choice is visible.

## Usage

```bash
cd snakemake/conservation_eval

# Dry-run to inspect the DAG
uv run snakemake -n

# Run locally (CPU-only; ~5-10 min total — bigWig downloads dominate)
uv run snakemake
```

The default profile (`workflow/profiles/default/config.yaml`) uses S3 storage at `s3://oa-bolinas/snakemake/conservation_eval/`. AWS credentials need S3 access — same setup as `snakemake/evals/` (see that pipeline's README for credentials).

## Outputs

```
results/
├── conservation/
│   ├── phyloP_100v.bw
│   ├── phyloP_241m.bw
│   └── phastCons_43p.bw
└── conservation_traitgym_v2/
    ├── phyloP_100v_train.parquet         # per-variant score (NaN preserved)
    ├── phyloP_100v_test.parquet
    ├── phyloP_241m_train.parquet
    ├── phyloP_241m_test.parquet
    ├── phastCons_43p_train.parquet
    ├── phastCons_43p_test.parquet
    ├── metrics_train.parquet              # long-form metrics table
    ├── metrics_test.parquet
    ├── results_table_train.md             # markdown for issue follow-up
    └── results_table_test.md
```

## Library

Snakemake rules are thin glue around `bolinas.evals.conservation`:
- `score_variants_at_positions(df, bw_path)` — bigWig lookup, NaN preserved.
- `aggregate_traitgym_metrics(parquet_paths)` — `(metrics_df, markdown)`.

Tests live at `tests/evals/test_conservation.py` (CPU-only, no real bigWig download).
