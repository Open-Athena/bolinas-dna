<p align="center">
  <img src="assets/logo.png" width="100" />
</p>

<h1 align="center">Bolinas</h1>

<p align="center">Open development of genomic language models — data, modeling, and evaluation.</p>

## Experiments

Development is driven by experiments tracked as GitHub issues.

### Baseline Runs

| Experiment | Status |
|-----------|--------|
| [#21](https://github.com/Open-Athena/bolinas-dna/issues/21) Promoters YOLO run | Closed - matches Evo 2 on promoter VEP but still behind GPN-Star |
| [#22](https://github.com/Open-Athena/bolinas-dna/issues/22) mRNA + promoters YOLO run | Closed - combined model consumed by coding regions; poor on promoter variants |
| [#27](https://github.com/Open-Athena/bolinas-dna/issues/27) CDS YOLO run | Closed - matches Evo 2 on missense variants but falls behind GPN-Star |

### Data

#### Genomic Regions

| Experiment | Status |
|-----------|--------|
| [#41](https://github.com/Open-Athena/bolinas-dna/issues/41) Promoters from mRNA vs. ncRNA | Closed - adding ncRNA promoters shows no significant difference in VEP performance |
| [#13](https://github.com/Open-Athena/bolinas-dna/issues/13) Mixing different genomic regions | Closed - balanced mixing gives balanced performance; proportional mixing dominated by CDS |
| [#53](https://github.com/Open-Athena/bolinas-dna/issues/53) Alternative datasets based on distance from CDS | Closed - distance-based heuristic (a la SpeciesLM) instead of UTR annotations |
| [#9](https://github.com/Open-Athena/bolinas-dna/issues/9) Repeat downweighting | Closed - downweighting repetitive elements improves VEP and stabilizes training |
| [#42](https://github.com/Open-Athena/bolinas-dna/issues/42) Promoter radius | Closed - smaller radius performs better; expanding to ±2kb degrades performance |
| [#43](https://github.com/Open-Athena/bolinas-dna/issues/43) Mixing 5 different regions | Closed - CDS, promoters, and 5' UTR learn well; 3' UTR and ncRNA show limited improvement |

#### Evolutionary Timescales

| Experiment | Status |
|-----------|--------|
| [#55](https://github.com/Open-Athena/bolinas-dna/issues/55) Promoters from different evolutionary timescales | Closed - mammals-trained model reaches good VEP performance fastest |
| [#58](https://github.com/Open-Athena/bolinas-dna/issues/58) CDS from different evolutionary timescales | Closed - longer timescales (animals) perform better for missense variants |
| [#59](https://github.com/Open-Athena/bolinas-dna/issues/59) Downstream regions from different evolutionary timescales | Closed - mammals trains fastest but all timescales converge with sufficient training |

#### Tokenization

| Experiment | Status |
|-----------|--------|
| [#64](https://github.com/Open-Athena/bolinas-dna/issues/64) K-mer tokenization for promoters | Closed - k-mer tokenization degrades VEP performance vs character-level; larger k performs worse |

### Modeling

#### Training Objectives

| Experiment | Status |
|-----------|--------|
| [#3](https://github.com/Open-Athena/bolinas-dna/issues/3) Different training objectives | Closed - CLM appears to do better than MLM at initial steps |

#### Architecture

| Experiment | Status |
|-----------|--------|
| [#37](https://github.com/Open-Athena/bolinas-dna/issues/37) Context size | Closed - 256bp and 512bp contexts perform similarly on VEP |
| [#14](https://github.com/Open-Athena/bolinas-dna/issues/14) Sliding window attention | Closed - alternating global/local attention matches all-global; local-only worse but large local model beats small global model |

#### Scaling

| Experiment | Status |
|-----------|--------|
| [#57](https://github.com/Open-Athena/bolinas-dna/issues/57) Scaling on a mixture dataset | Open |

### Evaluation

| Experiment | Status |
|-----------|--------|
| [#8](https://github.com/Open-Athena/bolinas-dna/issues/8) Understand relationship between perplexity and other metrics | Open |

---

## Installation

```bash
uv sync
```

## Development

```bash
# Install dev dependencies and pre-commit hooks
uv sync --group dev
uv run pre-commit install

# Run quality checks (ruff format/lint, snakefmt)
uv run pre-commit run

# Run tests
uv run pytest
```

## Project Structure

- `src/bolinas/` - Main Python package
  - `data/` - Genomic data structures (GenomicSet, etc.)
  - `evals/` - Evaluation utilities (inference, metrics, plotting)
- `snakemake/` - Snakemake workflows
  - [`training_dataset/`](snakemake/training_dataset/README.md) - Creates genomic training datasets from NCBI RefSeq genomes
  - [`evals/`](snakemake/evals/README.md) - Downloads and processes evaluation datasets
  - [`analysis/evals_v1/`](snakemake/analysis/evals_v1/README.md) - Evaluates trained models on variant effect prediction tasks
- `tests/` - Test suite
