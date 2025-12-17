# Genome selection

## Setup

Python dependencies are managed by the main project. See ../../README.md for installation.

Additionally, `conda` must be available for CLI tools.

## Usage

- Edit `config/config.yaml`
- Run pipeline:
```bash
uv run snakemake --cores all --use-conda
```
- Output is at `results/genomes/filtered.parquet`
