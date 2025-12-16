# Genome selection

## Setup

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
```

Additionally, the `conda` command must be available.

## Usage

- Edit `config/config.yaml`
- Run pipeline
```bash
source .venv/bin/activate
snakemake --cores all --use-conda
```
- Output is at `results/genomes/filtered.parquet`
