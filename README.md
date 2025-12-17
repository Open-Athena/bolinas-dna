# bolinas-dna

DNA training dataset creation and genomic interval utilities.

## Installation

Install the package and its dependencies:

```bash
uv sync
```

## Development Setup

To set up the development environment with linting, formatting, and testing:

```bash
# Install development dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install
```

## Development Tools

### Running Quality Checks

```bash
# Run all pre-commit hooks (ruff format/lint, snakefmt)
uv run pre-commit run
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=bolinas --cov-report=term-missing
```

## Project Structure

- `src/bolinas/` - Main Python package
  - `data/` - Genomic data structures (GenomicSet, etc.)
- `snakemake/` - Snakemake workflows
- `tests/` - Test suite

## Snakemake Workflows

The project includes Snakemake workflows for training dataset creation in `snakemake/training_dataset/`.

### Prerequisites

- All Python dependencies are managed by the main project (`uv sync`)
- Conda must be available for bioinformatics CLI tools

### Running Workflows

**Genome Selection:**
```bash
cd snakemake/training_dataset/genome_selection
uv run snakemake --cores all --use-conda
```

**Dataset Creation:**
```bash
cd snakemake/training_dataset/dataset_creation
uv run snakemake --cores all --use-conda
```

See individual workflow READMEs for configuration details.
