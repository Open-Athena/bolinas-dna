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
