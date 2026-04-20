# Project Guidelines

## Project Overview

**Bolinas** is a framework for developing genomic language models (gLMs). It includes training dataset creation and evaluations.

## Research Code Values

This is research code. Prioritize **reproducibility** and **correctness** over architectural elegance.

- **Coordinate system.** The codebase consistently uses 0-based, half-open intervals for all genomic coordinates. Assume this everywhere; call out any deviation explicitly.
- **Duplication beats premature abstraction.** Copying a function or a few lines between pipelines is fine when the alternative is a shared helper that couples unrelated experiments. Only promote code to `src/bolinas/` once the shape is stable and genuinely shared.
- **Modularity is a means, not a goal.** Don't refactor for reuse that may never come. Straight-line scripts that read top-to-bottom are often preferable to layered abstractions.
- **Test aggressively.** Every non-trivial function in `src/bolinas/` should have tests. For pipelines, add sanity checks on outputs (row counts, value ranges, coordinate invariants) rather than trusting that "it ran".
- **Assert defensively, everywhere.** Use `assert` liberally for invariants that *should* hold: coordinate bounds, dataframe shapes, no NaNs where none are expected, set membership, monotonicity, matching lengths between parallel arrays. A loud failure near the bug is worth far more than a silently corrupted result feeding into training.
- **Fail fast on silent-corruption risks.** Bioinformatics is full of off-by-one errors, strand mix-ups, and reference-build mismatches. When a result could be quietly wrong, prefer a check that crashes over a comment saying "this should be correct".
- **No premature generalizations.** If asked to implement a specific backend, dataset, or model variant, stick to that. Don't generalize to related use-cases on your own — offer the option, but only expand the scope when explicitly told to.
- **Stay in scope.** Don't remove or rewrite unrelated code in other pipelines or library modules while working on a task. Unrelated experiments may depend on exact current behavior.

## Code Structure

The codebase has two main components:

1. **Python Library** (`src/bolinas/`) - Core utilities for genomic interval manipulation and data processing. Code lives here once it's stable and genuinely shared across pipelines; see Research Code Values for when to promote.

2. **Pipelines** (`snakemake/`) - Data processing workflows implemented in Snakemake
   - Read the pipeline's README before working on it — each `snakemake/<pipeline>/` has its own. If you change pipeline behaviour, update the README in the same PR so the next human or agent can onboard from it.
   - Always dry-run first (`-n` / `--dry-run`) before any real invocation.
   - Stop before reruns of steps the changes you made for this task did not intentionally touch. If the dry-run shows Snakemake planning to rerun an upstream or unrelated step — retriggered by a timestamp change, an unrelated code edit, `--rerun-triggers` defaults, etc. — stop and ask before running. Default assumption: such reruns are unintended and potentially expensive (training, genome downloads, large bedtools jobs).
   - Invoke as `uv run snakemake …` from the repo root, not bare `snakemake`.
   - Put pipeline-wide defaults (`cores`, `use-conda`, `default-storage-provider`, etc.) in the pipeline's `workflow/profiles/default/config.yaml`, not on the CLI. Snakemake auto-loads that profile, so every invocation picks them up.

## Development Practices

- **Package management**: Use `uv` for Python dependencies
- **Bioinformatics tools**: Use Conda for external CLI tools (bedtools, twoBitToFa, etc.)
- **Testing**: Run `uv run pytest` before committing
- **Code quality**: Pre-commit hooks enforce ruff formatting and linting
- **Documentation**: Before merging a PR, make sure all the relevant READMEs are updated.

### Type Annotations
- Type-annotate all function parameters and return values in `src/bolinas/`. Pipeline scripts are lower priority but encouraged.
- Use Python 3.11+ syntax (`list[str]`, `X | None`); reach for `typing` only for constructs that still require it.

## Autonomy Boundaries

- Never push to `main` without explicit user approval.
- Never close or merge PRs/issues without explicit user approval.

## GitHub Communication

- When an agent creates a PR or issue, add the `agent-generated` label.
- Agent comments on PRs/issues must begin with `🤖`.
- For iterative investigations the user wants tracked in their own issue, use the `agent-research` skill — issue body is the living doc, comments are the append-only log with commit-pinned permalinks to code.
