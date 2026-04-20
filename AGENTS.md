# Project Guidelines

## Project Overview

**Bolinas** is a framework for developing genomic language models (gLMs). It includes training dataset creation and evaluations.

## Code Structure

The codebase has two main components:

1. **Python Library** (`src/bolinas/`) - Core utilities for genomic interval manipulation and data processing
   - Try to move us much shared functionality (e.g. functions) from the pipelines here, so they are subject to higher quality standards (e.g. typing, testing)

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
- Use Python 3.11+ type annotation syntax throughout
- Include type hints for all function parameters and return values
- Use `typing` module imports only when necessary for complex types
- Prefer built-in generic types (e.g., `list[str]` instead of `List[str]`)

### Constants Over Magic Numbers
- Replace hard-coded values with named constants
- Use descriptive constant names that explain the value's purpose
- Keep constants at the top of the file or in a dedicated constants file

### Meaningful Names
- Variables, functions, and classes should reveal their purpose
- Names should explain why something exists and how it's used

### Smart Comments
- Don't comment on what the code does - make the code self-documenting
- Use comments to explain why something is done a certain way
- Document APIs, complex algorithms, and non-obvious side effects

### Clean Structure
- Keep related code together
- Organize code in a logical hierarchy
- Use consistent file and folder naming conventions

### Code Quality Maintenance
- Refactor continuously
- Fix technical debt early
- Leave code cleaner than you found it

## Behavioral Guidelines

### Verify Information
Always verify information before presenting it. Do not make assumptions or speculate without clear evidence.

### No Apologies
Never use apologies.

### No Understanding Feedback
Avoid giving feedback about understanding in comments or documentation.

### No Summaries Of Your Work
Don't summarize changes made.

### No Unnecessary Confirmations
Don't ask for confirmation of information already provided in the context.

### Preserve Existing Code
Don't remove unrelated code or functionalities. Pay attention to preserving existing structures.

### No Implementation Checks
Don't ask the user to verify implementations that are visible in the provided context.

### No Unnecessary Updates
Don't suggest updates or changes to files when there are no actual modifications needed.

### Provide Real File Links
Always provide links to the real files, not x.md.

### No Current Implementation
Don't show or discuss the current implementation unless specifically requested.

### No Premature Generalizations
If you are asked to implement a specific backend, just stick to that. Do not generalize to other common or related use-cases. You can offer to implement these, but only do so if explicitly instructed to.

## GitHub Communication

- When an agent creates a PR or issue, add the `agent-generated` label.
- Agent comments on PRs/issues must begin with `🤖`.
- For iterative investigations the user wants tracked in their own issue, use the `agent-research` skill — issue body is the living doc, comments are the append-only log with commit-pinned permalinks to code.

## Autonomy Boundaries

- Never push to `main` without explicit user approval.
- Never close or merge PRs/issues without explicit user approval.

## Important Notes

- **Coordinate system**: The codebase consistently uses 0-based, half-open intervals for all genomic coordinates
- **Installation and usage**: See README.md for installation commands and general usage
- **Pipeline details**: Each pipeline has its own README with configuration options and usage instructions
