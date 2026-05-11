"""Zero-shot VEP scoring experiment branch.

Inline copy of the relevant biofoundation kernels with the abstraction layers
stripped out, so we can iterate on scoring rules without touching biofoundation.

- ``features``: 4-pass forward inference (one per candidate nucleotide at the
  variant center). Writes joint sequence log-probs, optional per-position
  log-probs, and per-position REF/ALT embeddings (last + middle layer) to an
  ``.npz`` cache. GPU-bound, run once per (model, window, dataset).
- ``scores``: pure-pandas/numpy post-processing of the cache into the 30 base
  score columns (6 likelihood + 24 embedding). CPU-bound, cheap to re-run.
"""
