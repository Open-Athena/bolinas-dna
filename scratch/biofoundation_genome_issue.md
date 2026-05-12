🤖 Filed in the course of [Open-Athena/bolinas-dna#175](https://github.com/Open-Athena/bolinas-dna/issues/175) (iter-3 scout — a 2-pass forward over ~15 K variants × 256 bp on A10G), where this became the run's dominant bottleneck.

## Symptom

`biofoundation.data.Genome.__call__` was responsible for ~all CPU time in a 2-pass forward over a small variant set on A10G: GPU at 0%, Python at 100%, processing ~5 var/sec (full pipeline went from ~5 min projected to ~50 min observed). `py-spy dump` on the hot Python thread:

```
__getitem__ (pandas/core/arrays/arrow/array.py:799)
_get_value  (pandas/core/series.py:1049)
__getitem__ (pandas/core/series.py:959)
__call__    (biofoundation/data.py:62)
_get_centered_window (...)
```

So the per-call cost is dominated by a pandas Series `__getitem__` that drops into the pyarrow-backed array's `__getitem__`, which is per-element O(many μs) when the Series is arrow-backed.

Workaround that fixed our scout (commit [`Open-Athena/bolinas-dna@0464553`](https://github.com/Open-Athena/bolinas-dna/commit/046455301e290836a79790164b2a345e1395c4a7), monkey-patch in [`scratch/zeroshot_vep_iter3_scout.py`](https://github.com/Open-Athena/bolinas-dna/blob/0464553/scratch/zeroshot_vep_iter3_scout.py)):

```python
genome = Genome(genome_path)
if hasattr(genome, "_genome"):
    genome._genome = {k: str(v) for k, v in dict(genome._genome).items()}
```

GPU went from 0% to 100% utilization; total runtime ~5 min. ~20× speedup.

## Where this matters

This affects any pipeline that calls `Genome(...)(chrom, start, end)` in a tight per-variant loop. With recent pandas (~2.2+, where many string columns are silently promoted to arrow-backed) the bottleneck appears without warning.

In bolinas-dna it touches at least:
- `snakemake/analysis/zeroshot_vep/` (iter-1 + iter-2 ran with an older pandas where this wasn't visible; iter-3's manual-install workflow on a fresh cluster exposed it)
- `snakemake/analysis/evals_v2/` (same Genome usage path)
- `snakemake/conservation_eval/` (likely the same)
- `scripts/evo2_eval/` (uses Genome via biofoundation)

## Proposed fixes (in order of cleanness)

1. **Cache resolved chromosomes inside `Genome`**: keep `self._genome` as the original Series for memory reasons but maintain a `self._cache: dict[str, str]` that's populated on first access per chromosome. Fastest follow-up.

   ```python
   class Genome:
       def __init__(self, path, subset_chroms=None):
           self._genome = read_fasta(path, subset_chroms=subset_chroms)
           self._chrom_sizes = {...}
           self._seq_cache: dict[str, str] = {}

       def _seq(self, chrom: str) -> str:
           if chrom not in self._seq_cache:
               self._seq_cache[chrom] = str(self._genome[chrom])
           return self._seq_cache[chrom]

       def __call__(self, chrom, start, end, strand="+"):
           ...
           seq = self._seq(chrom)[max(start, 0):min(end, chrom_size)]
           ...
   ```

   No API change, no new dependency.

2. **Drop arrow backing for the FASTA Series** by forcing object dtype in `read_fasta`. Loses arrow's memory efficiency but eliminates the slow per-call path. ~Easy if you don't otherwise need arrow ops on `self._genome`.

3. **Memory-map via `pyfaidx`** (the standard Python FASTA library). Loads only the chromosomes that are accessed; the index allows O(1) random seek; no `read_fasta` upfront. Bigger change but arguably the "right" answer for a generic genome accessor.

## Multi-worker / `multiprocessing` caveat (raised by @gonzalobenegas)

The naive workaround "convert `_genome` to a Python dict" has a memory pitfall when used with multi-worker `DataLoader` (or any `multiprocessing.Pool`):

- **Linux `fork` (DataLoader default)**: child processes inherit the parent's memory via copy-on-write. The dict is read-only, so pages stay shared. No memory blowup.
- **`spawn` (Python ≥3.14 default on Linux; always on macOS/Windows; PyTorch DataLoader can be configured to use it)**: each child gets the dict pickled in. ~3 GB × N_workers added RAM for the human genome.

The cached-dict-inside-class approach (fix 1) has the same caveat, but is easy to wrap with a flag to defer caching until first access per-worker. The `pyfaidx` approach (fix 3) sidesteps this entirely — workers reopen the mmap'd FASTA cheaply.

## Reproduce

On a fresh `g5.xlarge` (Ubuntu 22.04, driver 12.2) with the bolinas-dna pyproject.toml unmodified:

1. `sky launch` a g5.xlarge, `uv sync` (gets recent pandas + pyarrow).
2. Load Genome on GRCh38 primary assembly.
3. Run any per-variant loop calling `genome(chrom, start, end)` ~10,000 times.
4. `py-spy dump --pid <python-pid>` shows the hot path above; nvidia-smi shows GPU at 0% while Python at 100%.

Happy to PR fix (1) — it's the smallest change and is backward-compatible. Let me know.
