"""Cross-species interval projection via Cactus alignments (MAF or HAL).

Used by the ``snakemake/analysis/maf_vs_hal_projection`` benchmark to compare
MAF-stream and ``halLiftover`` backends on the Zoonomia 447-mammal alignment.
The library code lives here so it has tests; the backend that wins the
benchmark stays, the loser is removed in the v1 projection pipeline.
"""
