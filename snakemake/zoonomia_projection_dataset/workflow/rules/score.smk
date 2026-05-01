"""Per-window phyloP_447m scoring via pyBigWig, parallelised across chroms.

We previously tried a kentUtils chain
(``bigWigToBedGraph | awk threshold | bedGraphToBigWig`` plus
``bigWigAverageOverBed``) but the bioconda kentUtils binaries don't accept
pipes on ``stdin`` — both ``bedGraphToBigWig`` and ``faToTwoBit`` insist on
a regular file. So scoring is done in pyBigWig in a Snakemake ``run:``
block, fanned out across chromosomes.

Each ``score_windows_chrom`` worker:
  - reads the full windows BED (small — gzipped ~5 MB),
  - filters to its chrom,
  - opens the bigWig once and loops, calling ``bw.values(...)`` per window,
  - writes a per-chrom Parquet.

Then ``merge_scored`` concatenates the 24 per-chrom Parquets into one
final Parquet sorted by ``(chrom, start)``.

Throughput: pyBigWig is ~10K windows/s/core. 24 chroms × ~1M windows each =
~24M total. Single-threaded would take ~40–60 min wall; on 8 vCPU
chrom-parallel it's roughly 24/8 = 3 batches × 1.5 min = ~5 min wall.
"""

from bolinas.conservation.scoring import score_windows as _score_windows


rule score_windows_chrom:
    """Score 255 bp windows on a single chromosome against phyloP_447m."""
    input:
        windows="results/windows/{species}.bed.gz",
        bw="results/bigwig/phyloP_447m.bw",
    output:
        "results/scored/{species}/per_chrom/phyloP_447m_{chrom}.parquet",
    params:
        threshold=PHYLOP_447M_THRESHOLD,
    wildcard_constraints:
        chrom="|".join(STANDARD_CHROMS),
    run:
        windows_df = pl.read_csv(
            input.windows,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end", "name"],
            schema_overrides={
                "chrom": pl.Utf8,
                "start": pl.Int64,
                "end": pl.Int64,
                "name": pl.Utf8,
            },
        ).filter(pl.col("chrom") == wildcards.chrom)
        assert len(windows_df) > 0, (
            f"no windows on chrom {wildcards.chrom!r}; standard_chroms in config"
            f" must match what's in the windows BED"
        )
        assert (windows_df["end"] - windows_df["start"] == WINDOW_SIZE).all()

        scored = _score_windows(input.bw, windows_df, params.threshold)

        assert len(scored) == len(windows_df)
        assert (scored["conserved_bases"] >= 0).all()
        assert (
            (scored["conserved_bases"] <= scored["n_valid_bases"])
            | (scored["n_valid_bases"] == 0)
        ).all()
        assert (scored["n_valid_bases"] <= WINDOW_SIZE).all()
        assert scored["proportion_conserved"].min() >= 0.0
        assert scored["proportion_conserved"].max() <= 1.0
        scored.sort("start").write_parquet(output[0])


rule merge_scored:
    """Concatenate per-chrom Parquets into a single sorted Parquet."""
    input:
        expand(
            "results/scored/{{species}}/per_chrom/phyloP_447m_{chrom}.parquet",
            chrom=STANDARD_CHROMS,
        ),
    output:
        "results/scored/{species}/phyloP_447m_windows.parquet",
    run:
        dfs = [pl.read_parquet(p) for p in input]
        seen_chroms = {df["chrom"].unique().item() for df in dfs}
        assert seen_chroms == set(STANDARD_CHROMS), (
            f"missing chroms in per-chrom Parquets: "
            f"{set(STANDARD_CHROMS) - seen_chroms}"
        )
        merged = pl.concat(dfs).sort(["chrom", "start"])
        # Defensive cross-check after concat.
        assert (merged["end"] - merged["start"] == WINDOW_SIZE).all()
        assert (merged["conserved_bases"] >= 0).all()
        assert (merged["proportion_conserved"] <= 1.0).all()
        merged.write_parquet(output[0])
