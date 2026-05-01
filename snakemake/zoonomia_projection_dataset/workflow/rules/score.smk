"""Per-window phyloP_447m scoring via pyBigWig.

We previously tried a kentUtils chain
(``bigWigToBedGraph | awk | bedGraphToBigWig`` plus
``bigWigAverageOverBed``) but the bioconda kentUtils binaries don't accept
pipes on ``stdin`` — both ``bedGraphToBigWig`` and ``faToTwoBit`` insist on
a regular file. Materialising the per-base bedGraph would need ~30 GB of
temp disk and a separate kentUtils-shell + run-block split because
``run:`` blocks don't activate the rule's conda env.

So we do everything in one ``run:`` block with pyBigWig (which is in
the project's uv env, no conda needed). For each window we:
  - fetch the per-base values as a NumPy array,
  - count finite values (``n_valid_bases``),
  - count finite values >= threshold (``conserved_bases``),
  - compute ``np.nanmean`` over finite values (``mean_phylop``).

NaN handling: bases where the bigWig has no signal are explicitly counted
as **non-conserved** (= 0) — the count is over the full window length, so
``proportion_conserved = conserved_bases / window_size`` matches what
``bigWigAverageOverBed``'s ``mean0`` column would have produced.

Performance budget: pyBigWig is ~10K windows/sec/core on 255 bp windows.
24M windows → ~40 min single-threaded. The single ``run:`` block can't
parallelise across cores easily; if this becomes a bottleneck, split by
chrom (24× speedup with chrom-wildcarded fan-out).
"""

from bolinas.conservation.scoring import score_windows as _score_windows


rule score_windows:
    """Score every 255 bp window against phyloP_447m at the calibrated threshold."""
    input:
        windows="results/windows/{species}.bed.gz",
        bw="results/bigwig/phyloP_447m.bw",
    output:
        "results/scored/{species}/phyloP_447m_windows.parquet",
    params:
        threshold=PHYLOP_447M_THRESHOLD,
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
        )
        assert (windows_df["end"] - windows_df["start"] == WINDOW_SIZE).all(), (
            "windows BED has rows of unexpected length"
        )

        scored = _score_windows(input.bw, windows_df, params.threshold)

        # Defensive asserts on the scoring result.
        assert len(scored) == len(windows_df)
        assert (scored["conserved_bases"] >= 0).all()
        assert (
            (scored["conserved_bases"] <= scored["n_valid_bases"])
            | (scored["n_valid_bases"] == 0)
        ).all(), "conserved_bases > n_valid_bases for some rows"
        assert (scored["n_valid_bases"] <= WINDOW_SIZE).all()
        assert scored["proportion_conserved"].min() >= 0.0
        assert scored["proportion_conserved"].max() <= 1.0
        # proportion_conserved == conserved_bases / WINDOW_SIZE
        mismatch = (
            scored["proportion_conserved"]
            - scored["conserved_bases"].cast(pl.Float32) / WINDOW_SIZE
        ).abs().max()
        assert mismatch < 1e-3, (
            f"proportion_conserved disagrees with conserved_bases/window_size "
            f"by {mismatch}"
        )

        scored.sort(["chrom", "start"]).write_parquet(output[0])
