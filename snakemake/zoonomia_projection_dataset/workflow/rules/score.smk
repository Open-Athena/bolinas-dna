"""Binarize phyloP_447m at the calibrated threshold, then score windows.

Two passes of ``bigWigAverageOverBed`` per window set:

1. binary track   → ``sum`` (= conserved_bases), ``mean0`` (= proportion conserved)
2. raw track      → ``mean`` (= mean phyloP), ``covered`` (= n_valid_bases)

UCSC bigWig tracks use ``chr1``-style chrom names; the windows BED uses
Ensembl bare names. We rewrite chrom names on-the-fly with awk before
``bigWigAverageOverBed``.
"""


rule binarize_447m:
    """Binarize phyloP_447m: 1 where value >= threshold, 0/missing elsewhere.

    Wiggletools' ``apply gte T x`` propagates gaps, which is the desired
    behaviour: NaN-bases in the input are absent from the binary track.
    Combined with ``bigWigAverageOverBed``'s ``mean0`` column (denominator
    = window size, not covered length), NaN ends up counted as
    non-conserved — see the pipeline README and the test in
    ``tests/conservation/`` if you need a sanity check.
    """
    input:
        "results/bigwig/phyloP_447m.bw",
    output:
        "results/bigwig/phyloP_447m.binary.bw",
    conda:
        "../envs/bioinformatics.yaml"
    params:
        threshold=PHYLOP_447M_THRESHOLD,
    shell:
        r"""
        wiggletools write_bw {output} gte {params.threshold} {input}
        """


rule score_windows:
    """Per-window phyloP_447m scoring via two bigWigAverageOverBed passes."""
    input:
        windows="results/windows/{species}.bed.gz",
        binary_bw="results/bigwig/phyloP_447m.binary.bw",
        raw_bw="results/bigwig/phyloP_447m.bw",
    output:
        "results/scored/{species}/phyloP_447m_windows.parquet",
    conda:
        "../envs/bioinformatics.yaml"
    run:
        import subprocess
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            # Rewrite chrom names: Ensembl "1" → UCSC "chr1" for bigWig lookup.
            chr_bed = tmp / "windows.chr.bed"
            with gzip.open(input.windows, "rt") as fin, open(chr_bed, "w") as fout:
                for line in fin:
                    parts = line.rstrip("\n").split("\t")
                    parts[0] = (
                        parts[0] if parts[0].startswith("chr") else f"chr{parts[0]}"
                    )
                    fout.write("\t".join(parts) + "\n")

            binary_tsv = tmp / "binary.tsv"
            raw_tsv = tmp / "raw.tsv"

            subprocess.run(
                ["bigWigAverageOverBed", input.binary_bw, str(chr_bed), str(binary_tsv)],
                check=True,
            )
            subprocess.run(
                ["bigWigAverageOverBed", input.raw_bw, str(chr_bed), str(raw_tsv)],
                check=True,
            )

            binary = parse_bigwig_average_over_bed(binary_tsv)
            raw = parse_bigwig_average_over_bed(raw_tsv)
            assert binary.shape == raw.shape, "binary/raw row counts must match"
            # Rejoin with original (non-chr-prefixed) windows BED on `name`.
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

            merged = (
                windows_df.join(
                    binary.select(
                        pl.col("name"),
                        pl.col("sum").cast(pl.Int32).alias("conserved_bases"),
                        pl.col("mean0").cast(pl.Float32).alias("proportion_conserved"),
                    ),
                    on="name",
                    how="inner",
                )
                .join(
                    raw.select(
                        pl.col("name"),
                        pl.col("mean").cast(pl.Float32).alias("mean_phylop"),
                        pl.col("covered").cast(pl.Int32).alias("n_valid_bases"),
                    ),
                    on="name",
                    how="inner",
                )
                .sort(["chrom", "start"])
            )

            assert len(merged) == len(windows_df), (
                f"join shrunk row count: {len(windows_df)} → {len(merged)}; "
                f"some window names did not match bigWigAverageOverBed output"
            )
            assert (merged["conserved_bases"] >= 0).all()
            assert (
                (merged["conserved_bases"] <= merged["n_valid_bases"])
                | (merged["n_valid_bases"] == 0)
            ).all(), "conserved_bases > n_valid_bases for some rows"
            assert (merged["n_valid_bases"] <= WINDOW_SIZE).all()
            assert merged["proportion_conserved"].min() >= 0.0
            assert merged["proportion_conserved"].max() <= 1.0
            # mean0 = sum / size: cross-check
            mismatch = (
                merged["proportion_conserved"]
                - merged["conserved_bases"].cast(pl.Float32) / WINDOW_SIZE
            ).abs().max()
            assert mismatch < 1e-3, (
                f"proportion_conserved disagrees with conserved_bases/window_size "
                f"by {mismatch}"
            )

            merged.write_parquet(output[0])
