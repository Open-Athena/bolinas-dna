"""Binarize phyloP_447m at the calibrated threshold, then score windows.

Two passes of ``bigWigAverageOverBed`` per window set:

1. binary track   → ``sum`` (= conserved_bases), ``mean0`` (= proportion conserved)
2. raw track      → ``mean`` (= mean phyloP), ``covered`` (= n_valid_bases)

UCSC bigWig tracks use ``chr1``-style chrom names; the windows BED uses
Ensembl bare names. The ``bigwig_average_over_bed`` rule rewrites chrom
names on-the-fly with awk before each ``bigWigAverageOverBed`` call.

Note on rule splitting: ``bigWigAverageOverBed`` and ``wiggletools`` come
from a conda env, but Snakemake only activates ``conda:`` for ``shell:`` /
``script:`` rules — not ``run:``. So the kentUtils invocations live in
``shell:`` rules and the join/assertion logic lives in a ``run:`` rule.
"""


rule binarize_447m:
    """Binarize phyloP_447m: 1 where value >= threshold, gap elsewhere.

    Wiggletools' ``gte T x`` propagates gaps, which is the desired
    behaviour: NaN-bases in the input are absent from the binary track.
    Combined with ``bigWigAverageOverBed``'s ``mean0`` column (denominator
    = window size, not covered length), NaN ends up counted as
    non-conserved.
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


rule bigwig_average_over_bed:
    """Two bigWigAverageOverBed passes against the chr-prefixed windows BED.

    Emits ``binary.tsv`` (= conserved_bases / proportion_conserved) and
    ``raw.tsv`` (= mean_phylop / n_valid_bases). Joining + Parquet writing
    happens in the next rule.
    """
    input:
        windows="results/windows/{species}.bed.gz",
        binary_bw="results/bigwig/phyloP_447m.binary.bw",
        raw_bw="results/bigwig/phyloP_447m.bw",
    output:
        binary_tsv=temp("results/scored/{species}/phyloP_447m.binary.tsv"),
        raw_tsv=temp("results/scored/{species}/phyloP_447m.raw.tsv"),
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        r"""
        TMPBED=$(mktemp --suffix=.bed)
        trap "rm -f $TMPBED" EXIT
        zcat {input.windows} \
          | awk 'BEGIN{{OFS="\t"}} {{ if ($1 !~ /^chr/) $1="chr"$1; print }}' \
          > $TMPBED
        bigWigAverageOverBed {input.binary_bw} $TMPBED {output.binary_tsv}
        bigWigAverageOverBed {input.raw_bw}    $TMPBED {output.raw_tsv}
        """


rule score_windows:
    """Join the two bigWigAverageOverBed TSVs back to the windows BED → Parquet.

    Columns:
      chrom, start, end, name (from windows.bed.gz, bare Ensembl chrom names)
      conserved_bases (Int32) = `sum`  from binary track
      proportion_conserved (Float32) = `mean0` from binary track (= sum / size,
                                       NaN counted as 0 → desired semantics)
      mean_phylop (Float32) = `mean` from raw track (over covered bases only)
      n_valid_bases (Int32) = `covered` from raw track
    """
    input:
        windows="results/windows/{species}.bed.gz",
        binary_tsv="results/scored/{species}/phyloP_447m.binary.tsv",
        raw_tsv="results/scored/{species}/phyloP_447m.raw.tsv",
    output:
        "results/scored/{species}/phyloP_447m_windows.parquet",
    run:
        binary = parse_bigwig_average_over_bed(input.binary_tsv)
        raw = parse_bigwig_average_over_bed(input.raw_tsv)
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
