"""Filter scored windows by ``proportion_conserved`` and write a BED."""


rule filter_bed:
    input:
        "results/scored/{species}/phyloP_447m_windows.parquet",
    output:
        "results/bed/{species}/min{min_p}.bed.gz",
    run:
        min_p = float(wildcards.min_p)
        assert 0.0 <= min_p <= 1.0, f"min_p out of range: {min_p}"
        df = pl.read_parquet(input[0])
        kept = df.filter(pl.col("proportion_conserved") >= min_p).select(
            ["chrom", "start", "end", "name", "conserved_bases", "proportion_conserved"]
        )
        assert len(kept) <= len(df)
        # 0-based half-open BED, gzipped. 6 columns: chrom, start, end, name,
        # score (using conserved_bases), and strand "." -- this is BED6-ish but
        # downstream we only need the first 4. Using BED6 here to be friendly
        # to bedtools / bigWig consumers later. Actually BED6 wants strand;
        # standard convention is to put numeric score in col 5, strand in col 6.
        # We'll write 4-column BED to keep it simple.
        with gzip.open(output[0], "wt") as fout:
            for row in kept.iter_rows(named=True):
                fout.write(
                    f"{row['chrom']}\t{row['start']}\t{row['end']}\t{row['name']}\n"
                )
