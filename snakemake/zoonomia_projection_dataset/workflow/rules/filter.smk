"""Filter scored windows by ``proportion_conserved`` and write a BED."""


rule filter_bed:
    input:
        "results/scored/phyloP_447m_windows.parquet",
    output:
        "results/bed/min{min_p}.bed.gz",
    run:
        min_p = float(wildcards.min_p)
        assert 0.0 <= min_p <= 1.0, f"min_p out of range: {min_p}"
        df = pl.read_parquet(input[0])
        kept = df.filter(pl.col("proportion_conserved") >= min_p)
        assert len(kept) <= len(df)
        with gzip.open(output[0], "wt") as fout:
            for row in kept.iter_rows(named=True):
                fout.write(
                    f"{row['chrom']}\t{row['start']}\t{row['end']}\t{row['name']}\n"
                )
