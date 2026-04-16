"""Build the hg38 query FASTA (per-dELS) and the mm10 target FASTA (single window).

Asymmetric design:
- hg38 (query): dELS-class filter + ZRS-proper window + optional accession
  list + repeat-mask filter, then per-cCRE FASTA records.
- mm10 (target): one ~200 kb genomic FASTA record covering ZRS ± 100 kb.
  No class filter and no repeat-mask filter on the mm10 cCRE BED — that BED
  is only used downstream to annotate where mmseqs hits land, not to seed
  any search.
"""


rule subset_dels_to_window:
    """hg38 only: dELS that fall inside the hg38 search window (= ZRS proper)."""
    input:
        "results/cre/hg38/dels.parquet",
    output:
        "results/cre/hg38/dels_window.parquet",
    run:
        chrom, start, end = get_search_window("hg38")
        df = (
            pl.read_parquet(input[0])
            .filter(
                (pl.col("chrom") == chrom)
                & (pl.col("end") > start)
                & (pl.col("start") < end)
            )
            .sort(["chrom", "start"])
        )
        df.write_parquet(output[0])
        print(f"  hg38: {df.height} dELS in {chrom}:{start}-{end}")


rule select_query_accessions:
    """hg38 only: optionally restrict the query set to a configured accession list."""
    input:
        "results/cre/hg38/dels_window.parquet",
    output:
        parquet="results/cre/hg38/query.parquet",
        bed=temp("results/cre/hg38/query.4col.bed"),
    run:
        df = pl.read_parquet(input[0])
        accs = config.get("query_accessions") or None
        if accs:
            df = df.filter(pl.col("accession").is_in(list(accs)))
            print(
                f"  hg38: query restricted to {df.height} of {len(accs)} configured accessions"
            )
        else:
            print(
                f"  hg38: query is all {df.height} dELS in window (no accession restriction)"
            )
        df.write_parquet(output.parquet)
        df.select(["chrom", "start", "end", "accession"]).write_csv(
            output.bed, include_header=False, separator="\t"
        )


rule extract_query_fasta:
    """Per-dELS FASTA for the hg38 query set."""
    input:
        twobit="results/genome/hg38.2bit",
        bed="results/cre/hg38/query.4col.bed",
    output:
        "results/cre/hg38/query.fasta",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitToFa {input.twobit} {output} -bed={input.bed}"


rule filter_query_by_repeat:
    """Drop hg38 query dELS whose soft-masked fraction exceeds the threshold.

    mmseqs2 with `--mask-lower-case 1` has no k-mer seeds in heavily-masked
    sequence; querying with such a record returns no hits. Excluding those
    upfront keeps the per-query report honest (every reported query had real
    seeds available to search with).
    """
    input:
        fasta="results/cre/hg38/query.fasta",
        parquet="results/cre/hg38/query.parquet",
    output:
        fasta="results/cre/hg38/query.filtered.fasta",
        parquet="results/cre/hg38/query.filtered.parquet",
    params:
        max_frac=config["max_soft_masked_frac"],
    run:
        fracs = soft_masked_fraction_per_record(input.fasta)
        keep = {acc for acc, frac in fracs.items() if frac <= params.max_frac}
        n_total, n_kept = len(fracs), len(keep)
        print(
            f"  hg38: keeping {n_kept}/{n_total} query dELS "
            f"with soft-masked frac <= {params.max_frac}"
        )

        with open(input.fasta) as fin, open(output.fasta, "w") as fout:
            emit = False
            for line in fin:
                if line.startswith(">"):
                    emit = line[1:].strip().split()[0] in keep
                if emit:
                    fout.write(line)

        df = pl.read_parquet(input.parquet)
        df.filter(pl.col("accession").is_in(list(keep))).write_parquet(output.parquet)


rule subset_mm10_cres_to_window:
    """mm10 only: every Registry-V4 cCRE (any class) inside the mm10 search window.

    Used to annotate which mm10 cCRE each mmseqs hit overlaps. No filter — the
    target search itself happens against the genomic FASTA, not these records.
    """
    input:
        "results/cre/mm10/cres.parquet",
    output:
        "results/cre/mm10/cres_window.parquet",
    run:
        chrom, start, end = get_search_window("mm10")
        df = (
            pl.read_parquet(input[0])
            .filter(
                (pl.col("chrom") == chrom)
                & (pl.col("end") > start)
                & (pl.col("start") < end)
            )
            .sort(["chrom", "start"])
        )
        df.write_parquet(output[0])
        print(f"  mm10: {df.height} cCREs in {chrom}:{start}-{end} (any class)")


rule make_target_window_bed:
    output:
        temp("results/target/mm10_window.bed"),
    run:
        chrom, start, end = get_search_window("mm10")
        with open(output[0], "w") as f:
            f.write(f"{chrom}\t{start}\t{end}\tmm10_window\n")


rule extract_target_fasta:
    input:
        twobit="results/genome/mm10.2bit",
        bed="results/target/mm10_window.bed",
    output:
        "results/target/mm10_window.fasta",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitToFa {input.twobit} {output} -bed={input.bed}"
