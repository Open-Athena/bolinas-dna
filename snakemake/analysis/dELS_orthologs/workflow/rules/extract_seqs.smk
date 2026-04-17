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
    """hg38 only: gold-standard-in dELS restricted to the hg38 search region.

    Reads the gold-filtered set (`dels_in_gold.parquet`) so every query has,
    by construction, at least one gold-standard mm10 partner. Window
    restriction is applied on top: in windowed mode, keep only dELS inside
    the configured chrom:start-end; in `whole_chrom` mode, keep only dELS on
    the configured chromosome; in `whole_genome` mode, no further filtering.
    """
    input:
        "results/cre/hg38/dels_in_gold.parquet",
    output:
        "results/cre/hg38/dels_window.parquet",
    run:
        chrom, start, end = get_search_window("hg38")
        df = pl.read_parquet(input[0])
        if chrom is not None:
            df = df.filter(pl.col("chrom") == chrom)
        if start:
            df = df.filter(pl.col("end") > start)
        if end is not None:
            df = df.filter(pl.col("start") < end)
        df = df.sort(["chrom", "start"])
        df.write_parquet(output[0])
        scope = (
            "genome-wide"
            if chrom is None
            else (f"{chrom}" if end is None else f"{chrom}:{start}-{end}")
        )
        print(f"  hg38: {df.height} gold-standard-in dELS in {scope}")


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


rule make_query_flanked_bed:
    """4-col BED for query FASTA extraction with symmetric flank bp.

    `flank` is an integer; positive expands each side, negative contracts
    (so the query shrinks by 2 × |flank| bp total). Uses the post-repeat-
    filter accession set from `query.filtered.parquet` — filtered on the
    ORIGINAL cCRE sequence's masked fraction, so the accession set is
    stable across flank values before the degenerate-interval skip.

    For negative flanks, cCREs that would become non-positive-length
    (`hi <= lo` after contraction) are dropped from the BED, so very
    short queries simply disappear from the flank-contracted run rather
    than producing empty FASTA records.
    """
    input:
        parquet="results/cre/hg38/query.filtered.parquet",
        chrom_sizes="results/genome/hg38.chrom.sizes",
    output:
        temp("results/cre/hg38/flank_{flank}/query.4col.bed"),
    wildcard_constraints:
        flank=r"-?\d+",
    run:
        flank = int(wildcards.flank)
        sizes: dict[str, int] = {}
        with open(input.chrom_sizes) as f:
            for line in f:
                c, s = line.rstrip("\n").split("\t")
                sizes[c] = int(s)
        df = pl.read_parquet(input.parquet).sort(["chrom", "start"])
        n_total = df.height
        n_kept = 0
        with open(output[0], "w") as fout:
            for row in df.iter_rows(named=True):
                chrom = row["chrom"]
                chrom_len = sizes.get(chrom)
                lo = max(0, row["start"] - flank)
                hi = row["end"] + flank
                if chrom_len is not None:
                    hi = min(chrom_len, hi)
                if hi <= lo:
                    continue
                fout.write(f"{chrom}\t{lo}\t{hi}\t{row['accession']}\n")
                n_kept += 1
        print(
            f"  hg38 flank_{wildcards.flank}: {n_kept:,}/{n_total:,} queries "
            f"retained (dropped {n_total- n_kept:,} with degenerate interval)"
        )


rule extract_flanked_query_fasta:
    """Per-flank query FASTA. Records named by cCRE accession (no position
    suffix) so the aligner's per-hit `query` column matches the accession
    downstream."""
    input:
        twobit="results/genome/hg38.2bit",
        bed="results/cre/hg38/flank_{flank}/query.4col.bed",
    output:
        "results/cre/hg38/flank_{flank}/query.filtered.fasta",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitToFa {input.twobit} {output} -bed={input.bed}"


rule subset_mm10_cres_to_window:
    """mm10 only: every Registry-V4 cCRE (any class) inside the mm10 search region.

    Used to annotate which mm10 cCRE each aligner hit overlaps. The search
    itself runs against the genomic FASTA, not these records. In whole-genome
    mode (`whole_genome: true`) the candidate pool is every mm10 cCRE.
    """
    input:
        "results/cre/mm10/cres.parquet",
    output:
        "results/cre/mm10/cres_window.parquet",
    run:
        chrom, start, end = get_search_window("mm10")
        df = pl.read_parquet(input[0])
        if chrom is not None:
            df = df.filter(pl.col("chrom") == chrom)
        if start:
            df = df.filter(pl.col("end") > start)
        if end is not None:
            df = df.filter(pl.col("start") < end)
        df = df.sort(["chrom", "start"])
        df.write_parquet(output[0])
        scope = (
            "genome-wide"
            if chrom is None
            else (f"{chrom}" if end is None else f"{chrom}:{start}-{end}")
        )
        print(f"  mm10: {df.height} cCREs in {scope} (any class)")


rule make_target_window_bed:
    """BED describing the mm10 target extent.

    Three modes (via `config["search_region"]["mm10"]`):
    - whole-genome (`whole_genome: true`): one line per standard mm10
      chromosome (chr1..chr19, chrX, chrY); excludes chrM, chr*_random,
      chrUn_*, alt scaffolds.
    - whole-chromosome (`whole_chrom: true`): one line spanning the full
      configured chromosome.
    - windowed (default): one line around ZRS ± `flank_bp`.

    Each line's name column is the chromosome name (so the resulting FASTA
    record is named after its chrom and downstream `hit_chrom` can be lifted
    directly from the aligner's per-hit target name).
    """
    input:
        chrom_sizes="results/genome/mm10.chrom.sizes",
    output:
        temp("results/target/mm10_window.bed"),
    run:
        region = config["search_region"]["mm10"]
        sizes = {}
        with open(input.chrom_sizes) as f:
            for line in f:
                c, s = line.rstrip("\n").split("\t")
                sizes[c] = int(s)

        lines: list[str] = []
        if region.get("whole_genome"):
            for c in MM10_STANDARD_CHROMS:
                if c not in sizes:
                    continue
                lines.append(f"{c}\t0\t{sizes[c]}\t{c}")
        elif region.get("whole_chrom"):
            chrom = region["chrom"]
            lines.append(f"{chrom}\t0\t{sizes[chrom]}\t{chrom}")
        else:
            chrom = region["chrom"]
            flank = region.get("flank_bp", 0)
            start = max(0, region["start"] - flank)
            end = region["end"] + flank
            lines.append(f"{chrom}\t{start}\t{end}\t{chrom}")

        with open(output[0], "w") as f:
            for line in lines:
                f.write(line + "\n")


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
