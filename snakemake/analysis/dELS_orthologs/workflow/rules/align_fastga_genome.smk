"""FASTGA in its native regime: whole hg38 vs whole mm10, then post-hoc
intersect alignments with the hg38 cCRE BED to assign per-cCRE orthologs.

This fixes the "using a genome-scale tool with 300 bp queries" framing
problem that the cCRE-as-queries `align_fastga.smk` pipeline has. lastz
and minimap2 have the same issue to varying degrees; this is the FASTGA-
native version.

The output still feeds per_query_report.smk through the shared unified
hits schema, so recall/precision numbers are directly comparable across
aligner modes.

The `flank` wildcard is kept for downstream path compatibility, but only
`flank_0` is meaningful here — flanking at the cCRE level doesn't apply
to whole-genome alignment (the "flank" is implicitly infinite — every
base on both sides is in the search).
"""


HG38_STANDARD_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


rule make_hg38_wholegenome_bed:
    """BED describing the standard hg38 chromosomes (chr1..22, chrX, chrY).

    Parallels `make_target_window_bed` for mm10 on the target side.
    """
    input:
        chrom_sizes="results/genome/hg38.chrom.sizes",
    output:
        temp("results/genome/hg38_wholegenome.bed"),
    run:
        sizes: dict[str, int] = {}
        with open(input.chrom_sizes) as f:
            for line in f:
                c, s = line.rstrip("\n").split("\t")
                sizes[c] = int(s)
        with open(output[0], "w") as fout:
            for c in HG38_STANDARD_CHROMS:
                if c in sizes:
                    fout.write(f"{c}\t0\t{sizes[c]}\t{c}\n")


rule extract_hg38_wholegenome_fasta:
    input:
        twobit="results/genome/hg38.2bit",
        bed="results/genome/hg38_wholegenome.bed",
    output:
        "results/genome/hg38_wholegenome.fasta",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitToFa {input.twobit} {output} -bed={input.bed}"


rule fastga_uppercase_hg38_genome:
    input:
        "results/genome/hg38_wholegenome.fasta",
    output:
        "results/genome/hg38_wholegenome.upper.fasta",
    shell:
        "awk '/^>/ {{print; next}} {{print toupper($0)}}' {input} > {output}"


rule fastga_genome_align:
    """Whole hg38 vs whole mm10 via FASTGA, native regime.

    Both sides are upper-cased (FASTGA's FAtoGDB crashes on soft-masked input)
    and passed with multi-record support. Flags mirror the cCRE-queries
    fastga_align defaults (`-s100 -i.55`) so the methodological difference
    is isolated to the input framing, not the parameter choice.

    Output is PAF with hg38 coords on the query side (`name1`, `zstart1`, `end1`)
    and mm10 coords on the target side.
    """
    input:
        query="results/genome/hg38_wholegenome.upper.fasta",
        target="results/target/mm10_window.upper.fasta",
    output:
        paf="results/align/{aligner}/flank_0/raw.paf",
    wildcard_constraints:
        aligner=r"fastga_genome(_.*)?",
    params:
        flags=lambda wc: config.get("fastga_variants", {}).get(wc.aligner, "-s100 -i.55"),
    threads: workflow.cores
    resources:
        mem_mb=48000,
    conda:
        "../envs/fastga.yaml"
    shell:
        r"""
        FastGA -v -k -T{threads} -paf \
            {params.flags} \
            {input.query} {input.target} \
            > {output.paf}
        """


rule normalize_fastga_genome_hits:
    """Project whole-genome FASTGA PAF to the unified per-cCRE hits schema.

    The PAF rows describe alignments between hg38 and mm10 genomic intervals
    (no cCRE awareness). For each alignment, we intersect its hg38 span with
    every hg38 cCRE BED interval and emit one hit row per overlapping cCRE,
    with the cCRE accession as `query` and the original mm10 span as the hit.
    per_query_report.smk then ranks hits within each cCRE and computes
    recall/precision as for every other aligner.

    If one alignment overlaps multiple cCREs (rare with non-overlapping cCREs
    but possible near boundaries), each gets its own row sharing the same
    mm10 hit interval and score. Top-k ranking within each cCRE is then
    per-cCRE, so the same mm10 hit can legitimately be top-1 for two
    adjacent cCREs — that matches the eval semantics already used by the
    cCRE-as-queries pipeline.
    """
    input:
        paf="results/align/{aligner}/flank_0/raw.paf",
        query_cres="results/cre/hg38/query.filtered.parquet",
    output:
        "results/align/{aligner}/flank_0/hits.tsv",
    wildcard_constraints:
        aligner=r"fastga_genome(_.*)?",
    run:
        # Parse PAF rows into a pandas-friendly frame first (we need bioframe).
        paf_rows: list[dict] = []
        with open(input.paf) as f:
            for line in f:
                if not line.strip():
                    continue
                p = line.rstrip("\n").split("\t")
                qname = p[0]
                qlen = int(p[1])
                qstart, qend = int(p[2]), int(p[3])
                strand = p[4]
                tname = p[5]
                tstart, tend = int(p[7]), int(p[8])
                matches = int(p[9])
                fident: float | None = None
                for tag in p[12:]:
                    if tag.startswith("dv:f:"):
                        fident = 1.0 - float(tag[5:])
                        break
                if fident is None:
                    alnlen = int(p[10])
                    fident = matches / alnlen if alnlen else 0.0
                paf_rows.append(
                    {
                        "hg38_chrom": qname,
                        "hg38_start": qstart,
                        "hg38_end": qend,
                        "rev_strand": strand == "-",
                        "hit_chrom": tname,
                        "hit_start": tstart,
                        "hit_end": tend,
                        "score": matches,
                        "fident": fident,
                        "qcov": (qend - qstart) / qlen if qlen else 0.0,
                        "tcov": 0.0,  # target-coverage is meaningless at whole-chrom target scale
                    }
                )
        paf_df = pl.DataFrame(paf_rows) if paf_rows else pl.DataFrame(schema={
            "hg38_chrom": pl.Utf8, "hg38_start": pl.Int64, "hg38_end": pl.Int64,
            "rev_strand": pl.Boolean, "hit_chrom": pl.Utf8, "hit_start": pl.Int64,
            "hit_end": pl.Int64, "score": pl.Int64, "fident": pl.Float64,
            "qcov": pl.Float64, "tcov": pl.Float64,
        })

        # Intersect hg38 alignment spans with hg38 query cCRE intervals.
        cres = pl.read_parquet(input.query_cres).select("chrom", "start", "end", "accession")
        if paf_df.height == 0:
            unified = pl.DataFrame(schema={
                "query": pl.Utf8, "hit_chrom": pl.Utf8, "hit_start": pl.Int64,
                "hit_end": pl.Int64, "rev_strand": pl.Boolean, "score": pl.Int64,
                "fident": pl.Float64, "evalue": pl.Float64, "qcov": pl.Float64,
                "tcov": pl.Float64,
            })
        else:
            j = bf.overlap(
                paf_df.to_pandas(),
                cres.to_pandas(),
                cols1=("hg38_chrom", "hg38_start", "hg38_end"),
                cols2=("chrom", "start", "end"),
                suffixes=("", "_cre"),
                how="inner",
            )
            unified = (
                pl.from_pandas(j)
                .select(
                    pl.col("accession_cre").alias("query"),
                    pl.col("hit_chrom"),
                    pl.col("hit_start"),
                    pl.col("hit_end"),
                    pl.col("rev_strand"),
                    pl.col("score"),
                    pl.col("fident"),
                    pl.lit(None, dtype=pl.Float64).alias("evalue"),
                    pl.col("qcov"),
                    pl.col("tcov"),
                )
            )
        unified.write_csv(output[0], separator="\t", include_header=True)
        print(
            f"  {wildcards.aligner}: {unified.height} cCRE-assigned alignments "
            f"across {unified['query'].n_unique()} queries "
            f"(from {paf_df.height} PAF rows)"
        )
