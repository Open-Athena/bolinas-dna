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
        FastGA -v -k -T{threads} -pafx \
            {params.flags} \
            {input.query} {input.target} \
            > {output.paf}
        """


rule normalize_fastga_genome_hits:
    """Project whole-genome FASTGA PAF to the unified per-cCRE hits schema
    via exact CIGAR-based lift.

    FASTGA alignments at genome-vs-genome scale are multi-kb syntenic blocks
    (median ~9 kb on hg38↔mm10). We need per-cCRE mm10 coords, not block-
    level. An earlier proportional-lift approximation drifted by cumulative
    gap offsets (hundreds of bp), missing mm10 cCRE boundaries.

    Fix: use `bolinas.lift.cigar_lift` to walk each alignment's CIGAR string
    (present because we pass `-pafx` to FASTGA) and compute the exact mm10
    sub-coords corresponding to each overlapping hg38 cCRE's span. Tested
    at `tests/test_lift.py`.
    """
    input:
        paf="results/align/{aligner}/flank_0/raw.paf",
        query_cres="results/cre/hg38/query.filtered.parquet",
    output:
        "results/align/{aligner}/flank_0/hits.tsv",
    wildcard_constraints:
        aligner=r"fastga_genome(_.*)?",
    run:
        from bolinas.lift import cigar_lift

        # Parse PAF rows, retaining CIGAR from the `cg:Z:` tag (-pafx output).
        paf_rows: list[dict] = []
        with open(input.paf) as f:
            for line in f:
                if not line.strip():
                    continue
                p = line.rstrip("\n").split("\t")
                qname = p[0]
                qstart, qend = int(p[2]), int(p[3])
                strand = p[4]
                tname = p[5]
                tstart, tend = int(p[7]), int(p[8])
                matches = int(p[9])
                cigar = ""
                fident: float | None = None
                for tag in p[12:]:
                    if tag.startswith("cg:Z:"):
                        cigar = tag[5:]
                    elif tag.startswith("dv:f:"):
                        fident = 1.0 - float(tag[5:])
                if fident is None:
                    alnlen = int(p[10])
                    fident = matches / alnlen if alnlen else 0.0
                paf_rows.append({
                    "hg38_chrom": qname,
                    "hg38_start": qstart,
                    "hg38_end": qend,
                    "strand": strand,
                    "hit_chrom": tname,
                    "hit_start": tstart,
                    "hit_end": tend,
                    "score": matches,
                    "fident": fident,
                    "cigar": cigar,
                })
        print(f"  {wildcards.aligner}: {len(paf_rows)} PAF rows, "
              f"{sum(1 for r in paf_rows if r['cigar'])} with CIGAR")

        cres = pl.read_parquet(input.query_cres).select("chrom", "start", "end", "accession")

        schema_unified = {
            "query": pl.Utf8, "hit_chrom": pl.Utf8, "hit_start": pl.Int64,
            "hit_end": pl.Int64, "rev_strand": pl.Boolean, "score": pl.Int64,
            "fident": pl.Float64, "evalue": pl.Float64, "qcov": pl.Float64,
            "tcov": pl.Float64,
        }

        if not paf_rows:
            unified = pl.DataFrame(schema=schema_unified)
        else:
            paf_df_for_overlap = pl.DataFrame([
                {"hg38_chrom": r["hg38_chrom"], "hg38_start": r["hg38_start"],
                 "hg38_end": r["hg38_end"], "paf_idx": i}
                for i, r in enumerate(paf_rows)
            ]).to_pandas()
            j = bf.overlap(
                paf_df_for_overlap,
                cres.to_pandas(),
                cols1=("hg38_chrom", "hg38_start", "hg38_end"),
                cols2=("chrom", "start", "end"),
                suffixes=("", "_cre"),
                how="inner",
            )
            out_rows: list[dict] = []
            for row in j.itertuples(index=False):
                paf = paf_rows[row.paf_idx]
                if not paf["cigar"]:
                    continue  # can't lift without CIGAR
                lifted = cigar_lift(
                    q_start=paf["hg38_start"], q_end=paf["hg38_end"],
                    t_start=paf["hit_start"], t_end=paf["hit_end"],
                    strand=paf["strand"], cigar=paf["cigar"],
                    lift_q_start=row.start_cre, lift_q_end=row.end_cre,
                )
                if lifted is None:
                    continue
                mt_start, mt_end = lifted
                out_rows.append({
                    "query": row.accession_cre,
                    "hit_chrom": paf["hit_chrom"],
                    "hit_start": mt_start,
                    "hit_end": mt_end,
                    "rev_strand": paf["strand"] == "-",
                    "score": int(paf["score"] * (row.end_cre - row.start_cre) / max(1, paf["hg38_end"] - paf["hg38_start"])),
                    "fident": paf["fident"],
                    "evalue": None,
                    "qcov": 1.0,
                    "tcov": 0.0,
                })
            unified = pl.DataFrame(out_rows, schema=schema_unified) if out_rows else pl.DataFrame(schema=schema_unified)

        unified.write_csv(output[0], separator="\t", include_header=True)
        print(
            f"  {wildcards.aligner}: {unified.height} cCRE-assigned alignments "
            f"across {unified['query'].n_unique()} queries"
        )
