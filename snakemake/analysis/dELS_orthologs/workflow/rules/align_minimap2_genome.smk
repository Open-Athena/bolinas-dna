"""minimap2 in its genome-vs-genome regime: whole hg38 vs whole mm10.

Parallels `align_fastga_genome.smk`. minimap2's native asm-vs-asm preset
is `-x asm20` (up to ~20% divergence) — the right tool-documented choice
for hg38↔mm10. Unlike the cCRE-queries minimap2 variant, chains are now
formed across genome-scale context (flanking conservation anchors aid
placement of cCRE-scale alignments within longer syntenic blocks).

Output PAF → post-hoc intersected with the hg38 cCRE BED in
`normalize_minimap2_genome_hits`. Same unified hits schema as every other
aligner, so eval runs unchanged.
"""


rule minimap2_genome_align:
    input:
        target="results/target/mm10_window.fasta",
        query="results/genome/hg38_wholegenome.fasta",
    output:
        paf="results/align/{aligner}/flank_0/raw.paf",
    wildcard_constraints:
        aligner=r"minimap2_genome(_.*)?",
    params:
        flags=lambda wc: config.get("minimap2_variants", {}).get(wc.aligner, "-cx asm20 --secondary=yes -N 50"),
    threads: workflow.cores
    resources:
        mem_mb=64000,
    conda:
        "../envs/minimap2.yaml"
    shell:
        r"""
        minimap2 {params.flags} -t {threads} \
            {input.target} {input.query} > {output.paf}
        """


rule normalize_minimap2_genome_hits:
    """Post-hoc cCRE assignment: intersect each alignment's hg38 interval with
    every overlapping hg38 cCRE, emit one row per (cCRE, alignment). Top-k
    ranking inside per_query_report picks the best hit per cCRE.
    """
    input:
        paf="results/align/{aligner}/flank_0/raw.paf",
        query_cres="results/cre/hg38/query.filtered.parquet",
    output:
        "results/align/{aligner}/flank_0/hits.tsv",
    wildcard_constraints:
        aligner=r"minimap2_genome(_.*)?",
    run:
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
                alnlen = int(p[10])
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
                        "fident": matches / alnlen if alnlen else 0.0,
                        "qcov": 0.0,  # fraction of whole-chrom query covered — meaningless here
                        "tcov": 0.0,
                    }
                )
        schema = {
            "hg38_chrom": pl.Utf8, "hg38_start": pl.Int64, "hg38_end": pl.Int64,
            "rev_strand": pl.Boolean, "hit_chrom": pl.Utf8, "hit_start": pl.Int64,
            "hit_end": pl.Int64, "score": pl.Int64, "fident": pl.Float64,
            "qcov": pl.Float64, "tcov": pl.Float64,
        }
        paf_df = pl.DataFrame(paf_rows, schema=schema) if paf_rows else pl.DataFrame(schema=schema)

        # Proportional-lift each cCRE's hg38 sub-interval → mm10 sub-interval.
        # See the rationale in align_fastga_genome.smk: long genome-scale
        # alignments otherwise credit the whole block's mm10 span to every
        # overlapping cCRE and tank precision.
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
            out_rows: list[dict] = []
            for row in j.itertuples(index=False):
                lifted = proportional_lift(
                    qs=row.hg38_start, qe=row.hg38_end,
                    ts=row.hit_start, te=row.hit_end,
                    rev=bool(row.rev_strand),
                    cs=row.start_cre, ce=row.end_cre,
                )
                if lifted is None:
                    continue
                mt_start, mt_end = lifted
                ccre_len = row.end_cre - row.start_cre
                aln_qlen = row.hg38_end - row.hg38_start
                score_scaled = int(round(row.score * min(ccre_len, aln_qlen) / aln_qlen)) if aln_qlen else 0
                out_rows.append({
                    "query": row.accession_cre,
                    "hit_chrom": row.hit_chrom,
                    "hit_start": mt_start,
                    "hit_end": mt_end,
                    "rev_strand": bool(row.rev_strand),
                    "score": score_scaled,
                    "fident": float(row.fident),
                    "evalue": None,
                    "qcov": 1.0,
                    "tcov": 0.0,
                })
            schema_unified = {
                "query": pl.Utf8, "hit_chrom": pl.Utf8, "hit_start": pl.Int64,
                "hit_end": pl.Int64, "rev_strand": pl.Boolean, "score": pl.Int64,
                "fident": pl.Float64, "evalue": pl.Float64, "qcov": pl.Float64,
                "tcov": pl.Float64,
            }
            unified = pl.DataFrame(out_rows, schema=schema_unified) if out_rows else pl.DataFrame(schema=schema_unified)
        unified.write_csv(output[0], separator="\t", include_header=True)
        print(
            f"  {wildcards.aligner}: {unified.height} cCRE-assigned alignments "
            f"across {unified['query'].n_unique()} queries "
            f"(from {paf_df.height} PAF rows)"
        )
