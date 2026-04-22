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
    """Per-cCRE assignment via CIGAR-exact lift from whole-genome PAF.

    minimap2 `-c` emits CIGAR in the `cg:Z:` PAF tag. Walk the CIGAR with
    `bolinas.lift.cigar_lift` to compute exact mm10 coords for each hg38
    cCRE contained in each alignment. See align_fastga_genome.smk for the
    rationale on why exact (not proportional) lift matters here.
    """
    input:
        paf="results/align/{aligner}/flank_0/raw.paf",
        query_cres="results/cre/hg38/query.filtered.parquet",
    output:
        "results/align/{aligner}/flank_0/hits.tsv",
    wildcard_constraints:
        aligner=r"minimap2_genome(_.*)?",
    run:
        from bolinas.lift import cigar_lift

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
                alnlen = int(p[10])
                cigar = ""
                for tag in p[12:]:
                    if tag.startswith("cg:Z:"):
                        cigar = tag[5:]
                        break
                paf_rows.append({
                    "hg38_chrom": qname, "hg38_start": qstart, "hg38_end": qend,
                    "strand": strand, "hit_chrom": tname, "hit_start": tstart, "hit_end": tend,
                    "score": matches,
                    "fident": matches / alnlen if alnlen else 0.0,
                    "cigar": cigar,
                })
        print(f"  {wildcards.aligner}: {len(paf_rows)} PAF rows, "
              f"{sum(1 for r in paf_rows if r['cigar'])} with CIGAR")

        schema_unified = {
            "query": pl.Utf8, "hit_chrom": pl.Utf8, "hit_start": pl.Int64,
            "hit_end": pl.Int64, "rev_strand": pl.Boolean, "score": pl.Int64,
            "fident": pl.Float64, "evalue": pl.Float64, "qcov": pl.Float64,
            "tcov": pl.Float64,
        }
        cres = pl.read_parquet(input.query_cres).select("chrom", "start", "end", "accession")
        if not paf_rows:
            unified = pl.DataFrame(schema=schema_unified)
        else:
            paf_df_for_overlap = pl.DataFrame([
                {"hg38_chrom": r["hg38_chrom"], "hg38_start": r["hg38_start"],
                 "hg38_end": r["hg38_end"], "_paf_idx": i}
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
                paf = paf_rows[row._paf_idx]
                if not paf["cigar"]:
                    continue
                lifted = cigar_lift(
                    q_start=paf["hg38_start"], q_end=paf["hg38_end"],
                    t_start=paf["hit_start"], t_end=paf["hit_end"],
                    strand=paf["strand"], cigar=paf["cigar"],
                    lift_q_start=row.start_cre, lift_q_end=row.end_cre,
                )
                if lifted is None:
                    continue
                mt_start, mt_end = lifted
                aln_qlen = paf["hg38_end"] - paf["hg38_start"]
                score_scaled = int(paf["score"] * (row.end_cre - row.start_cre) / max(1, aln_qlen))
                out_rows.append({
                    "query": row.accession_cre,
                    "hit_chrom": paf["hit_chrom"],
                    "hit_start": mt_start,
                    "hit_end": mt_end,
                    "rev_strand": paf["strand"] == "-",
                    "score": score_scaled,
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
