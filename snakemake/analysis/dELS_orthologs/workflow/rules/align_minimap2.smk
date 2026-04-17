"""Minimap2 alignment variants of hg38 query dELS against the mm10 target.

Each entry in `config["minimap2_variants"]` (keyed by aligner name matching
the `minimap2_.+` wildcard) maps to a full flag string. Downstream rules
(`normalize_minimap2_hits`, `per_query_report`, `recall_by_conservation`)
fan out automatically on the same `{aligner}` wildcard.

See issue #120 for the default / tuning hypotheses. The `-x map-ont` preset
was chosen as the initial baseline because asm5/10/20 defaults reject all
hg38↔mm10 dELS alignments at ~20% divergence on 200–400 bp queries.
"""


rule minimap2_align:
    input:
        target="results/target/mm10_window.fasta",
        query="results/cre/hg38/query.filtered.fasta",
    output:
        paf="results/align/{aligner}/raw.paf",
    wildcard_constraints:
        aligner="minimap2_.+",
    params:
        flags=lambda wildcards: config["minimap2_variants"][wildcards.aligner],
    threads: workflow.cores
    resources:
        mem_mb=8000,
    conda:
        "../envs/minimap2.yaml"
    shell:
        """
        minimap2 {params.flags} -t {threads} \
            {input.target} {input.query} > {output.paf}
        """


rule normalize_minimap2_hits:
    """Project minimap2 PAF to the aligner-agnostic unified schema.

    Unified schema (tab-separated, with header):
        query  hit_chrom  hit_start  hit_end  rev_strand  score  fident  evalue  qcov  tcov

    Coordinates are absolute 0-based half-open mm10 BED coords. minimap2 PAF
    already uses 0-based half-open, so no off-by-one needed beyond adding the
    window start. `evalue` is null — minimap2 doesn't emit one.
    """
    input:
        "results/align/{aligner}/raw.paf",
    output:
        "results/align/{aligner}/hits.tsv",
    wildcard_constraints:
        aligner="minimap2_.+",
    run:
        _, win_start, _ = get_search_window("mm10")

        # PAF columns (0-indexed): 0=qname 1=qlen 2=qstart 3=qend 4=strand
        # 5=tname 6=tlen 7=tstart 8=tend 9=matches 10=alnlen 11=mapq, then
        # optional SAM-style tags. We keep the first 12 fixed fields.
        # `hit_chrom` is taken from PAF tname (col 5) — equals the chromosome
        # name because make_target_window_bed names every record after its chrom.
        rows: list[dict] = []
        with open(input[0]) as f:
            for line in f:
                if not line.strip():
                    continue
                p = line.rstrip("\n").split("\t")
                qname = p[0]
                qlen = int(p[1])
                qstart, qend = int(p[2]), int(p[3])
                strand = p[4]
                tname = p[5]
                tlen = int(p[6])
                tstart, tend = int(p[7]), int(p[8])
                matches = int(p[9])
                alnlen = int(p[10])
                rows.append(
                    {
                        "query": qname,
                        "hit_chrom": tname,
                        "hit_start": win_start + tstart,
                        "hit_end": win_start + tend,
                        "rev_strand": strand == "-",
                        "score": matches,
                        "fident": matches / alnlen if alnlen else 0.0,
                        "evalue": None,
                        "qcov": (qend - qstart) / qlen if qlen else 0.0,
                        "tcov": (tend - tstart) / tlen if tlen else 0.0,
                    }
                )

        if rows:
            df = pl.DataFrame(
                rows,
                schema={
                    "query": pl.Utf8,
                    "hit_chrom": pl.Utf8,
                    "hit_start": pl.Int64,
                    "hit_end": pl.Int64,
                    "rev_strand": pl.Boolean,
                    "score": pl.Int64,
                    "fident": pl.Float64,
                    "evalue": pl.Float64,
                    "qcov": pl.Float64,
                    "tcov": pl.Float64,
                },
            )
        else:
            df = pl.DataFrame(
                schema={
                    "query": pl.Utf8,
                    "hit_chrom": pl.Utf8,
                    "hit_start": pl.Int64,
                    "hit_end": pl.Int64,
                    "rev_strand": pl.Boolean,
                    "score": pl.Int64,
                    "fident": pl.Float64,
                    "evalue": pl.Float64,
                    "qcov": pl.Float64,
                    "tcov": pl.Float64,
                }
            )
        df.write_csv(output[0], separator="\t", include_header=True)
        print(
            f"  {wildcards.aligner}: {df.height} alignments across "
            f"{df['query'].n_unique()} queries"
        )
