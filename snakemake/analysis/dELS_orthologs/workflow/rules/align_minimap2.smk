"""Minimap2 alignment of hg38 query dELS against the mm10 target window.

Uses `-x map-ont` rather than `asm5/10/20`. The asm presets have mismatch and
gap penalties tuned for ≤5% divergence between contig-scale assemblies; on
~20% divergent non-coding dELS pairs they reject all alignments. `map-ont`'s
noisy-read scoring (lower `-B` mismatch penalty, lower `-O,E` gap penalties)
tolerates that divergence and also uses a smaller k-mer (`-k 15` vs `-k 19`),
which is necessary to seed at all on 200-400 bp queries with ~20% error.

The issue body's reference to sweeping `-x asm5/asm10/asm20 + hand-tuned
scoring` remains the eventual goal; for this hello-world we just pick a
single preset that demonstrably recovers hits.
"""


rule minimap2_align:
    input:
        target="results/target/mm10_window.fasta",
        query="results/cre/hg38/query.filtered.fasta",
    output:
        paf="results/align/minimap2/raw.paf",
    threads: workflow.cores
    resources:
        mem_mb=4000,
    conda:
        "../envs/minimap2.yaml"
    shell:
        """
        minimap2 -cx map-ont -t {threads} --secondary=yes -N 50 \
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
        "results/align/minimap2/raw.paf",
    output:
        "results/align/minimap2/hits.tsv",
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
            f"  minimap2: {df.height} alignments across {df['query'].n_unique()} queries"
        )
