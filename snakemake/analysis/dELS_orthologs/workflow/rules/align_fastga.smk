"""FASTGA pairwise alignment of hg38 query dELS against whole mm10.

FASTGA (Myers, [thegenemyers/FASTGA](https://github.com/thegenemyers/FASTGA))
is an adaptive-seed genome-vs-genome aligner; multi-threaded via `-T<N>`.

Critical parameter override: FASTGA's default `-s 1000` requires alignments
of ≥1 kb, which rejects every query in this pipeline (hg38 cCREs are typically
200–500 bp). Lowering to `-s 100` allows the real 300–500 bp alignments
through. Also `-i 0.5` loosens the fractional-identity floor from 0.7 to 0.5
to match ~75% identity in cross-species conserved regions; default would
drop some true orthologs on the noisier end of the distribution.

`-k` keeps the FASTGA index (.1gdb + .gix) next to the input FASTAs so
subsequent flank-wildcard re-invocations on the same query FASTA reuse them.
"""


rule fastga_align:
    input:
        target="results/target/mm10_window.fasta",
        query="results/cre/hg38/flank_{flank}/query.filtered.fasta",
    output:
        paf="results/align/fastga/flank_{flank}/raw.paf",
    wildcard_constraints:
        flank=r"-?\d+",
    threads: workflow.cores
    resources:
        mem_mb=32000,
    conda:
        "../envs/fastga.yaml"
    shell:
        r"""
        FastGA -v -k -T{threads} -paf \
            -s 100 -i 0.5 \
            {input.query} {input.target} \
            > {output.paf}
        """


rule normalize_fastga_hits:
    """Project FASTGA PAF output to the aligner-agnostic unified schema.

    Unified schema (tab-separated, header):
        query  hit_chrom  hit_start  hit_end  rev_strand  score  fident  evalue  qcov  tcov

    PAF columns are 0-based half-open, consistent with the minimap2 PAF path.
    `score` uses the matches count (col 10), same convention as
    `normalize_minimap2_hits`, so the two aligners' score columns are
    directly comparable in the evaluation rules.
    """
    input:
        "results/align/fastga/flank_{flank}/raw.paf",
    output:
        "results/align/fastga/flank_{flank}/hits.tsv",
    wildcard_constraints:
        flank=r"-?\d+",
    run:
        _, win_start, _ = get_search_window("mm10")
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
        schema = {
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
        df = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
        df.write_csv(output[0], separator="\t", include_header=True)
        print(
            f"  fastga flank={wildcards.flank}: {df.height} alignments across "
            f"{df['query'].n_unique()} queries"
        )
