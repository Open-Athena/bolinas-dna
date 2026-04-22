"""lastz pairwise alignment of hg38 query dELS against whole mm10.

lastz is single-threaded, so we parallelize across mm10 chromosomes — one rule
instance per `{chrom}`, fanned out at `{aligner}/flank_{flank}/by_chrom/`. The
concat+normalize step unions them into the aligner-agnostic unified schema
that per_query_report consumes.

Scoring: HOXD70 substitution matrix (divergent cross-species standard, the
same matrix UCSC uses for hg/mm chain/net). Lastz parameters follow the
divergent-DNA preset from the UCSC pipeline (hspthresh=2200, ydrop=3400,
gap=400,30, seed=12of19 with --notransition, --chain).
"""


rule extract_mm10_chrom_fasta:
    """Per-chromosome mm10 FASTA, split from the whole-genome target FASTA.

    Lastz itself has no chromosome-subset flag, and we need per-chrom inputs
    to parallelize. `faidx` would also work but we avoid the extra env since
    the target FASTA is already on disk from `extract_target_fasta`.
    """
    input:
        "results/target/mm10_window.fasta",
    output:
        "results/target/by_chrom/{chrom}.fasta",
    wildcard_constraints:
        chrom=r"chr[0-9XY]+",
    run:
        wanted = wildcards.chrom
        found = False
        with open(input[0]) as fin, open(output[0], "w") as fout:
            emit = False
            for line in fin:
                if line.startswith(">"):
                    name = line[1:].strip().split()[0]
                    emit = name == wanted
                    if emit:
                        found = True
                if emit:
                    fout.write(line)
        assert found, f"chromosome {wanted} not found in {input[0]}"


LASTZ_SCORING_MATRIX = {
    "lastz": "HOXD70.Q",
    "lastz_hoxd55": "HOXD55.Q",
}


rule lastz_align_per_chrom:
    """lastz one mm10 chromosome vs all queries at a given flank.

    The `{aligner}` wildcard selects the scoring matrix via `LASTZ_SCORING_MATRIX`:
    - `lastz` → HOXD70 (default divergent-species scoring)
    - `lastz_hoxd55` → HOXD55 (more permissive; slightly weaker mismatch penalties,
      used by UCSC for the most-divergent mammal pairs)

    Output format: `general-` (headerless tabular) with the subset of fields
    we normalize downstream. `zstart1` and `zstart2` are 0-based starts; paired
    `end1`/`end2` are 0-based half-open (so `end - zstart == length`). `name1`
    is the target chrom (mm10), `name2` is the query accession (hg38 cCRE).
    `strand1` is always '+' because we pass target without the `[revcomp]`
    modifier; `strand2` encodes the match orientation.
    """
    input:
        target="results/target/by_chrom/{chrom}.fasta",
        query="results/cre/hg38/flank_{flank}/query.filtered.fasta",
        scores=lambda wc: f"workflow/resources/{LASTZ_SCORING_MATRIX[wc.aligner]}",
    output:
        temp("results/align/{aligner}/flank_{flank}/by_chrom/{chrom}.general"),
    wildcard_constraints:
        aligner=r"lastz(_.*)?",
        chrom=r"chr[0-9XY]+",
        flank=r"-?\d+",
    threads: 1
    resources:
        mem_mb=8000,
    conda:
        "../envs/lastz.yaml"
    shell:
        r"""
        lastz '{input.target}[multiple]' '{input.query}[multiple]' \
            --scores={input.scores} \
            --ambiguous=iupac \
            --hspthresh=2200 \
            --ydrop=3400 \
            --gap=400,30 \
            --seed=12of19 \
            --notransition \
            --chain \
            --format='general-:name1,zstart1,end1,strand1,name2,zstart2,end2,strand2,length2,score,nmatch,ncolumn' \
            --output={output}
        """


rule normalize_lastz_hits:
    """Concatenate per-chromosome lastz outputs into the unified hits schema.

    Unified schema: query, hit_chrom, hit_start, hit_end, rev_strand, score,
    fident, evalue, qcov, tcov. Coordinates are 0-based half-open; `qcov` is
    aligned query columns / query length (fident's denominator `ncolumn`
    counts aligned columns including gaps; the numerator `nmatch` counts
    identical bases).
    """
    input:
        lambda wc: expand(
            "results/align/{aligner}/flank_{flank}/by_chrom/{chrom}.general",
            aligner=[wc.aligner],
            flank=[wc.flank],
            chrom=MM10_STANDARD_CHROMS,
        ),
    output:
        "results/align/{aligner}/flank_{flank}/hits.tsv",
    wildcard_constraints:
        aligner=r"lastz(_.*)?",
        flank=r"-?\d+",
    run:
        _, win_start, _ = get_search_window("mm10")
        rows: list[dict] = []
        for fp in input:
            with open(fp) as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line or line.startswith("#"):
                        continue
                    p = line.split("\t")
                    # name1, zstart1, end1, strand1, name2, zstart2, end2, strand2, length2, score, nmatch, ncolumn
                    hit_chrom = p[0]
                    zstart1 = int(p[1])
                    end1 = int(p[2])
                    strand1 = p[3]
                    query = p[4]
                    zstart2 = int(p[5])
                    end2 = int(p[6])
                    strand2 = p[7]
                    qlen = int(p[8])
                    score = int(p[9])
                    nmatch = int(p[10])
                    ncolumn = int(p[11])
                    rev = (strand1 != strand2)
                    rows.append(
                        {
                            "query": query,
                            "hit_chrom": hit_chrom,
                            "hit_start": win_start + zstart1,
                            "hit_end": win_start + end1,
                            "rev_strand": rev,
                            "score": score,
                            "fident": nmatch / ncolumn if ncolumn else 0.0,
                            "evalue": None,
                            "qcov": (end2 - zstart2) / qlen if qlen else 0.0,
                            "tcov": (end1 - zstart1) / (end1 - zstart1) if (end1 > zstart1) else 0.0,
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
            f"  {wildcards.aligner} flank={wildcards.flank}: {df.height} alignments "
            f"across {df['query'].n_unique()} queries"
        )
