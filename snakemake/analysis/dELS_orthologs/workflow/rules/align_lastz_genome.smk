"""lastz in its native regime: per-chromosome-pair hg38 vs mm10, with cCRE
lift via CIGAR (no chain/net yet).

Structure mirrors `align_fastga_genome.smk`:
- Extract per-chromosome FASTAs for both hg38 and mm10.
- Fan out lastz across (hg38_chrom × mm10_chrom) pairs (24 × 21 = 504 jobs)
  to parallelize the single-threaded lastz.
- Output lastz's `general` format with `cigarx` field (=/X per-base ops).
- Normalize: for each lastz alignment row, intersect its hg38 interval with
  the hg38 cCRE BED; `cigar_lift` each overlapping cCRE's hg38 span to a
  mm10 sub-interval.

This is the **raw-HSPs-without-chain/net** pairwise lastz. The full UCSC
pipeline (axtChain → chainMergeSort → chainNet → liftOver) can be added
later if this baseline leaves headroom.

Scoring: same HOXD70 matrix and UCSC-divergent-preset flags as the
cCRE-queries lastz rule (hspthresh=2200, ydrop=3400, gap=400,30,
seed=12of19, --notransition, --chain).
"""


HG38_LASTZ_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


rule extract_hg38_chrom_fasta:
    """Per-chromosome hg38 FASTA (for the genome-vs-genome lastz path)."""
    input:
        "results/genome/hg38_wholegenome.fasta",
    output:
        "results/genome/hg38_by_chrom/{chrom}.fasta",
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


rule lastz_genome_align_pair:
    """lastz one hg38 chrom × one mm10 chrom, output general-format with cigarx.

    Field order (11 cols): name1 (mm10 chrom), zstart1, end1, strand1,
    name2 (hg38 chrom), zstart2, end2, strand2, score, length2 (hg38 len),
    cigarx (lastz CIGAR with = / X ops).
    """
    input:
        target="results/target/by_chrom/{mm10_chrom}.fasta",
        query="results/genome/hg38_by_chrom/{hg38_chrom}.fasta",
        scores="workflow/resources/HOXD70.Q",
    output:
        temp("results/align/lastz_genome/flank_0/pairs/{hg38_chrom}__{mm10_chrom}.general"),
    wildcard_constraints:
        hg38_chrom=r"chr[0-9XY]+",
        mm10_chrom=r"chr[0-9XY]+",
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
            --format='general-:name1,zstart1,end1,strand1,name2,zstart2,end2,strand2,score,length2,cigarx' \
            --output={output}
        """


rule normalize_lastz_genome_hits:
    """Concatenate per-chrom-pair lastz outputs and lift to per-cCRE hits.

    For each lastz row (hg38 span = name2 / zstart2 / end2 / strand2,
    mm10 span = name1 / zstart1 / end1, always target '+' strand here),
    intersect with hg38 cCREs and CIGAR-lift each overlapping cCRE's
    hg38 span to its exact mm10 sub-interval.

    lastz general-format coords are 0-based half-open when we use
    zstart1/zstart2 (the `z` prefix); no off-by-one conversion needed.
    """
    input:
        lambda wc: expand(
            "results/align/lastz_genome/flank_0/pairs/{hg38_chrom}__{mm10_chrom}.general",
            hg38_chrom=HG38_LASTZ_CHROMS,
            mm10_chrom=MM10_STANDARD_CHROMS,
        ),
        query_cres="results/cre/hg38/query.filtered.parquet",
    output:
        "results/align/lastz_genome/flank_0/hits.tsv",
    run:
        from bolinas.lift import cigar_lift

        # lastz general-format row (per our `--format=general-:...`):
        #   0: name1 (mm10 chrom)    1: zstart1   2: end1      3: strand1
        #   4: name2 (hg38 chrom)    5: zstart2   6: end2      7: strand2
        #   8: score                 9: length2 (hg38 alignment length)
        #  10: cigarx (='ed CIGAR with = / X / I / D ops)

        paf_rows: list[dict] = []
        # All per-pair files are in input.* (list, via expand's lambda).
        for fp in input[:-1]:  # last element is query_cres
            with open(fp) as f:
                for line in f:
                    if not line.strip():
                        continue
                    p = line.rstrip("\n").split("\t")
                    if len(p) < 11:
                        continue
                    mm10_chrom = p[0]
                    mm10_start, mm10_end = int(p[1]), int(p[2])
                    mm10_strand = p[3]
                    hg38_chrom = p[4]
                    hg38_start, hg38_end = int(p[5]), int(p[6])
                    hg38_strand = p[7]
                    score = int(p[8])
                    # Combined relative orientation for the lift step below —
                    # lastz-general tells us strand1 (mm10) and strand2 (hg38);
                    # we pass target always +, so the alignment is "reverse"
                    # iff mm10_strand != hg38_strand.
                    rev = (mm10_strand != hg38_strand)
                    cigar = p[10]
                    paf_rows.append({
                        "hg38_chrom": hg38_chrom, "hg38_start": hg38_start,
                        "hg38_end": hg38_end, "hit_chrom": mm10_chrom,
                        "hit_start": mm10_start, "hit_end": mm10_end,
                        "rev": rev, "score": score, "cigar": cigar,
                    })
        print(f"  lastz_genome: {len(paf_rows)} raw HSPs across 24×21 pairs")

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
                lifted = cigar_lift(
                    q_start=paf["hg38_start"], q_end=paf["hg38_end"],
                    t_start=paf["hit_start"], t_end=paf["hit_end"],
                    strand="-" if paf["rev"] else "+",
                    cigar=paf["cigar"],
                    lift_q_start=row.start_cre, lift_q_end=row.end_cre,
                )
                if lifted is None:
                    continue
                mt_start, mt_end = lifted
                aln_qlen = max(1, paf["hg38_end"] - paf["hg38_start"])
                score_scaled = int(paf["score"] * (row.end_cre - row.start_cre) / aln_qlen)
                out_rows.append({
                    "query": row.accession_cre,
                    "hit_chrom": paf["hit_chrom"],
                    "hit_start": mt_start,
                    "hit_end": mt_end,
                    "rev_strand": paf["rev"],
                    "score": score_scaled,
                    "fident": 0.0,  # lastz general-cigarx row doesn't emit per-aln identity; could derive from CIGAR later
                    "evalue": None,
                    "qcov": 1.0,
                    "tcov": 0.0,
                })
            unified = pl.DataFrame(out_rows, schema=schema_unified) if out_rows else pl.DataFrame(schema=schema_unified)

        unified.write_csv(output[0], separator="\t", include_header=True)
        print(
            f"  lastz_genome: {unified.height} cCRE-assigned alignments "
            f"across {unified['query'].n_unique()} queries"
        )
