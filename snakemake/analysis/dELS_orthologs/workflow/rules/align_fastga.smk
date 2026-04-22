"""FASTGA pairwise alignment of hg38 query dELS against whole mm10.

FASTGA (Myers, [thegenemyers/FASTGA](https://github.com/thegenemyers/FASTGA))
is an adaptive-seed genome-vs-genome aligner; multi-threaded via `-T<N>`.

Three critical non-default flags:

- `-s100` — override FASTGA's default min alignment length of 1 kb, which
  rejects every query (hg38 cCREs are 200–500 bp).
- `-i.55` — fractional-identity floor at 0.55 (FASTGA's documented lower
  bound; the 0.7 default would miss ~75%-identity cross-species orthologs on
  the noisier tail).
- **Upper-cased FASTAs** — FASTGA's FAtoGDB crashes with `free(): invalid
  pointer` on soft-masked input (the UCSC 2bit-derived FASTAs have lowercase
  repeat bases). Pre-processing rules strip the masking for FASTGA only;
  upstream soft-masked filtering already happened in `filter_query_by_repeat`,
  so removing lowercase here doesn't re-admit repeat-heavy queries.

`-k` keeps FASTGA's `.1gdb` + `.gix` index files next to the input FASTAs
for reuse across flank-wildcard invocations.
"""


rule fastga_uppercase_target:
    """Uppercase-only copy of the mm10 target FASTA for FASTGA.

    FASTGA's FAtoGDB crashes on soft-masked input. The mmseqs2 / minimap2 /
    lastz paths keep the original soft-masking (which those aligners honour
    natively); only FASTGA needs the stripped copy.
    """
    input:
        "results/target/mm10_window.fasta",
    output:
        "results/target/mm10_window.upper.fasta",
    shell:
        "awk '/^>/ {{print; next}} {{print toupper($0)}}' {input} > {output}"


rule fastga_uppercase_query:
    """Uppercase-only copy of the hg38 per-flank query FASTA for FASTGA."""
    input:
        "results/cre/hg38/flank_{flank}/query.filtered.fasta",
    output:
        "results/cre/hg38/flank_{flank}/query.filtered.upper.fasta",
    wildcard_constraints:
        flank=r"-?\d+",
    shell:
        "awk '/^>/ {{print; next}} {{print toupper($0)}}' {input} > {output}"


rule fastga_align:
    input:
        target="results/target/mm10_window.upper.fasta",
        query="results/cre/hg38/flank_{flank}/query.filtered.upper.fasta",
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
            -s100 -i.55 \
            {input.query} {input.target} \
            > {output.paf}
        """


rule normalize_fastga_hits:
    """Project FASTGA PAF output to the aligner-agnostic unified schema.

    Unified schema (tab-separated, header):
        query  hit_chrom  hit_start  hit_end  rev_strand  score  fident  evalue  qcov  tcov

    PAF columns are 0-based half-open, consistent with the minimap2 PAF path.
    `score` uses the matches count (col 10), same as `normalize_minimap2_hits`,
    so score columns are directly comparable across aligners for ranking.

    Two FASTGA-specific PAF quirks vs the minimap2 PAF parser:
    - `fident` comes from the `dv:f:<divergence>` tag (fident = 1 - dv), not
      from `matches / alnlen` — FASTGA's "alignment block length" column
      roughly double-counts (e.g. a 340 bp alignment reports alnlen ≈ 687),
      so `matches / alnlen` would underestimate identity by ~2×.
    - Query output record names are `<accession>.<int>` (FASTGA appends a
      subsequence index when processing multi-record FASTAs); we strip the
      trailing `.N` to recover the original cCRE accession.
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
                qname_raw = p[0]
                # FASTGA appends `.<int>` to record names; strip it.
                qname = qname_raw.rsplit(".", 1)[0] if "." in qname_raw else qname_raw
                qlen = int(p[1])
                qstart, qend = int(p[2]), int(p[3])
                strand = p[4]
                tname_raw = p[5]
                # target names may also gain a .<int> suffix — strip for chroms too.
                tname = tname_raw.rsplit(".", 1)[0] if tname_raw.startswith("chr") and "." in tname_raw else tname_raw
                tlen = int(p[6])
                tstart, tend = int(p[7]), int(p[8])
                matches = int(p[9])
                # fident from dv:f:<float> tag; fall back to matches/alnlen.
                fident: float | None = None
                for tag in p[12:]:
                    if tag.startswith("dv:f:"):
                        fident = 1.0 - float(tag[5:])
                        break
                if fident is None:
                    alnlen = int(p[10])
                    fident = matches / alnlen if alnlen else 0.0
                rows.append(
                    {
                        "query": qname,
                        "hit_chrom": tname,
                        "hit_start": win_start + tstart,
                        "hit_end": win_start + tend,
                        "rev_strand": strand == "-",
                        "score": matches,
                        "fident": fident,
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
