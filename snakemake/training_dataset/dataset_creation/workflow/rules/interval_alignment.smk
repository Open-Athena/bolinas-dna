"""Generalized interval projection across genomes via local alignment.

Produces `results/intervals/{name}/{g}.parquet` for each genome `g`. On the
source genome, the file is a chrom-normalized copy of the configured source
parquet. On target genomes, the file is the result of aligning the source
intervals onto `g` with minimap2 and keeping the best hit per query.

Configured via the `interval_mappings` block in config.yaml. For v1 every
mapping uses `HUMAN_GENOME` as its source; the wildcard constraints below
assume that, and should be generalized if more source genomes are added.
"""

MAPPINGS = {m["name"]: m for m in config.get("interval_mappings", [])}
_MAPPING_NAMES = "|".join(MAPPINGS.keys()) if MAPPINGS else "__never_matches__"


rule genome_fa:
    """Whole-genome FASTA; required as the minimap2 target."""
    input:
        "results/genome/{g}.2bit",
    output:
        "results/genome/{g}.fa",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitToFa {input} {output}"


rule intervals_source_unified:
    """Stage the source-genome intervals at the unified path.

    For CRE-derived sources, chroms are normalized from UCSC-stripped (bare
    digit, e.g. "1", "X") to the RefSeq NC_* form used by the per-genome 2bit
    files, via config/human_chrom_mapping.tsv. Same remap pattern as
    intervals_recipe_v17/v18 in intervals.smk.
    """
    input:
        source_parquet=lambda w: MAPPINGS[w.name]["source_parquet"],
        chrom_mapping=local("config/human_chrom_mapping.tsv"),
    output:
        f"results/intervals/{{name}}/{HUMAN_GENOME}.parquet",
    wildcard_constraints:
        name=_MAPPING_NAMES,
    run:
        cfg = MAPPINGS[wildcards.name]
        df = pl.read_parquet(input.source_parquet)
        if cfg.get("source_chrom_style") == "ucsc_stripped":
            chrom_map = pl.read_csv(input.chrom_mapping, separator="\t")
            simple_to_refseq = dict(
                zip(chrom_map["ucsc"].str.replace("chr", ""), chrom_map["refseq"])
            )
            df = df.with_columns(pl.col("chrom").replace_strict(simple_to_refseq))
        df.select(["chrom", "start", "end"]).write_parquet(output[0])


rule make_alignment_query_bed:
    """4-col BED from the source intervals with symmetric flank.

    Flank positive expands each side; flank 0 (v1 default) passes the
    intervals through unchanged. End is clamped to the chrom size; degenerate
    intervals (hi <= lo) are dropped.

    The 4th column is the source interval id (`chrom:start-end`) so hits can
    be traced back to their query in debug workflows.
    """
    input:
        source_parquet=lambda w: (
            f"results/intervals/{w.name}/"
            f"{MAPPINGS[w.name]['source_genome']}.parquet"
        ),
        chrom_sizes=lambda w: (
            f"results/chrom_sizes/{MAPPINGS[w.name]['source_genome']}.tsv"
        ),
    output:
        temp("results/interval_alignment/{name}/{g}.query.bed"),
    wildcard_constraints:
        name=_MAPPING_NAMES,
    run:
        cfg = MAPPINGS[wildcards.name]
        flank = int(cfg.get("flank_bp", 0))
        sizes: dict[str, int] = {}
        with open(input.chrom_sizes) as f:
            for line in f:
                c, s = line.rstrip("\n").split("\t")
                sizes[c] = int(s)
        df = pl.read_parquet(input.source_parquet)
        kept = 0
        with open(output[0], "w") as fout:
            for row in df.iter_rows(named=True):
                chrom = row["chrom"]
                lo = max(0, row["start"] - flank)
                hi = row["end"] + flank
                if chrom in sizes:
                    hi = min(sizes[chrom], hi)
                if hi <= lo:
                    continue
                name = f"{chrom}:{row['start']}-{row['end']}"
                fout.write(f"{chrom}\t{lo}\t{hi}\t{name}\n")
                kept += 1
        print(
            f"  {wildcards.name} → {wildcards.g}: {kept}/{df.height} queries "
            f"(flank={flank})"
        )


rule extract_alignment_query_fasta:
    """Per-query FASTA from the source 2bit."""
    input:
        twobit=lambda w: f"results/genome/{MAPPINGS[w.name]['source_genome']}.2bit",
        bed="results/interval_alignment/{name}/{g}.query.bed",
    output:
        temp("results/interval_alignment/{name}/{g}.query.fa"),
    wildcard_constraints:
        name=_MAPPING_NAMES,
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitToFa {input.twobit} {output} -bed={input.bed}"


rule align_intervals_minimap2:
    """Run minimap2 aligning the source-interval FASTAs against target genome."""
    input:
        query="results/interval_alignment/{name}/{g}.query.fa",
        target="results/genome/{g}.fa",
    output:
        "results/interval_alignment/{name}/{g}.paf.gz",
    wildcard_constraints:
        name=_MAPPING_NAMES,
    params:
        preset=lambda w: MAPPINGS[w.name]["preset"],
    threads: workflow.cores
    resources:
        mem_mb=16000,
    conda:
        "../envs/minimap2.yaml"
    shell:
        """
        minimap2 {params.preset} -t {threads} {input.target} {input.query} | \
            gzip > {output}
        """


rule project_intervals_minimap2:
    """Pick the best hit per query from the PAF and emit target-coord intervals."""
    input:
        paf="results/interval_alignment/{name}/{g}.paf.gz",
    output:
        "results/intervals/{name}/{g}.parquet",
    wildcard_constraints:
        name=_MAPPING_NAMES,
        g=f"(?!{HUMAN_GENOME}).*",
    run:
        import gzip
        import tempfile

        from bolinas.alignment.minimap2 import best_hit_per_query, parse_paf

        # parse_paf takes a text path; decompress inline to keep it simple.
        with gzip.open(input.paf, "rt") as fin:
            with tempfile.NamedTemporaryFile("w", suffix=".paf", delete=False) as fout:
                fout.write(fin.read())
                tmp_paf = fout.name

        df = parse_paf(tmp_paf)
        best = best_hit_per_query(df)
        (
            best.select(["chrom", "start", "end"])
            .sort(["chrom", "start", "end"])
            .write_parquet(output[0])
        )
        print(
            f"  {wildcards.name} → {wildcards.g}: "
            f"{best.height}/{df['query'].n_unique() if df.height else 0} "
            f"queries mapped ({df.height} total alignments)"
        )


rule mapped_intervals_parquet_to_recipe_bed:
    """Adapter: convert unified-namespace mapped parquet to the bed.gz path
    the existing dataset pipeline expects.

    Wildcard-constrained to configured mapping names, so this rule never
    collides with the legacy `intervals_recipe_v{N}` rules (v1..v19), which
    produce their `.bed.gz` directly.
    """
    input:
        "results/intervals/{name}/{g}.parquet",
    output:
        "results/intervals/recipe/{name}/{g}.bed.gz",
    wildcard_constraints:
        name=_MAPPING_NAMES,
    run:
        GenomicSet.read_parquet(input[0]).write_bed(output[0])
