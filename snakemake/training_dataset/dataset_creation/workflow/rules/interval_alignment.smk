"""Generalized interval projection across genomes via mmseqs2 nucleotide search.

Produces `results/intervals/{name}/{g}.parquet` for each genome `g`. On the
source genome, the file is a chrom-normalized copy of the configured source
parquet. On target genomes, the file is the result of aligning the source
intervals onto `g` with mmseqs2 (nucleotide mode) and keeping the best hit
per query by bit score.

Configured via the `interval_mappings` block in config.yaml. For v1 every
mapping uses `HUMAN_GENOME` as its source; the wildcard constraints below
assume that, and should be generalized if more source genomes are added.

Choice of aligner (mmseqs2 over minimap2) follows issue #120, where mmseqs2
at `-s 7.5` is Pareto-optimal on the low-cost end of the recall-vs-compute
frontier for hg38↔mm10 cCRE orthologs (~70% R@1 at ~97% P@1 on the
phyloP_241m-conserved subset, vs. minimap2's ~40% R@1).
"""

MAPPINGS = {m["name"]: m for m in config.get("interval_mappings", [])}
_MAPPING_NAMES = "|".join(MAPPINGS.keys()) if MAPPINGS else "__never_matches__"


rule genome_fa:
    """Whole-genome FASTA; required as the mmseqs2 target input."""
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
    be traced back to their query.
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


rule mmseqs2_target_db:
    """Build a reusable mmseqs2 nucleotide DB from the target genome FASTA.

    Factored per-target (not per-mapping) so multiple mappings with the
    same `{g}` share one targetDB. `--mask-lower-case 1` stores soft-masked
    flags; the search rule respects them. Outputs the `.dbtype` sentinel so
    downstream rules can depend on the prefix without listing every sidecar
    file mmseqs2 produces. The explicit `mkdir -p` is needed because with
    the S3 storage backend Snakemake doesn't materialize the local parent
    directory for a `send to storage` output before the shell runs, so
    mmseqs2's first `.source` write would otherwise fail.
    """
    input:
        fasta="results/genome/{g}.fa",
    output:
        dbtype="results/interval_alignment/mmseqs2_target_db/{g}/targetDB.dbtype",
    params:
        prefix="results/interval_alignment/mmseqs2_target_db/{g}/targetDB",
    threads: 1
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mkdir -p $(dirname {params.prefix})
        mmseqs createdb {input.fasta} {params.prefix} --mask-lower-case 1
        """


rule mmseqs2_query_db:
    """Build an mmseqs2 nucleotide DB from the per-mapping query FASTA."""
    input:
        fasta="results/interval_alignment/{name}/{g}.query.fa",
    output:
        dbtype="results/interval_alignment/{name}/queryDB/{g}/queryDB.dbtype",
    params:
        prefix="results/interval_alignment/{name}/queryDB/{g}/queryDB",
    wildcard_constraints:
        name=_MAPPING_NAMES,
    threads: 1
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mkdir -p $(dirname {params.prefix})
        mmseqs createdb {input.fasta} {params.prefix} --mask-lower-case 1
        """


rule align_intervals_mmseqs2:
    """Run mmseqs2 nucleotide search of the query DB against the target DB.

    Flags mirror snakemake/analysis/dELS_orthologs (issue #120):
    - `--search-type 3` forces nucleotide mode (no auto-detect surprises).
    - `--strand 2` searches forward + reverse complement targets.
    - `--mask-lower-case 1` excludes soft-masked repeats from k-mer seeding.
    - `-s` (sensitivity) and `--max-accept` come from the mapping config.
    - `--split-memory-limit` lets the rule fit a small-RAM box at the cost
      of wall time; whole-mm10 nucleotide search needs ~50-80 GB resident
      unsplit, so large targets require either a big instance or a small
      split.
    """
    input:
        query_dbtype="results/interval_alignment/{name}/queryDB/{g}/queryDB.dbtype",
        target_dbtype="results/interval_alignment/mmseqs2_target_db/{g}/targetDB.dbtype",
    output:
        result_index="results/interval_alignment/{name}/{g}.resultDB.index",
        result_dbtype="results/interval_alignment/{name}/{g}.resultDB.dbtype",
    params:
        query_prefix="results/interval_alignment/{name}/queryDB/{g}/queryDB",
        target_prefix="results/interval_alignment/mmseqs2_target_db/{g}/targetDB",
        result_prefix="results/interval_alignment/{name}/{g}.resultDB",
        tmp_dir="results/interval_alignment/{name}/{g}.mmseqs2_tmp",
        sensitivity=lambda w: MAPPINGS[w.name].get("sensitivity", 7.5),
        max_accept=lambda w: MAPPINGS[w.name].get("max_accept", 1),
        split_memory_limit=lambda w: MAPPINGS[w.name].get("split_memory_limit", "12G"),
    wildcard_constraints:
        name=_MAPPING_NAMES,
    threads: workflow.cores
    resources:
        mem_mb=lambda w: MAPPINGS[w.name].get("mem_mb", 14000),
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mkdir -p {params.tmp_dir}
        mmseqs search \
            {params.query_prefix} \
            {params.target_prefix} \
            {params.result_prefix} \
            {params.tmp_dir} \
            --search-type 3 \
            --strand 2 \
            --mask-lower-case 1 \
            --split-memory-limit {params.split_memory_limit} \
            -s {params.sensitivity} \
            --max-accept {params.max_accept} \
            --threads {threads}
        rm -rf {params.tmp_dir}
        """


rule convertalis_mmseqs2:
    """Convert the mmseqs2 resultDB to a TSV of per-hit alignments."""
    input:
        query_dbtype="results/interval_alignment/{name}/queryDB/{g}/queryDB.dbtype",
        target_dbtype="results/interval_alignment/mmseqs2_target_db/{g}/targetDB.dbtype",
        result_index="results/interval_alignment/{name}/{g}.resultDB.index",
    output:
        tsv=temp("results/interval_alignment/{name}/{g}.hits.tsv"),
    params:
        query_prefix="results/interval_alignment/{name}/queryDB/{g}/queryDB",
        target_prefix="results/interval_alignment/mmseqs2_target_db/{g}/targetDB",
        result_prefix="results/interval_alignment/{name}/{g}.resultDB",
    wildcard_constraints:
        name=_MAPPING_NAMES,
    threads: 1
    conda:
        "../envs/mmseqs2.yaml"
    shell:
        """
        mmseqs convertalis \
            {params.query_prefix} \
            {params.target_prefix} \
            {params.result_prefix} \
            {output.tsv} \
            --format-output "query,target,tstart,tend,bits,evalue,fident,qcov,tcov"
        """


rule project_intervals_mmseqs2:
    """Project mmseqs2 hits to target-coord intervals and keep one per query."""
    input:
        tsv="results/interval_alignment/{name}/{g}.hits.tsv",
    output:
        "results/intervals/{name}/{g}.parquet",
    wildcard_constraints:
        name=_MAPPING_NAMES,
        g=f"(?!{HUMAN_GENOME}).*",
    run:
        from bolinas.alignment.mmseqs2 import (
            best_hit_per_query,
            parse_mmseqs2_hits,
            project_hits_to_intervals,
        )

        hits = parse_mmseqs2_hits(input.tsv)
        projected = project_hits_to_intervals(hits)
        best = best_hit_per_query(projected)
        (
            best.select(["chrom", "start", "end"])
            .sort(["chrom", "start", "end"])
            .write_parquet(output[0])
        )
        n_queries = projected["query"].n_unique() if projected.height else 0
        print(
            f"  {wildcards.name} → {wildcards.g}: "
            f"{best.height}/{n_queries} queries mapped "
            f"({projected.height} total alignments)"
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
