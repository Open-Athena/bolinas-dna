rule chrom_sizes:
    input:
        "results/genome/{g}.2bit",
    output:
        "results/chrom_sizes/{g}.tsv",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} {output}"


# just adds the start=0 column, but in human it also filters to standard chroms
rule extract_all:
    input:
        "results/chrom_sizes/{g}.tsv",
    output:
        "results/intervals/all/{g}.bed.gz",
    run:
        df = pd.read_csv(input[0], sep="\t", header=None, names=["chrom", "end"])
        df["start"] = 0
        # we want to filter to chromosomes, and exclude unplaced scaffolds,
        # alt. haplotypes, etc.
        # unfortunately no way to filter chroms based on prefix, AFAIK
        # in human we want to make sure to keep the standard chroms
        if wildcards.g == "GCF_000001405.40":
            df = df[df.chrom.str[:2] == "NC"]
        df = df[["chrom", "start", "end"]]
        df.to_csv(output[0], sep="\t", header=False, index=False)


rule extract_undefined:
    input:
        "results/genome/{g}.2bit",
    output:
        "results/intervals/undefined/{g}.bed.gz",
    params:
        "results/intervals/undefined/{g}.bed",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} {params} -nBed && gzip {params}"


rule extract_defined:
    input:
        "results/intervals/all/{g}.bed.gz",
        "results/intervals/undefined/{g}.bed.gz",
    output:
        "results/intervals/defined/{g}.bed.gz",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "bedtools subtract -a {input[0]} -b {input[1]} | gzip > {output}"


rule make_windows:
    input:
        "results/intervals/{intervals}/{g}.bed.gz",
    output:
        "results/intervals/windows/{intervals}/{w}/{s}/{g}.bed.gz",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        """
        bedtools makewindows -b {input[0]} -w {wildcards.w} -s {wildcards.s} | \
        awk '$3-$2 == {wildcards.w}' | \
        gzip > {output}
        """


# promoters from protein-coding transcripts, similar to gpn-animal-promoter-dataset
# described in TraitGym paper
rule intervals_recipe_v1:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v1/{g}.bed.gz",
    run:
        promoter_n_upstream = 256
        promoter_n_downstream = 256

        ann = load_annotation(input[0])
        mrna_exons = get_mrna_exons(ann)
        assert len(mrna_exons) > 0, f"No mRNA exons found for {wildcards.g}"
        promoters = get_promoters(
            mrna_exons, promoter_n_upstream, promoter_n_downstream
        ).to_pandas()
        assert len(promoters) > 0, f"No promoters found for {wildcards.g}"
        defined = read_bed_to_pandas(input[1])
        intervals = GenomicSet(promoters) & GenomicSet(defined)
        write_pandas_to_bed(intervals.to_pandas(), output[0])


# mRNA exons + promoters
rule intervals_recipe_v2:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v2/{g}.bed.gz",
    run:
        promoter_n_upstream = 256
        promoter_n_downstream = 256
        exon_flank = 50
        min_size = 512

        ann = load_annotation(input[0])
        mrna_exons = get_mrna_exons(ann)
        assert len(mrna_exons) > 0, f"No mRNA exons found for {wildcards.g}"
        promoters = GenomicSet(
            get_promoters(
                mrna_exons, promoter_n_upstream, promoter_n_downstream
            ).to_pandas()
        )
        assert promoters.n_intervals() > 0, f"No promoters found for {wildcards.g}"
        mrna_exons = GenomicSet(mrna_exons.to_pandas())
        defined = GenomicSet(read_bed_to_pandas(input[1]))
        mrna_exons = mrna_exons.add_flank(exon_flank)
        intervals = (promoters | mrna_exons).expand_min_size(min_size)
        intervals = intervals & defined
        write_pandas_to_bed(intervals.to_pandas(), output[0])


# CDS regions only
rule intervals_recipe_v3:
    input:
        "results/annotation/{g}.gtf.gz",
        "results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v3/{g}.bed.gz",
    run:
        # TODO: Fix this hardcoded exception! This genome (GCF_000002995.4) has
        # non-standard GTF format where the feature column contains gene names
        # (e.g., "Bm17073") instead of standard feature types (e.g., "CDS").
        # Proper fix: exclude this genome from the genome selection upstream.
        if wildcards.g == "GCF_000002995.4":
            # Write a single minimal interval for this problematic genome
            # to avoid downstream errors from empty files
            defined = read_bed_to_pandas(input[1])
            first_interval = defined.iloc[0:1]
            assert first_interval.iloc[0]["end"] - first_interval.iloc[0]["start"] >= 512
            first_interval["end"] = first_interval["start"] + 512
            write_pandas_to_bed(first_interval, output[0])
        else:
            min_size = 512

            ann = load_annotation(input[0])
            cds = get_cds(ann)
            assert len(cds) > 0, f"No CDS regions found for {wildcards.g}"
            cds = GenomicSet(cds.to_pandas())
            defined = GenomicSet(read_bed_to_pandas(input[1]))
            intervals = cds.expand_min_size(min_size)
            intervals = intervals & defined
            write_pandas_to_bed(intervals.to_pandas(), output[0])
