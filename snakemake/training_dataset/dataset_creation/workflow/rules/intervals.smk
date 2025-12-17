rule chrom_sizes:
    input:
        "results/genome/{g}.2bit",
    output:
        "results/chrom_sizes/{g}.tsv",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} {output}"


rule extract_all:
    input:
        "results/chrom_sizes/{g}.tsv",
    output:
        "results/intervals/all/{g}.bed.gz",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        """awk 'BEGIN {{OFS="\t"}} {{print $1, 0, $2}}' {input} | gzip > {output}"""


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
        "results/intervals/undefined/{g}.bed.gz",
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
        undefined = read_bed_to_pandas(input[1])
        intervals = GenomicSet(promoters) - GenomicSet(undefined)
        write_pandas_to_bed(intervals.to_pandas(), output[0])
