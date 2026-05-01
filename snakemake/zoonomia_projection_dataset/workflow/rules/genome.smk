"""Genome download + 2bit + chrom_sizes + N-region BED.

Patterns copied from ``snakemake/enhancer_classification/workflow/rules/data.smk``
to keep this pipeline standalone (no cross-pipeline imports).
"""


rule download_genome:
    output:
        "results/genome/{species}.fa.gz",
    params:
        url=lambda wildcards: config["genome_urls"][wildcards.species],
    shell:
        "wget -q -O {output} {params.url}"


rule genome_to_2bit:
    input:
        "results/genome/{species}.fa.gz",
    output:
        "results/genome/{species}.2bit",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "zcat {input} | faToTwoBit stdin {output}"


rule chrom_sizes:
    input:
        "results/genome/{species}.2bit",
    output:
        "results/genome/{species}.chrom.sizes",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} {output}"


rule chrom_sizes_filtered:
    """Restrict to ``standard_chroms[{species}]`` (autosomes + X + Y for human)."""
    input:
        "results/genome/{species}.chrom.sizes",
    output:
        "results/genome/{species}.chrom.sizes.filtered",
    run:
        standard = config["standard_chroms"][wildcards.species]
        df = pd.read_csv(input[0], sep="\t", header=None, names=["chrom", "size"])
        df = df[df["chrom"].isin(standard)]
        assert len(df) == len(standard), (
            f"missing chroms in 2bit for {wildcards.species}: "
            f"{set(standard) - set(df['chrom'])}"
        )
        df.to_csv(output[0], sep="\t", header=False, index=False)


rule undefined_regions:
    """N-region BED. ``twoBitInfo -nBed`` reports stretches of undefined bases."""
    input:
        "results/genome/{species}.2bit",
    output:
        "results/genome/{species}.undefined.bed",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} /dev/stdout -nBed > {output}"
