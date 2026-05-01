"""Genome download + 2bit + chrom_sizes + N-region BED for hg38.

Patterns copied from ``snakemake/enhancer_classification/workflow/rules/data.smk``
to keep this pipeline standalone (no cross-pipeline imports).
"""


rule download_genome:
    output:
        "results/genome/hg38.fa.gz",
    params:
        url=config["genome_url"],
    shell:
        "wget -q -O {output} {params.url}"


rule genome_to_2bit:
    input:
        "results/genome/hg38.fa.gz",
    output:
        "results/genome/hg38.2bit",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "zcat {input} | faToTwoBit stdin {output}"


rule chrom_sizes:
    input:
        "results/genome/hg38.2bit",
    output:
        "results/genome/hg38.chrom.sizes",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} {output}"


rule chrom_sizes_filtered:
    """Restrict to ``standard_chroms`` (autosomes + X + Y)."""
    input:
        "results/genome/hg38.chrom.sizes",
    output:
        "results/genome/hg38.chrom.sizes.filtered",
    run:
        df = pd.read_csv(input[0], sep="\t", header=None, names=["chrom", "size"])
        df = df[df["chrom"].isin(STANDARD_CHROMS)]
        assert len(df) == len(STANDARD_CHROMS), (
            f"missing chroms in 2bit: {set(STANDARD_CHROMS) - set(df['chrom'])}"
        )
        df.to_csv(output[0], sep="\t", header=False, index=False)


rule undefined_regions:
    """N-region BED. ``twoBitInfo -nBed`` reports stretches of undefined bases."""
    input:
        "results/genome/hg38.2bit",
    output:
        "results/genome/hg38.undefined.bed",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} /dev/stdout -nBed > {output}"
