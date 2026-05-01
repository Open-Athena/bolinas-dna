"""Tile + N-filter the genome into fixed-length windows, per chromosome.

Per-chrom outputs because score_windows_chrom fans out by ``{chrom}`` and
having each worker load only its own ~30 MB BED (instead of the full
700 MB) is the difference between fitting in 16 GB and OOMing on
``c6id.2xlarge``.

``bedtools makewindows`` is invoked on a single-row chrom_sizes file so it
emits windows for one chrom only. Window names embed the chrom so they're
globally unique across per-chrom BEDs (downstream concatenation needs that).
"""


rule make_windows:
    """255 bp windows on a single chromosome, dropping any that overlap N regions."""
    input:
        sizes="results/genome/{species}.chrom.sizes.filtered",
        undefined="results/genome/{species}.undefined.bed",
    output:
        "results/windows/{species}/{chrom}.bed.gz",
    wildcard_constraints:
        chrom="|".join(STANDARD_CHROMS),
    conda:
        "../envs/bioinformatics.yaml"
    params:
        w=WINDOW_SIZE,
        s=STEP_SIZE,
    shell:
        r"""
        TMPSIZES=$(mktemp)
        trap "rm -f $TMPSIZES" EXIT
        awk -v c={wildcards.chrom} '$1 == c' {input.sizes} > $TMPSIZES
        bedtools makewindows -g $TMPSIZES -w {params.w} -s {params.s} \
          | awk -v w={params.w} -v c={wildcards.chrom} 'BEGIN {{OFS="\t"}}
              $3 - $2 == w {{ printf "%s\t%d\t%d\twin_%s_%09d\n", $1, $2, $3, c, NR }}' \
          | bedtools intersect -a stdin -b {input.undefined} -v \
          | gzip > {output}
        """
