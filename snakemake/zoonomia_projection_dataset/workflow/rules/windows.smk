"""Tile + N-filter the genome into fixed-length windows.

Combined into one rule. ``bedtools makewindows`` produces tiled windows from
the chrom-sizes file; we keep only those of exactly ``window_size`` (drops
the trailing partial windows on each chrom), then drop any window that
overlaps an undefined-region BED interval. Each window gets a sequential
``win_NNNNNNNNN`` name in column 4 (required by ``bigWigAverageOverBed``).
"""


rule make_windows:
    input:
        sizes="results/genome/{species}.chrom.sizes.filtered",
        undefined="results/genome/{species}.undefined.bed",
    output:
        "results/windows/{species}.bed.gz",
    conda:
        "../envs/bioinformatics.yaml"
    params:
        w=WINDOW_SIZE,
        s=STEP_SIZE,
    shell:
        r"""
        bedtools makewindows -g {input.sizes} -w {params.w} -s {params.s} \
          | awk -v w={params.w} 'BEGIN {{OFS="\t"}} $3 - $2 == w {{
              printf "%s\t%d\t%d\twin_%09d\n", $1, $2, $3, NR
          }}' \
          | bedtools intersect -a stdin -b {input.undefined} -v \
          | gzip > {output}
        """
