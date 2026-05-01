"""Download phyloP_447m bigWig.

URL comes from ``bolinas.evals.conservation.CONSERVATION_TRACKS`` — single
source of truth shared with ``conservation_eval``.
"""


rule download_bigwig:
    output:
        "results/bigwig/{track}.bw",
    params:
        url=lambda wc: CONSERVATION_TRACKS[wc.track],
    wildcard_constraints:
        track="|".join(CONSERVATION_TRACKS),
    shell:
        "wget -q {params.url} -O {output}"
