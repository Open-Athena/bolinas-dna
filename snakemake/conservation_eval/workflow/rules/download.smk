rule download_bigwig:
    output:
        "results/conservation/{score}.bw",
    params:
        url=lambda wc: CONSERVATION_TRACKS[wc.score],
    wildcard_constraints:
        score="|".join(CONSERVATION_TRACKS),
    shell:
        "wget -q {params.url} -O {output}"
