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


rule download_annotation:
    """Ensembl release 115 GTF (matches genome_url release).

    Used by ``derive_subset_v2_tss_mrna`` to extract protein_coding TSSes
    for the v2 subset definition.
    """
    output:
        "results/annotation/Homo_sapiens.GRCh38.115.gtf.gz",
    params:
        url="https://ftp.ensembl.org/pub/release-115/gtf/homo_sapiens/Homo_sapiens.GRCh38.115.gtf.gz",
    shell:
        "wget -q -O {output} {params.url}"
