from bolinas.data.utils import load_annotation


rule annotation_download:
    output:
        "results/annotation.gtf.gz",
    params:
        url=config["annotation_url"],
    shell:
        "wget -O {output} {params.url}"


rule extract_tss:
    input:
        "results/annotation.gtf.gz",
    output:
        "results/intervals/tss.parquet",
    run:
        load_annotation(input[0]).pipe(get_tss).write_parquet(output[0])


rule extract_exon:
    input:
        "results/annotation.gtf.gz",
    output:
        "results/intervals/exon.parquet",
    run:
        load_annotation(input[0]).pipe(get_exon).write_parquet(output[0])
