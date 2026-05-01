rule consequence_download:
    output:
        "results/consequences/{chrom}.parquet",
    params:
        url=lambda wc: f"https://huggingface.co/datasets/{config['consequences_repo']}/resolve/main/{wc.chrom}.parquet",
    shell:
        "wget -O {output} {params.url}"
