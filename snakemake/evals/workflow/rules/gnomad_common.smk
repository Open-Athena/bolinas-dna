rule gnomad_download:
    output:
        "results/gnomad/all.parquet",
    params:
        url=lambda wc: f"https://huggingface.co/datasets/{config['gnomad_full_repo']}/resolve/main/test.parquet",
    shell:
        "wget -O {output} {params.url}"


rule gnomad_common:
    input:
        "results/gnomad/all.parquet",
    output:
        "results/gnomad/common.parquet",
    run:
        (
            pl.scan_parquet(input[0])
            .filter(
                pl.col("AN") >= config["gnomad_min_AN"],
                pl.col("AF") > config["gnomad_common_min_AF"],
            )
            .sink_parquet(output[0])
        )
