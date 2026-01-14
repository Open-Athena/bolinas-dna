rule hf_upload:
    input:
        expand(
            "results/dataset/{{dataset}}/{split}.parquet",
            split=SPLITS,
        ),
    output:
        touch("results/upload.done/{dataset}"),
    params:
        repo_name=lambda wildcards: f"{config['output_hf_prefix']}-{wildcards.dataset}",
    threads: workflow.cores
    shell:
        "hf upload-large-folder {params.repo_name} --repo-type dataset results/dataset/{wildcards.dataset}"
