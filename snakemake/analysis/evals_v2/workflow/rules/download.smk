"""Download rules: reference genome + per-model checkpoint dirs from GCS."""


rule download_genome:
    output:
        "results/genome.fa.gz",
    params:
        url=config["genome_url"],
    shell:
        "wget {params.url} -O {output}"


rule download_model:
    """Pull a specific checkpoint dir from GCS into results/checkpoints/{model}/.

    The `mkdir` + glob is to flatten: `gcloud storage cp -r src dst` nests
    src under dst, but the inference rule expects HF model files directly
    under {output}. Auth uses application-default credentials.
    """
    output:
        directory("results/checkpoints/{model}"),
    wildcard_constraints:
        model="|".join(MODELS),
    params:
        gcs_path=lambda wc: get_model_config(wc.model)["gcs_path"],
    shell:
        "mkdir -p {output} && "
        "gcloud storage cp -r '{params.gcs_path}/*' {output}/"
