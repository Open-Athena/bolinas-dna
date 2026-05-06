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

    The trailing slash on the source URL makes `gcloud storage cp -r` copy
    the contents of the source dir into {output}, so HF model files land
    directly at results/checkpoints/{model}/ (no extra step-N/ subdir).

    Auth: relies on `gcloud auth application-default login` (user-side) or
    GCE/EC2 default credentials. We don't add the snakemake-gcs storage
    plugin — a single shell rule keeps the pipeline simple and matches the
    explicit-download style of `download_genome`.
    """
    output:
        directory("results/checkpoints/{model}"),
    wildcard_constraints:
        model="|".join(MODELS),
    params:
        gcs_path=lambda wc: get_model_config(wc.model)["gcs_path"],
    shell:
        "gcloud storage cp -r {params.gcs_path}/ {output}"
