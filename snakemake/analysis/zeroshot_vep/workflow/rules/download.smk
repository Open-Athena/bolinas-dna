"""Download rules: reference genome + per-model checkpoint dirs from GCS.

Identical to evals_v2/workflow/rules/download.smk — kept inline so this
pipeline is self-contained and doesn't import from a sibling workflow.
"""


rule download_genome:
    output:
        "results/genome.fa.gz",
    params:
        url=config["genome_url"],
    shell:
        "wget {params.url} -O {output}"


rule download_model:
    """Pull a specific HF checkpoint dir from GCS into results/checkpoints/{model}/."""
    output:
        directory("results/checkpoints/{model}"),
    wildcard_constraints:
        model="|".join(MODELS),
    params:
        gcs_path=lambda wc: get_model_config(wc.model)["gcs_path"],
    shell:
        "mkdir -p {output} && "
        "gcloud storage cp -r '{params.gcs_path}/*' {output}/"
