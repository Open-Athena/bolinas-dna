"""Per-step GCS download for `gcs_path`-backed model entries.

Mirrors the `evals_v2` pattern (``evals_v2/workflow/rules/download.smk``).
The `compute_scores` rule depends on this output for entries that specify
`gcs_path` in `config/config.yaml`; legacy `base_path` entries still resolve
the local checkpoint without a download step.
"""

GCS_MODELS = [m["name"] for m in config["models"] if "gcs_path" in m]


rule download_model_step:
    """Pull `{gcs_path}/step-{step}/` into `results/checkpoints/{model}/step-{step}/`.

    `gcloud storage cp -r src dst` nests src under dst, so we glob with `/*`
    to land HF model files directly under the output directory. Auth uses
    application-default credentials.
    """
    output:
        directory("results/checkpoints/{model}/step-{step}"),
    wildcard_constraints:
        model="|".join(GCS_MODELS) if GCS_MODELS else "(?!)",
    params:
        gcs_path=lambda wc: get_model_config(wc.model)["gcs_path"],
    shell:
        "mkdir -p {output} && "
        "gcloud storage cp -r '{params.gcs_path}/step-{wildcards.step}/*' {output}/"
