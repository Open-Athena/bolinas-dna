"""Per-step GCS download for `gcs_path`-backed model entries."""

GCS_MODELS = [m["name"] for m in config["models"] if "gcs_path" in m]


if GCS_MODELS:

    rule download_model_step:
        output:
            directory("results/checkpoints/{model}/step-{step}"),
        wildcard_constraints:
            model="|".join(GCS_MODELS),
        params:
            gcs_path=lambda wc: get_model_config(wc.model)["gcs_path"],
        # `gcloud cp -r src dst` nests src under dst, so glob with /* to land
        # HF model files directly under {output}.
        shell:
            "mkdir -p {output} && "
            "gcloud storage cp -r '{params.gcs_path}/step-{wildcards.step}/*' {output}/"
