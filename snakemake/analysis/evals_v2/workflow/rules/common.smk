"""Common imports + helpers for evals_v2 rules."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

from bolinas.pipelines.evals.conservation import REQUIRED_VARIANT_COLUMNS
from bolinas.pipelines.evals.inference import compute_variant_scores
from bolinas.pipelines.evals.metrics import compute_pairwise_metrics


def get_dataset_config(name):
    for d in config["datasets"]:
        if d["name"] == name:
            return d
    raise ValueError(f"dataset {name!r} not found in config")


def get_model_config(name):
    for m in config["models"]:
        if m["name"] == name:
            return m
    raise ValueError(f"model {name!r} not found in config")


# Each model entry must declare exactly one source — fail loud here so a
# typo in config doesn't surface as a confusing rule error later.
for _m in config["models"]:
    _has_gcs = "gcs_path" in _m
    _has_hf = "hf_repo" in _m
    assert _has_gcs ^ _has_hf, (
        f"model {_m['name']!r} must have exactly one of `gcs_path` or `hf_repo`"
    )


# Wildcard alternations used across rules.
DATASETS = [d["name"] for d in config["datasets"]]
MODELS = [m["name"] for m in config["models"]]
