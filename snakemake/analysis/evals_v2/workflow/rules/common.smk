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
    assert (
        _has_gcs ^ _has_hf
    ), f"model {_m['name']!r} must have exactly one of `gcs_path` or `hf_repo`"


# Wildcard alternations used across rules.
DATASETS = [d["name"] for d in config["datasets"]]
MODELS = [m["name"] for m in config["models"]]


def get_model_datasets(model_name):
    """Datasets a given model is evaluated on.

    Defaults to all configured datasets; a model entry may set
    ``datasets: [name, …]`` to restrict evaluation to a subset.
    """
    cfg = get_model_config(model_name)
    if "datasets" not in cfg:
        return DATASETS
    bad = [d for d in cfg["datasets"] if d not in DATASETS]
    assert not bad, (
        f"model {model_name!r} `datasets` references unknown names: {bad} "
        f"(known: {DATASETS})"
    )
    return cfg["datasets"]


def get_model_batch_size(model_name):
    """Per-model ``batch_size`` if set, else the global ``inference.batch_size``."""
    bs = get_model_config(model_name).get(
        "batch_size", config["inference"]["batch_size"]
    )
    assert (
        isinstance(bs, int) and bs > 0
    ), f"model {model_name!r} `batch_size` must be a positive int, got {bs!r}"
    return bs
