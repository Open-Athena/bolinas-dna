"""Common imports + helpers for the zeroshot_vep pipeline."""

from itertools import product
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from bolinas.evals.metrics import pairwise_accuracy
from bolinas.zeroshot_vep.features import extract_features, read_cache
from bolinas.zeroshot_vep.scores import SCORE_NAMES, score_cache


# Required columns on every matched-pair eval dataset (mendelian / complex /
# eqtl share the same schema).
REQUIRED_VARIANT_COLUMNS = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "subset",
    "match_group",
)


def get_dataset_config(name: str) -> dict:
    for d in config["datasets"]:
        if d["name"] == name:
            return d
    raise ValueError(f"dataset {name!r} not found in config")


def get_model_config(name: str) -> dict:
    for m in config["models"]:
        if m["name"] == name:
            return m
    raise ValueError(f"model {name!r} not found in config")


MODELS = [m["name"] for m in config["models"]]
DATASETS = [d["name"] for d in config["datasets"]]

# Every (model, window, dataset) combo this pipeline will produce. Listing per
# model because window lists are model-specific (255 vs 256 native).
MODEL_WINDOW_DATASET_TRIPLES: list[tuple[str, int, str]] = [
    (m["name"], int(w), d["name"])
    for m in config["models"]
    for w in m["windows"]
    for d in config["datasets"]
]


def all_cache_paths() -> list[str]:
    return [
        f"results/cache/{model}__win{w}__{dataset}"
        for model, w, dataset in MODEL_WINDOW_DATASET_TRIPLES
    ]


def all_score_paths() -> list[str]:
    return [
        f"results/scores/{model}__win{w}__{dataset}.parquet"
        for model, w, dataset in MODEL_WINDOW_DATASET_TRIPLES
    ]


def all_metric_paths() -> list[str]:
    return [
        f"results/metrics/{model}__win{w}__{dataset}.parquet"
        for model, w, dataset in MODEL_WINDOW_DATASET_TRIPLES
    ]
