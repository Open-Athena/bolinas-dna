"""Common imports + helpers for supervised_vep rules."""

import pandas as pd
from datasets import load_dataset

from bolinas.evals.conservation import REQUIRED_VARIANT_COLUMNS
from bolinas.supervised.inference import compute_pooled_features
from bolinas.supervised.pipeline import (
    compute_metrics_from_oof,
    compute_zeroshot_baseline_metrics,
    fit_oof_predictions,
    is_recipe_dataset_compatible,
    write_fold_records_json,
)


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


DATASETS = [d["name"] for d in config["datasets"]]
MODELS = [m["name"] for m in config["models"]]
RECIPES = list(config.get("recipes", []))
CLASSIFIERS = list(config.get("classifiers", []))


def valid_dataset_recipe_pairs():
    """Yield ``(dataset, recipe)`` for combos the pipeline should run."""
    for d in DATASETS:
        for r in RECIPES:
            if is_recipe_dataset_compatible(r, d):
                yield d, r


def metric_targets():
    """All supervised-metrics targets the leaderboard will aggregate over."""
    out = []
    for m in MODELS:
        for d, r in valid_dataset_recipe_pairs():
            for c in CLASSIFIERS:
                out.append(
                    f"results/metrics/{m}/{d}/{r}/{c}.parquet"
                )
    return out


def baseline_metric_targets():
    """Zero-shot baseline metrics, one per (model, dataset)."""
    return [
        f"results/baseline_metrics/{m}/{d}.parquet"
        for m in MODELS
        for d in DATASETS
    ]
