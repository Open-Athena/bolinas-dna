"""Common imports + helpers for supervised_vep rules."""

import pandas as pd
from datasets import load_dataset

from bolinas.evals.conservation import REQUIRED_VARIANT_COLUMNS
from bolinas.supervised.inference import compute_pooled_features


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
