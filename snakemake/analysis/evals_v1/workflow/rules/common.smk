"""Common imports, constants, and helper functions for all rules."""

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score

from bolinas.evals.inference import compute_variant_scores
from bolinas.evals.metrics import aggregate_metrics, compute_metrics
from bolinas.evals.plotting import plot_metrics_vs_step, plot_models_comparison

COORDINATES = ["chrom", "pos", "ref", "alt"]


def get_dataset_config(dataset_name):
    for dataset in config["datasets"]:
        if dataset["name"] == dataset_name:
            return dataset
    raise ValueError(f"Dataset {dataset_name} not found in config")


def get_model_config(model_name):
    for model in config["models"]:
        if model["name"] == model_name:
            return model
    raise ValueError(f"Model {model_name} not found in config")


def get_original_dataset_path(dataset_name):
    return config["baselines"]["dataset_mapping"][dataset_name]
