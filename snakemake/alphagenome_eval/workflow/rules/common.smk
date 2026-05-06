"""Common imports + helpers for alphagenome_eval rules."""

import os

import pandas as pd
from datasets import load_dataset

from bolinas.evals.alphagenome import score_variants_alphagenome
from bolinas.evals.conservation import REQUIRED_VARIANT_COLUMNS
from bolinas.evals.metrics import compute_pairwise_metrics


DATASETS = config["datasets"]
INPUT_HF_PREFIX = config["input_hf_prefix"]
SPLIT = config["split"]
SCORE_COLUMN = config["score_column"]
NUM_WORKERS = config["num_workers"]
SUBSET_N_PAIRS = config.get("subset_n_pairs")
