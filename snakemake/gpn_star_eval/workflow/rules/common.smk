"""Common imports + helpers for gpn_star_eval rules."""

import pandas as pd
from datasets import load_dataset

from bolinas.evals.conservation import REQUIRED_VARIANT_COLUMNS
from bolinas.evals.gpn_star import predictions_url, score_variants_gpn_star
from bolinas.evals.metrics import compute_pairwise_metrics


DATASETS = config["datasets"]
MODELS = config["models"]
SCORE_COLUMNS = config["score_columns"]
