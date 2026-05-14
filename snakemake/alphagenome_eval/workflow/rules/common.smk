"""Common imports + helpers for alphagenome_eval rules."""

import os

import pandas as pd
from datasets import load_dataset

from bolinas.pipelines.evals.alphagenome import score_variants_alphagenome
from bolinas.pipelines.evals.conservation import REQUIRED_VARIANT_COLUMNS
from bolinas.pipelines.evals.metrics import compute_pairwise_metrics


DATASETS = config["datasets"]
