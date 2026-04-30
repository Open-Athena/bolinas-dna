from pathlib import Path

import pandas as pd

from datasets import load_dataset

from bolinas.evals.conservation import (
    CONSERVATION_TRACKS,
    REQUIRED_VARIANT_COLUMNS,
    aggregate_traitgym_metrics,
    score_variants_at_positions,
)
from bolinas.evals.metrics import compute_metrics


SCORES = config["scores"]
SPLITS = config["splits"]
DATASET_HF_PATH = config["dataset_hf_path"]

# Sanity-check up-front: every score listed in config must be a known track.
_unknown = set(SCORES) - set(CONSERVATION_TRACKS)
assert not _unknown, (
    f"unknown scores in config: {_unknown}. " f"Known: {sorted(CONSERVATION_TRACKS)}"
)
