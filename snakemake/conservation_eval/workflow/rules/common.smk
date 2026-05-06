from pathlib import Path

import pandas as pd

from datasets import load_dataset

from bolinas.evals.conservation import (
    CONSERVATION_TRACKS,
    REQUIRED_VARIANT_COLUMNS,
    aggregate_conservation_metrics,
    score_variants_at_positions,
)


SCORES = config["scores"]
SPLITS = config["splits"]
DATASETS = config["datasets"]
INPUT_HF_PREFIX = config["input_hf_prefix"]

# Sanity-check up-front: every score listed in config must be a known track.
_unknown = set(SCORES) - set(CONSERVATION_TRACKS)
assert not _unknown, (
    f"unknown scores in config: {_unknown}. " f"Known: {sorted(CONSERVATION_TRACKS)}"
)
