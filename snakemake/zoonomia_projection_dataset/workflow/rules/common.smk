"""Shared imports and constants for ``zoonomia_projection_dataset``.

URLs come from ``CONSERVATION_TRACKS`` — single source of truth in
``src/bolinas/evals/conservation.py``. Only ``phyloP_447m`` is downloaded
at pipeline runtime; ``phyloP_241m`` is fetched only by the one-off
calibration script.
"""

import gzip

import numpy as np
import pandas as pd
import polars as pl

from bolinas.evals.conservation import CONSERVATION_TRACKS


SPECIES = config["species"]
WINDOW_SIZE = int(config["window_size"])
STEP_SIZE = int(config["step_size"])
PHYLOP_447M_THRESHOLD = float(config["phyloP_447m_threshold"])
STANDARD_CHROMS = list(config["standard_chroms"][SPECIES])

# Sanity-check: track must be in the registry.
assert "phyloP_447m" in CONSERVATION_TRACKS, (
    "phyloP_447m must be present in bolinas.evals.conservation.CONSERVATION_TRACKS"
)
