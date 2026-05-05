"""Shared imports and constants for ``zoonomia_projection_dataset``.

URLs come from ``CONSERVATION_TRACKS`` — single source of truth in
``src/bolinas/evals/conservation.py``. Only ``phyloP_447m`` is downloaded
at pipeline runtime; ``phyloP_241m`` is fetched only by the one-off
calibration script.

Pipeline is hg38-only by design — the phyloP bigWigs are human-anchored
and the downstream cross-mammal projection is human-anchored too. No
``{species}`` wildcard.
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from bolinas.evals.conservation import CONSERVATION_TRACKS


WINDOW_SIZE = int(config["window_size"])
STEP_SIZE = int(config["step_size"])
PHYLOP_447M_THRESHOLD = float(config["phyloP_447m_threshold"])
STANDARD_CHROMS = list(config["standard_chroms"])

# Sanity-check: track must be in the registry.
assert "phyloP_447m" in CONSERVATION_TRACKS, (
    "phyloP_447m must be present in bolinas.evals.conservation.CONSERVATION_TRACKS"
)


# ===== Cross-mammal projection knobs (rule all_projected) =====

HAL_PATH = config["hal_path"]
SPECIES_TSV = config["species_tsv"]
PROJECT_MIN_P = str(config["project_min_p"])
TARGET_LEN = int(config["target_len"])
PRE_RESIZE_MIN = int(config["pre_resize_min_len"])
PRE_RESIZE_MAX = int(config["pre_resize_max_len"])
TIER = config.get("tier", "full")
assert TIER in {"full", "smoke"}, f"tier must be full or smoke, got {TIER!r}"
assert TARGET_LEN == WINDOW_SIZE, (
    f"projection target_len ({TARGET_LEN}) must equal source WINDOW_SIZE "
    f"({WINDOW_SIZE}); the 255+BOS=256 invariant decouples from the input "
    f"window length only intentionally."
)
assert 0 < PRE_RESIZE_MIN <= TARGET_LEN <= PRE_RESIZE_MAX

SPECIES = pl.read_csv(SPECIES_TSV, separator="\t")["species"].to_list()
assert {"Homo_sapiens", "Mus_musculus", "Bos_taurus"}.issubset(SPECIES), (
    "species TSV must include Homo_sapiens, Mus_musculus, Bos_taurus "
    "(force-included by the dedup policy)"
)
if TIER == "smoke":
    SPECIES = ["Homo_sapiens", "Mus_musculus", "Bos_taurus", "Loxodonta_africana"]
