import bioframe as bf
import pandas as pd
import polars as pl
from pathlib import Path

from biofoundation.data import Genome
from datasets import Dataset
from huggingface_hub import HfApi

from bolinas.evals.materialize import materialize_sequences


CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
SPLIT_CHROMS = {
    "train": CHROMS[::2],  # odd chroms
    "test": CHROMS[1::2],  # even chroms
}
SPLITS = list(SPLIT_CHROMS.keys())
COORDS = ["chrom", "pos", "ref", "alt"]
