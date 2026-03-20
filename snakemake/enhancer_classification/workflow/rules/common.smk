import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.metrics import precision_recall_curve
import pandas as pd
import polars as pl
import pyBigWig
from huggingface_hub import hf_hub_download

from bolinas.data.intervals import GenomicSet
from bolinas.data.utils import ENHANCER_CRE_CLASSES, add_rc, load_fasta


SPECIES = list(config["genome_urls"].keys())

DATASET_NAMES = list(config["datasets"].keys())

ALL_INTERVALS: set[str] = set()
ALL_SPLIT_NAMES: set[str] = set()
for _dataset_config in config["datasets"].values():
    ALL_INTERVALS.add(_dataset_config["intervals"])
    _split_config = config["splits"][_dataset_config["split"]]
    for _species_splits in _split_config.values():
        ALL_SPLIT_NAMES.update(_species_splits.keys())

ALL_CONSERVATIONS: set[str] = set()
for _species_cons in config.get("conservation", {}).values():
    ALL_CONSERVATIONS.update(_species_cons.keys())

TRAIN_SPLIT = "train"

MODEL_DEFAULTS = config["models"]["default"]


def get_model_config(model_name: str) -> dict:
    return {**MODEL_DEFAULTS, **config["models"].get(model_name, {})}


def get_all_dataset_outputs() -> list[str]:
    outputs = []
    for dataset_name, dataset_config in config["datasets"].items():
        split_config = config["splits"][dataset_config["split"]]
        split_names: set[str] = set()
        for species_splits in split_config.values():
            split_names.update(species_splits.keys())
        for split_name in split_names:
            outputs.append(f"results/dataset/{dataset_name}/{split_name}.parquet")
    return outputs


def fasta_to_df(path: str, label: int, genome: str) -> pd.DataFrame:
    series = load_fasta(path)
    df = series.to_frame().reset_index(names="id")
    coords = df["id"].str.split(":", expand=True)
    df["chrom"] = coords[0]
    start_end = coords[1].str.split("-", expand=True)
    df["start"] = start_end[0].astype(int)
    df["end"] = start_end[1].astype(int)
    df["strand"] = "+"
    df["label"] = label
    df["genome"] = genome
    return df[["genome", "chrom", "start", "end", "strand", "seq", "label"]]
