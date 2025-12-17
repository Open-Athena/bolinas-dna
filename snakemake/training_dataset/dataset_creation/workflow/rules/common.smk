# Core data science libraries
import pandas as pd
import polars as pl
from tqdm import tqdm

# Bolinas module imports
from bolinas.data.intervals import GenomicSet
from bolinas.data.utils import (
    add_rc,
    get_array_split_pairs,
    get_mrna_exons,
    get_promoters,
    load_annotation,
    load_fasta,
    read_bed_to_pandas,
    write_pandas_to_bed,
)

tqdm.pandas()


def load_genome_info(path: str) -> pd.DataFrame:
    genomes = pd.read_parquet(path).set_index("Assembly Accession")
    genomes["Assembly Name"] = genomes["Assembly Name"].str.replace(" ", "_")
    genomes["genome_path"] = (
        "tmp/"
        + genomes.index
        + "/ncbi_dataset/data/"
        + genomes.index
        + "/"
        + genomes.index
        + "_"
        + genomes["Assembly Name"]
        + "_genomic.fna"
    )
    genomes["annotation_path"] = (
        "tmp/"
        + genomes.index
        + "/ncbi_dataset/data/"
        + genomes.index
        + "/"
        + "genomic.gtf"
    )
    return genomes
