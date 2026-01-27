# Core data science libraries
import pandas as pd
import polars as pl
from tqdm import tqdm

# Bolinas module imports
from bolinas.data.intervals import GenomicSet
from bolinas.data.utils import (
    add_rc,
    get_3_prime_utr,
    get_5_prime_utr,
    get_array_split_pairs,
    get_cds,
    get_mrna_exons,
    get_ncrna_exons,
    get_promoters,
    get_promoters_from_exons,
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


def load_genome_sets(
    genomes: pd.DataFrame, genome_sets_config: list[dict]
) -> dict[str, list[str]]:
    """
    Load genome sets based on taxonomic filtering criteria.

    Args:
        genomes: DataFrame with genomes indexed by Assembly Accession
        genome_sets_config: List of dicts with 'name', 'rank_key', and 'rank_value' keys

    Returns:
        Dictionary mapping subset names to lists of genome Assembly Accessions
    """
    genome_sets = {}
    for genome_set in genome_sets_config:
        name = genome_set["name"]
        rank_key = genome_set["rank_key"]
        rank_value = genome_set["rank_value"]

        # Filter genomes based on the rank criteria
        filtered = genomes[genomes[rank_key] == rank_value]
        genome_sets[name] = filtered.index.tolist()

    return genome_sets
