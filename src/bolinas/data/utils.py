import numpy as np
import pandas as pd
import polars as pl
from Bio import SeqIO
from Bio.Seq import Seq


def get_array_split_pairs(L: int, n_shards: int) -> list[tuple[int, int]]:
    """
    Calculates the (start, end) slice pairs exactly like np.array_split.

    Args:
        L (int): Total length of the array/DataFrame.
        n_shards (int): Number of shards to create.

    Returns:
        list[tuple]: A list of (start, end) tuples for slicing.
    """
    base_size = L // n_shards
    remainder = L % n_shards

    # Create a list of the *size* of each shard
    shard_sizes = [base_size + 1] * remainder + [base_size] * (n_shards - remainder)

    # Calculate the cumulative sum to get the slice indices
    # e.g., [0, 5, 10, 15, 19, 23]
    indices = np.cumsum([0] + shard_sizes)

    # Create and return the (start, end) pairs
    # e.g., [(0, 5), (5, 10), (10, 15), (15, 19), (19, 23)]
    return list(zip(indices[:-1], indices[1:]))


def load_annotation(path: str) -> pl.DataFrame:
    """
    Load a GTF/GFF annotation file and convert to BED-like format.

    Args:
        path (str): Path to the GTF/GFF annotation file.

    Returns:
        pl.DataFrame: DataFrame with columns [chrom, source, feature, start, end,
                      score, strand, frame, attribute]. Start coordinates are
                      converted from 1-based (GTF) to 0-based (BED). Only features
                      with strand '+' or '-' are retained.
    """
    return (
        pl.read_csv(
            path,
            has_header=False,
            separator="\t",
            comment_prefix="#",
            new_columns=[
                "chrom",
                "source",
                "feature",
                "start",
                "end",
                "score",
                "strand",
                "frame",
                "attribute",
            ],
        )
        .with_columns(pl.col("start") - 1)  # gtf to bed conversion
        .filter(pl.col("strand").is_in(["+", "-"]))
    )


def get_mrna_exons(ann: pl.DataFrame) -> pl.DataFrame:
    """
    Extract mRNA exons from an annotation DataFrame.

    Args:
        ann (pl.DataFrame): Annotation DataFrame from load_annotation() with
                           columns including 'feature', 'attribute', 'chrom',
                           'start', 'end', and 'strand'.

    Returns:
        pl.DataFrame: DataFrame with columns [chrom, start, end, strand, transcript_id]
                     containing only mRNA exons. Supports both transcript_biotype
                     (e.g., human) and gbkey (e.g., some species) annotations.
    """
    return (
        ann.filter(pl.col("feature") == "exon")
        .with_columns(
            pl.col("attribute")
            .str.extract(r'transcript_id "(.*?)"')
            .alias("transcript_id"),
            pl.col("attribute")
            .str.extract(r'transcript_biotype "(.*?)"')
            .alias("transcript_biotype"),
            pl.col("attribute").str.extract(r'gbkey "(.*?)"').alias("gbkey"),
        )
        # most annotations use transcript_biotype (e.g. GCF_000001405.40 Homo sapiens),
        # but some use gbkey (e.g. GCF_000691845.1 Merops nubicus)
        .filter((pl.col("transcript_biotype") == "mRNA") | (pl.col("gbkey") == "mRNA"))
        .select(["chrom", "start", "end", "strand", "transcript_id"])
    )


def get_promoters(
    exons: pl.DataFrame,
    n_upstream: int,
    n_downstream: int,
) -> pl.DataFrame:
    """
    Extract promoter regions from exon DataFrame.

    Args:
        exons (pl.DataFrame): Exon DataFrame with columns [chrom, start, end,
                             strand, transcript_id] from get_mrna_exons().
        n_upstream (int): Number of bases upstream of TSS to include.
        n_downstream (int): Number of bases downstream of TSS to include.

    Returns:
        pl.DataFrame: DataFrame with columns [chrom, start, end] containing
                     unique promoter regions. For '+' strand, promoter is
                     [TSS - n_upstream, TSS + n_downstream]. For '-' strand,
                     promoter is [TSS - n_downstream, TSS + n_upstream].
    """
    return (
        exons.group_by("transcript_id")
        .agg(
            pl.col("chrom").first(),
            pl.col("start").min(),
            pl.col("end").max(),
            pl.col("strand").first(),
        )
        .with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("start") - n_upstream)
            .otherwise(pl.col("end") - n_downstream)
            .alias("start")
        )
        .with_columns((pl.col("start") + n_upstream + n_downstream).alias("end"))
        .select(["chrom", "start", "end"])
        .unique(["chrom", "start", "end"])
        .sort(["chrom", "start", "end"])
    )


def read_bed_to_pandas(path: str) -> pd.DataFrame:
    """
    Read a BED file into a pandas DataFrame.

    Args:
        path (str): Path to the BED file.

    Returns:
        pd.DataFrame: DataFrame with columns [chrom, start, end].
    """
    return pd.read_csv(path, sep="\t", header=None, names=["chrom", "start", "end"])


def write_pandas_to_bed(intervals: pd.DataFrame, path: str) -> None:
    """
    Write a pandas DataFrame to a BED file.

    Args:
        intervals (pd.DataFrame): DataFrame with columns [chrom, start, end].
        path (str): Path to write the BED file.
    """
    intervals.to_csv(path, sep="\t", header=False, index=False)


def load_fasta(path: str) -> pd.Series:
    """
    Load a FASTA file into a pandas Series.

    Args:
        path (str): Path to the FASTA file.

    Returns:
        pd.Series: Series with sequence IDs as index and sequences as values.
                  Series name is 'seq'.
    """
    with open(path) as handle:
        return pd.Series(
            {rec.id: str(rec.seq) for rec in SeqIO.parse(handle, "fasta")},
            name="seq",
        )


def add_rc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add reverse complement sequences to a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns ['id', 'seq'] where 'seq'
                          contains DNA sequences.

    Returns:
        pd.DataFrame: DataFrame with twice the rows. Original sequences have
                     '_+' appended to their IDs, reverse complements have '_-'
                     appended to their IDs.
    """
    df_pos = df.copy()
    df_neg = df.copy()
    df_pos["id"] = df_pos["id"] + "_+"
    df_neg["id"] = df_neg["id"] + "_-"
    df_neg["seq"] = df_neg["seq"].apply(lambda x: str(Seq(x).reverse_complement()))
    return pd.concat([df_pos, df_neg], ignore_index=True)
