"""Training-shard prep for the zoonomia HF datasets (issue #149).

Mirrors snakemake/training_dataset/dataset_creation pattern (RC augmentation,
shuffle, shard) but vectorised in polars (much faster than the existing
pandas + Bio.Seq path on the ~250 M-row v1 dataset).
"""

import json
from pathlib import Path

import polars as pl

# DNA complement for ACGT (upper + lower). Anything else (N or other IUPAC
# ambiguity codes — R, Y, S, W, K, M, B, D, H, V — is left as-is by
# str.replace_many, which is fine for our training data; we only need
# strict-RC behaviour on ACGT.
_COMPLEMENT = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "a": "t",
    "t": "a",
    "c": "g",
    "g": "c",
}


def reverse_complement_col(seq: pl.Expr) -> pl.Expr:
    """Vectorised reverse-complement: complement via replace_many, then reverse.

    Non-ACGT characters are preserved unchanged (replace_many leaves
    unmapped chars alone). This is intentional and approximate for IUPAC
    ambiguity codes; sequences in our pipeline are mostly ACGT after the
    upstream N-region filter.
    """
    return seq.str.replace_many(_COMPLEMENT).str.reverse()


def prepare_shards(
    parquet_path: str | Path,
    shard_paths: list[str],
    add_rc: bool,
    shuffle_seed: int,
) -> None:
    """Read source Parquet → optional RC augment → shuffle → shard to JSONL.

    Args:
        parquet_path: source Parquet (e.g. all_species_with_sequence.parquet
            or subsets/v2.parquet). Must include a ``sequence`` column.
        shard_paths: ordered list of N output JSONL paths. Row count is
            split evenly (np.array_split semantics).
        add_rc: if True, prepend an ``augmentation`` column with values
            ``"+"`` (original) and ``"-"`` (reverse-complemented sequence).
            Total row count doubles.
        shuffle_seed: passed to ``df.sample(fraction=1, shuffle=True, seed=...)``
            for cross-species interleaving.
    """
    assert len(shard_paths) > 0
    df = pl.read_parquet(str(parquet_path))
    assert "sequence" in df.columns, f"missing sequence column: {df.columns}"

    if add_rc:
        df_pos = df.with_columns(pl.lit("+").alias("augmentation"))
        df_neg = df.with_columns(
            reverse_complement_col(pl.col("sequence")).alias("sequence"),
            pl.lit("-").alias("augmentation"),
        )
        df = pl.concat([df_pos, df_neg])

    df = df.sample(fraction=1.0, shuffle=True, seed=shuffle_seed)

    n_total = len(df)
    n_shards = len(shard_paths)
    base, rem = divmod(n_total, n_shards)
    cur = 0
    for i, path in enumerate(shard_paths):
        size = base + (1 if i < rem else 0)
        shard = df.slice(cur, size)
        cur += size
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for row in shard.iter_rows(named=True):
                fh.write(json.dumps(row) + "\n")
    assert cur == n_total
