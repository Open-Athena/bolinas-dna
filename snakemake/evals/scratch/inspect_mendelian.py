"""Inspect results/dataset_unsplit/mendelian_traits.parquet (S3-cached).

Reports row counts, label balance, gene-match invariants, columns, subset
distribution, and source distribution.
"""

import polars as pl


PATH = (
    "s3://oa-bolinas/snakemake/evals/results/dataset_unsplit/mendelian_traits.parquet"
)


def main() -> None:
    V = pl.read_parquet(PATH)
    print(f"rows: {len(V)}")
    print(f"columns: {V.columns}")
    print()

    label_counts = V["label"].value_counts(sort=True)
    print("label balance:")
    print(label_counts)
    print()

    # Strict 1:1 invariant
    n_pos = V.filter(pl.col("label")).height
    n_neg = V.filter(~pl.col("label")).height
    print(
        f"n_pos: {n_pos}, n_neg: {n_neg}, ratio: {n_neg / n_pos if n_pos else float('nan'):.3f}"
    )
    assert n_pos == n_neg, "expected strict 1:1 matching"
    print()

    # Gene-match invariants: each match_group has one positive and one negative
    # sharing chrom, consequence_final, tss_closest_gene_id, exon_closest_gene_id.
    gm = V.group_by("match_group").agg(
        n=pl.len(),
        n_chroms=pl.col("chrom").n_unique(),
        n_consequence=pl.col("consequence_final").n_unique(),
        n_tss_gene=pl.col("tss_closest_gene_id").n_unique(),
        n_exon_gene=pl.col("exon_closest_gene_id").n_unique(),
        n_labels=pl.col("label").n_unique(),
    )
    bad_size = gm.filter(pl.col("n") != 2).height
    bad_chrom = gm.filter(pl.col("n_chroms") != 1).height
    bad_cons = gm.filter(pl.col("n_consequence") != 1).height
    bad_tss = gm.filter(pl.col("n_tss_gene") != 1).height
    bad_exon = gm.filter(pl.col("n_exon_gene") != 1).height
    bad_label = gm.filter(pl.col("n_labels") != 2).height
    print(
        f"match_group invariants: groups={gm.height}, "
        f"bad_size={bad_size}, bad_chrom={bad_chrom}, bad_cons={bad_cons}, "
        f"bad_tss={bad_tss}, bad_exon={bad_exon}, bad_label={bad_label}"
    )
    assert bad_size == 0
    assert bad_chrom == 0
    assert bad_cons == 0
    assert bad_tss == 0
    assert bad_exon == 0
    assert bad_label == 0
    print("all gene-match invariants hold")
    print()

    # subset column
    if "subset" in V.columns:
        print("subset distribution:")
        print(V["subset"].value_counts(sort=True))
    else:
        print("WARNING: no `subset` column!")
    print()

    # source distribution (positives only)
    if "source" in V.columns:
        print("source distribution (positives):")
        print(V.filter(pl.col("label"))["source"].value_counts(sort=True))
    print()

    # Chromosome distribution
    print("chrom distribution:")
    print(V["chrom"].value_counts(sort=True).head(30))


if __name__ == "__main__":
    main()
