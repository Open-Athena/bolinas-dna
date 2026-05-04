"""Diagnose: why do some splice_*_variant variants have exon_dist > 30?

Per Ensembl VEP docs (https://useast.ensembl.org/info/genome/variation/prediction/predicted_data.html),
all of:
  splice_donor_5th_base_variant     — 5th intronic base of donor splice site
  splice_region_variant             — 1-3bp exonic OR 3-8bp intronic
  splice_donor_region_variant       — 3-6 intronic OR 1-3 exonic from donor
  splice_polypyrimidine_tract_variant — intronic 3-17 from acceptor

… should be within ~17bp of an exon-intron boundary. Yet our data shows ~19K
negatives + 8 positives with exon_dist > 30.

Most likely cause: our `exon` parquet (in `trait_intervals.get_exon`) is filtered
to protein_coding transcripts only. If VEP labels a variant as splice_* via a
NON-coding transcript, the variant could be far from the nearest protein-coding
exon. Let's verify.
"""
import polars as pl
import polars_bio as pb

from bolinas.data.utils import load_annotation

print("Loading dataset_all...", flush=True)
V = pl.read_parquet(
    "s3://oa-bolinas/snakemake/evals/results/mendelian_traits/dataset_all.parquet",
    columns=["chrom", "pos", "ref", "alt", "label", "consequence_final",
             "consequence_group", "tss_dist", "exon_dist",
             "tss_closest_gene_id", "exon_closest_gene_id"],
)
print(f"  {V.height} rows", flush=True)

# Splicing variants with exon_dist > 30 (the surprising group)
suspect = V.filter(
    (pl.col("consequence_group") == "splicing") & (pl.col("exon_dist") > 30)
)
print(f"\n{suspect.height} 'splicing' variants with exon_dist > 30", flush=True)

print("\nDistribution of exon_dist among these:")
print(
    suspect.group_by(["label", "consequence_final"])
    .agg(
        n=pl.len(),
        median_exon_dist=pl.col("exon_dist").median(),
        q90_exon_dist=pl.col("exon_dist").quantile(0.90),
        q99_exon_dist=pl.col("exon_dist").quantile(0.99),
        max_exon_dist=pl.col("exon_dist").max(),
    )
    .sort(["label", "n"], descending=[True, True])
)

# Now: load the FULL annotation (all biotypes) and recompute exon_dist
# vs the protein-coding-only one.
print("\nLoading full GTF annotation...", flush=True)
annotation_url = "http://ftp.ensembl.org/pub/release-107/gtf/homo_sapiens/Homo_sapiens.GRCh38.107.chr.gtf.gz"
ann = load_annotation(annotation_url)
print(f"  {ann.height} annotation rows", flush=True)

# All-biotype exons (parsing biotype + gene_id from attribute)
all_exons = (
    ann.filter(pl.col("feature") == "exon")
    .with_columns(
        pl.col("attribute").str.extract(r'gene_id "([^;]*)";').alias("gene_id"),
        pl.col("attribute").str.extract(r'transcript_biotype "([^;]*)";').alias("biotype"),
    )
    .filter(pl.col("chrom").is_in([f"{i}" for i in range(1, 23)] + ["X", "Y"]))
    .select(["chrom", "start", "end", "gene_id", "biotype"])
    .unique(subset=["chrom", "start", "end"])
    .sort(["chrom", "start"])
)
print(f"  {all_exons.height} unique exons (all biotypes)", flush=True)
print(f"  biotypes:\n{all_exons['biotype'].value_counts().sort('count', descending=True).head(10)}")

# Sample check: take 10 of the suspect variants and look up their
# nearest exon in BOTH protein-coding-only and all-biotype lists.
sample = suspect.head(10).select("chrom", "pos", "consequence_final", "exon_dist", "exon_closest_gene_id")
print(f"\nSample suspect variants and their PC-exon distance:")
print(sample)

# For each sample variant, find the nearest exon in all-biotype list
print("\nFinding nearest exon in all-biotype annotation for each sample...")
sample_intervals = sample.with_columns(
    end=pl.col("pos"),
    start=pl.col("pos") - 1,
).select("chrom", "start", "end", "consequence_final")

near = pb.nearest(
    sample_intervals,
    all_exons,
    cols1=("chrom", "start", "end"),
    cols2=("chrom", "start", "end"),
    suffixes=("", "_exon"),
    output_type="polars.DataFrame",
)
print(f"  near columns: {near.columns}")
print(near.select("chrom", "start", "consequence_final",
                  pl.col("distance").alias("exon_dist_all_biotypes"),
                  "biotype_exon",
                  "gene_id_exon"))

# Aggregate: for ALL suspect variants, recompute exon_dist using all biotypes
print(f"\nRecomputing exon_dist for ALL {suspect.height} suspect variants using all-biotype exons...")
suspect_intervals = suspect.with_columns(
    end=pl.col("pos"),
    start=pl.col("pos") - 1,
).select("chrom", "start", "end", "label", "consequence_final", "exon_dist")

all_near = pb.nearest(
    suspect_intervals,
    all_exons,
    cols1=("chrom", "start", "end"),
    cols2=("chrom", "start", "end"),
    suffixes=("", "_exon"),
    output_type="polars.DataFrame",
)

print(f"\nDistance using all-biotype exons (vs protein-coding-only):")
summary = (
    all_near.with_columns(
        all_dist=pl.col("distance"),
        gain=pl.col("exon_dist") - pl.col("distance"),  # how much closer
    )
    .group_by(["label", "consequence_final"])
    .agg(
        n=pl.len(),
        median_pc=pl.col("exon_dist").median(),
        median_all=pl.col("all_dist").median(),
        n_within_30_all=(pl.col("all_dist") <= 30).sum(),
        n_with_biotype=(pl.col("biotype_exon").is_not_null()).sum(),
    )
    .sort(["label", "n"], descending=[True, True])
)
print(summary)

# Top biotypes for the variants that ARE within 30 in all-biotype list but were >30 in PC list
print("\nBiotype of nearest exon for variants now within 30 (in all-biotype list):")
print(
    all_near.filter(pl.col("distance") <= 30)
    .group_by(["biotype_exon"])
    .agg(n=pl.len())
    .sort("n", descending=True)
    .head(15)
)
