"""Rules for downloading and preparing sequence data from HuggingFace."""


rule download_dataset:
    """Download train and validation splits from HuggingFace and save as FASTA."""
    output:
        train_fasta="results/data/{dataset}/train.fasta",
        val_fasta="results/data/{dataset}/validation.fasta",
        metadata="results/data/{dataset}/metadata.parquet",
    params:
        hf_path=lambda wildcards: get_hf_path(wildcards.dataset),
        seq_column=config["analysis"]["sequence_column"],
        canonicalize=config["analysis"]["consider_reverse_complement"],
    run:
        # Load train and validation splits
        print(f"Loading dataset from {params.hf_path}...")
        if params.canonicalize:
            print("  Canonicalizing sequences (treating seq and reverse complement as identical)")

        train_df = load_sequences_from_hf(
            params.hf_path, "train", params.seq_column, canonicalize=params.canonicalize
        )
        val_df = load_sequences_from_hf(
            params.hf_path, "validation", params.seq_column, canonicalize=params.canonicalize
        )

        print(f"  Train sequences: {len(train_df):,}")
        print(f"  Validation sequences: {len(val_df):,}")

        # Save as FASTA for MMseqs2
        def write_fasta(df: pl.DataFrame, path: str):
            with open(path, "w") as f:
                for row in df.iter_rows(named=True):
                    f.write(f">{row['id']}\n{row['seq']}\n")

        Path(output.train_fasta).parent.mkdir(parents=True, exist_ok=True)
        write_fasta(train_df, output.train_fasta)
        write_fasta(val_df, output.val_fasta)

        # Save metadata
        metadata = pl.concat([train_df, val_df])
        metadata.write_parquet(output.metadata)

        print(f"Saved FASTA files and metadata to results/data/{wildcards.dataset}/")


rule merge_fasta:
    """Merge train and validation FASTA files for clustering."""
    input:
        train="results/data/{dataset}/train.fasta",
        val="results/data/{dataset}/validation.fasta",
    output:
        merged="results/data/{dataset}/all_sequences.fasta",
    shell:
        "cat {input.train} {input.val} > {output.merged}"
