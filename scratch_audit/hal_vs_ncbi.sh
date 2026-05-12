#!/usr/bin/env bash
# Compare genome-wide lowercase fraction between:
#   - HAL-derived per-species FASTA (from s3://oa-bolinas/.../projection/genomes/{species}.2bit)
#   - NCBI Datasets `genomic.fna` for the same accession
#
# Output: ~/audit/hal_vs_ncbi/comparison.tsv

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate audit

# Install NCBI Datasets if not already there.
if ! command -v datasets >/dev/null 2>&1; then
    conda install -y -n audit -c conda-forge -c bioconda ncbi-datasets-cli >/dev/null 2>&1
fi

S3=s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results/projection/genomes
WD=~/audit/hal_vs_ncbi
mkdir -p $WD && cd $WD

# Counter as a real file (avoids heredoc-stdin issues from the original draft).
cat > count_seq.py << 'PY'
import sys
total = n_count = lower_count = 0
for line in sys.stdin:
    if not line or line.startswith(">"):
        continue
    s = line.rstrip("\n")
    total += len(s)
    n_count += s.count("N") + s.count("n")
    for c in s:
        if c in "acgt":
            lower_count += 1
non_n = total - n_count
print(f"{total}\t{n_count}\t{lower_count}\t{lower_count/non_n if non_n else 0:.6f}\t{n_count/total if total else 0:.6f}")
PY

# (species, accession) pairs spanning the mask spectrum we observed.
cat > pairs.tsv << 'EOF'
Homo_sapiens	GCA_000001405.27
Ceratotherium_simum	GCF_000283155.1
Petromus_typicus	GCA_004026965.1
Mus_musculus	GCF_000001635.26
EOF

echo -e "species\taccession\tprefix\tsource\ttotal_bases\tn_bases\tlower_bases\tlower_frac_of_acgt\tn_frac" > comparison.tsv

while IFS=$'\t' read -r SPECIES ACC; do
    PREFIX=${ACC:0:3}
    echo ""
    echo "==== $SPECIES ($ACC) ===="

    # --- HAL-derived ---
    echo "[$SPECIES] downloading HAL 2bit ..."
    aws s3 cp $S3/${SPECIES}.2bit ${SPECIES}.hal.2bit --quiet
    echo "[$SPECIES] counting HAL bases ..."
    HAL_STATS=$(twoBitToFa ${SPECIES}.hal.2bit /dev/stdout | python3 count_seq.py)
    echo "  hal: $HAL_STATS"
    echo -e "$SPECIES\t$ACC\t$PREFIX\thal\t$HAL_STATS" >> comparison.tsv
    rm -f ${SPECIES}.hal.2bit

    # --- NCBI Datasets ---
    echo "[$SPECIES] downloading NCBI Datasets ${ACC} ..."
    rm -rf ncbi_${SPECIES}
    mkdir -p ncbi_${SPECIES}
    (cd ncbi_${SPECIES} && datasets download genome accession ${ACC} --include genome --no-progressbar 2>&1 | tail -5 && unzip -q ncbi_dataset.zip)
    NCBI_FNA=$(find ncbi_${SPECIES}/ncbi_dataset/data/${ACC} -name '*genomic.fna*' | head -1)
    if [ -z "$NCBI_FNA" ]; then
        echo "  WARNING: no FASTA found for ${ACC}"
        rm -rf ncbi_${SPECIES}
        continue
    fi
    echo "[$SPECIES] counting NCBI bases (file: $NCBI_FNA) ..."
    if [[ "$NCBI_FNA" == *.gz ]]; then
        NCBI_STATS=$(zcat "$NCBI_FNA" | python3 count_seq.py)
    else
        NCBI_STATS=$(cat "$NCBI_FNA" | python3 count_seq.py)
    fi
    echo "  ncbi: $NCBI_STATS"
    echo -e "$SPECIES\t$ACC\t$PREFIX\tncbi_datasets\t$NCBI_STATS" >> comparison.tsv
    rm -rf ncbi_${SPECIES}

done < pairs.tsv

echo ""
echo "===== COMPARISON ====="
column -t -s $'\t' comparison.tsv
