#!/usr/bin/env bash
# Section A — trace win_7_000231918 / Ceratotherium_simum end-to-end.
#
# Run on the SkyPilot cluster via `sky exec zoonomia-v1-audit ...`.
# Outputs go to ~/audit/section_a/ on the cluster; we'll fetch them after.

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate audit

S3=s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results

WD=~/audit
mkdir -p $WD/section_a
cd $WD

# ---- Pull artifacts (parallel) ----
echo "[A] Pulling artifacts from S3..."
(aws s3 cp $S3/human/intervals/filtered/min0.20.bed.gz min0.20.bed.gz --quiet) &
(aws s3 cp $S3/human/genome.2bit human.2bit --quiet) &
(aws s3 cp $S3/projection/genomes/Ceratotherium_simum.2bit Ceratotherium_simum.2bit --quiet) &
(aws s3 cp $S3/projection/min0.20/per_species/Ceratotherium_simum.parquet Ceratotherium_simum.parquet --quiet) &
(aws s3 cp $S3/human/intervals/undefined.bed human_undefined.bed --quiet) &
wait
echo "[A] Artifacts pulled."

cd $WD/section_a

# ---- 1. Find the human anchor in min0.20.bed.gz ----
echo ""
echo "==[A.1] Find win_7_000231918 in human anchor BED=="
ANCHOR=$(zcat ../min0.20.bed.gz | awk '$4 == "win_7_000231918"')
echo "$ANCHOR" | tee anchor.bed
HCHR=$(echo "$ANCHOR" | awk '{print $1}')
HSTART=$(echo "$ANCHOR" | awk '{print $2}')
HEND=$(echo "$ANCHOR" | awk '{print $3}')
echo "  → chrom=$HCHR start=$HSTART end=$HEND length=$((HEND - HSTART))"

# ---- 2. Confirm human anchor has 0 N's at hg38 coords ----
echo ""
echo "==[A.2] Human anchor sequence in hg38=="
twoBitToFa -seq=${HCHR} -start=${HSTART} -end=${HEND} ../human.2bit human_anchor.fa  # human 2bit uses Ensembl bare names ("7"), not UCSC ("chr7")
HUMAN_SEQ=$(grep -v '^>' human_anchor.fa | tr -d '\n')
HUMAN_LEN=${#HUMAN_SEQ}
HUMAN_N=$(echo -n "$HUMAN_SEQ" | tr -d -c 'Nn' | wc -c)
echo "  → length=$HUMAN_LEN  N_count=$HUMAN_N"
echo "  → seq: $HUMAN_SEQ"

# ---- 3. Re-extract the 255 bp rhino window (must match HF row) ----
echo ""
echo "==[A.3] Rhino window at JH767724.1:24562726-24562981 (the HF row's coords)=="
twoBitToFa -seq=JH767724.1 -start=24562726 -end=24562981 ../Ceratotherium_simum.2bit rhino_255.fa
RHINO_SEQ=$(grep -v '^>' rhino_255.fa | tr -d '\n')
RHINO_LEN=${#RHINO_SEQ}
RHINO_N=$(echo -n "$RHINO_SEQ" | tr -d -c 'Nn' | wc -c)
echo "  → length=$RHINO_LEN  N_count=$RHINO_N"
echo "  → seq: $RHINO_SEQ"

# ---- 4. Wider gap context: 2.5 kb on each side ----
echo ""
echo "==[A.4] Wider rhino context JH767724.1:24560000-24565500=="
twoBitToFa -seq=JH767724.1 -start=24560000 -end=24565500 ../Ceratotherium_simum.2bit rhino_wide.fa
python3 << 'PY'
seq = ""
for line in open("rhino_wide.fa"):
    if not line.startswith(">"):
        seq += line.strip()
# context starts at 24560000
N_runs = []
i = 0
while i < len(seq):
    if seq[i] in "Nn":
        j = i
        while j < len(seq) and seq[j] in "Nn":
            j += 1
        N_runs.append((24560000 + i, 24560000 + j, j - i))
        i = j
    else:
        i += 1
print(f"  Total N runs in 24,560,000-24,565,500: {len(N_runs)}")
for s, e, n in N_runs:
    print(f"  N-run: {s}-{e}  (len={n})")
print(f"\n  Window of interest: 24,562,726-24,562,981 (255 bp)")
print(f"  → relative-to-context: bases {24562726 - 24560000}-{24562981 - 24560000}")
PY

# ---- 5. Full rhino N-region BED (intersect with window) ----
echo ""
echo "==[A.5] All rhino N-regions overlapping the 255bp window=="
twoBitInfo -nBed ../Ceratotherium_simum.2bit ../rhino_nbed.bed
awk '$1=="JH767724.1" && $2 < 24562981 && $3 > 24562726' ../rhino_nbed.bed | tee rhino_overlap.bed

echo ""
echo "==[A.5b] All rhino N-regions on JH767724.1 within 100 kb of window=="
awk '$1=="JH767724.1" && $2 < 24700000 && $3 > 24450000' ../rhino_nbed.bed | head -30

# ---- 6. Per-species projection parquet row ----
echo ""
echo "==[A.6] rhino per-species projection row for win_7_000231918=="
python3 << 'PY'
import polars as pl
df = pl.read_parquet("../Ceratotherium_simum.parquet")
row = df.filter(pl.col("query_name") == "win_7_000231918")
print(row)
print(f"\n  total rhino rows: {df.height}")
PY

# ---- 7. Reverse lookup from all_species_with_sequence parquet (range-filtered) ----
echo ""
echo "==[A.7] HF row sanity check from S3 parquet (download cost only the column subset for this anchor)=="
python3 << 'PY'
import polars as pl
# scan_parquet supports lazy filtering; for one row, just download the file
# and filter — but it's 11 GB. Skip for the example trace; the per-species
# parquet (A.6) is the canonical source and the bedtools extraction in A.3
# is what would land in the HF row by construction.
print("  (skipped — see A.6 for the canonical projection row; the sequence column would be the A.3 extraction)")
PY

echo ""
echo "[A] DONE."
