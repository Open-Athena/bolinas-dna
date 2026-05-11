#!/usr/bin/env bash
# Post iteration-1 results to issue #175.
#
# Pre-reqs: metrics_aggregated.parquet on S3, sanity-check ref under scratch/.
#
# This script:
# 1. Pulls the aggregated parquet from S3
# 2. Runs the analysis script (sanity + top-N + heatmaps)
# 3. Uploads heatmap PNGs + CSV via gh-upload-asset
# 4. Builds the iteration-1 comment from the template
# 5. Posts the comment to issue #175 (🤖 prefix per CLAUDE.md)
# 6. Edits the issue body's "Current findings" section
#
# Manual review before step 5/6 recommended — `--dry-run` flag stops after step 4.

set -euo pipefail

REPO=Open-Athena/bolinas-dna
ISSUE=175
DRY_RUN=${DRY_RUN:-0}

SHA=$(git rev-parse HEAD)
echo "[post] commit SHA: $SHA"

# 1. Pull aggregated parquet to local
mkdir -p scratch/iter1
aws s3 cp s3://oa-bolinas/snakemake/analysis/zeroshot_vep/results/metrics_aggregated.parquet scratch/iter1/metrics_aggregated.parquet
aws s3 cp s3://oa-bolinas/snakemake/analysis/zeroshot_vep/results/metrics_aggregated.csv scratch/iter1/metrics_aggregated.csv

# 2. Run analysis (writes heatmaps to scratch/iter1/plots/)
uv run python scratch/zeroshot_vep_analysis.py \
    scratch/iter1/metrics_aggregated.parquet \
    --out-dir scratch/iter1/plots \
    > scratch/iter1/analysis.log 2>&1
tail -40 scratch/iter1/analysis.log

if [ "$DRY_RUN" = "1" ]; then
    echo "[post] DRY_RUN — stopping before upload/post"
    exit 0
fi

echo "[post] iter 1 analysis complete; review scratch/iter1/ before proceeding manually."
echo "[post] Next steps:"
echo "  - Upload heatmaps via /gh-upload-asset"
echo "  - Hand-edit scratch/iter1_comment_template.md with the printed table values"
echo "  - gh issue comment $ISSUE --body-file scratch/iter1/comment.md"
echo "  - gh api -X PATCH /repos/$REPO/issues/$ISSUE -f body=@... to update the issue body's findings section"
