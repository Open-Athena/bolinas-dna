# Enhancer Classification

Binary enhancer classifier. For design decisions, results, and iteration
history see [#96](https://github.com/Open-Athena/bolinas-dna/issues/96).

This pipeline also includes a **per-bin enhancer segmentation** formulation
(issue [#115](https://github.com/Open-Athena/bolinas-dna/issues/115)) that
shares the conserved-enhancer definition with the classifier but predicts one
logit per 128 bp bin inside a 16384 bp window.

## Dataset output schema

### Classification (255 bp windows)

| Column | Type | Description |
|--------|------|-------------|
| `genome` | str | Species (`"homo_sapiens"` or `"mus_musculus"`) |
| `chrom` | str | Chromosome (Ensembl naming: `"1"`, `"X"`, etc.) |
| `start` | int | 0-based start coordinate |
| `end` | int | End coordinate (exclusive) |
| `strand` | str | `"+"` (forward) or `"-"` (reverse complement) |
| `seq` | str | DNA sequence (255 bp) |
| `label` | int | 1 = enhancer, 0 = non-enhancer |

### Segmentation (16384 bp windows, 128 × 128 bp bins)

| Column | Type | Description |
|--------|------|-------------|
| `genome` | str | Species |
| `chrom` | str | Chromosome |
| `start` | int | 0-based window start |
| `end` | int | Window end (= start + 16384) |
| `strand` | str | `"+"` or `"-"` (RC augmentation reverses `labels`) |
| `seq` | str | DNA sequence (16384 bp) |
| `labels` | list[int8] | Per-bin label (length 128); `1` if ≥50 % of the bin overlaps a conserved-enhancer positive interval, `0` otherwise. Gray-zone-aware datasets (see below) also use `-1` for bins that overlap an *intermediate*-conservation CRE — masked out of loss + AUPRC. |

## Code layout

| File | Description |
|------|-------------|
| `src/bolinas/enhancer_classification/{dataset,model,train}.py` | 255 bp binary classifier |
| `src/bolinas/enhancer_segmentation/{dataset,model,train}.py` | Per-bin segmenter (Conv1d head on encoder) |
| `src/bolinas/enhancer_segmentation/labeling.py` | `label_windows_by_bin_overlap` — bin-level labels from enhancer intervals |
| `src/bolinas/enhancer_segmentation/misclassified.py` | `top_misclassified_bins` — per-species FP/FN ranking with exon/SCREEN-cCRE/phastCons annotations |
| `workflow/rules/model.smk` | Classifier training rules |
| `workflow/rules/segmentation.smk` | Segmentation data build + training rules |

## Prerequisites

- AWS credentials configured (EC2 IAM role or `aws configure`)
- **Single GPU required** for training (multi-GPU not supported)
- All available CPU cores are used for data loading (`threads: workflow.cores`)

## Usage

```bash
uv sync --group enhancer-classification
uv run snakemake
```

## Configuration

See `config/config.yaml`.

### Positive-set variants via `cre_class_groups`

The `{cre_class}` wildcard in the interval path (e.g. `noexon/conserved/phastCons_43p/20/ELS`) is a **group alias** defined in `config["cre_class_groups"]`. Each alias maps to a list of raw ENCODE SCREEN v4 classes:

- `ELS` → `[dELS, pELS]` — enhancer-like signatures, the original positive set.
- `ALL` → all 8 SCREEN v4 classes (`PLS, pELS, dELS, CA, CA-CTCF, CA-H3K4me3, CA-TF, TF`).

Add your own alias (e.g. `PLS_only: [PLS]`) to `cre_class_groups` and reference it in a `seg_datasets` entry's `intervals` path to run the pipeline against a new positive-set definition — no Python changes needed. The four knobs compose freely:

| Axis | How to change |
|------|---------------|
| CRE class | Swap `{cre_class}` suffix (e.g. `.../20/ALL` instead of `.../20/ELS`). |
| Conservation stringency | Bump `{n}` (minimum conserved bp inside the 150 bp center, out of 150). |
| Conservation track | Use a different `{conservation}` segment (e.g. `phyloP_241m` instead of `phastCons_43p`); add a new entry under `config.conservation.{species}` if needed. |
| Exon subtraction | Include or drop the leading `noexon/` segment. |

### Positive-set exploration (sub-issue of #96)

`seg_datasets` entries `seg_pos_all_64k`, `seg_pos_cons30_64k`, `seg_pos_cons50_64k`, `seg_pos_withexon_64k` — five-run serial sweep varying the positive-set definition one axis at a time, sharing the `xfmr2_w64k_s42` model (2 transformer layers, 64k window, step-capped at 811). Launched via `skypilot/seg_positive_set.sky.yaml`.

### Gray-zone exclusion (3-tier labels)

`seg_pos_grayzone_64k` extends the binary labeler with a third "ignore"
tier. The dataset's per-species `intervals` field encodes both a positive
CRE path and an ignore CRE path separated by the literal `__ignore__`:

```yaml
seg_pos_grayzone_64k:
    intervals:
        homo_sapiens: noexon/conserved/phastCons_43p/50/ELS__ignore__noexon/grayzone/phastCons_43p/10_50/ELS
```

The `cre_filter_gray_zone` rule produces the ignore-set parquet (CREs with
`n_low ≤ conserved < n_high`). The `label_segmentation_windows` rule
reads both, calls `label_windows_by_bin_overlap` with `ignore_intervals`,
and emits `int8` labels in `{-1, 0, 1}`. Downstream the model's BCE
loss and AUPRC mask out `-1` bins; the `val_grayzone_prob` metric
tracks the model's mean predicted probability on those bins as a
calibration check.
