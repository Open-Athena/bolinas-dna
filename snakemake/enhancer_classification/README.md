# Enhancer Classification

Creates binary classification datasets (enhancer vs non-enhancer) and trains
binary classifiers using AlphaGenome's CNN encoder as a feature extractor.

## Dataset

### Output schema

| Column | Type | Description |
|--------|------|-------------|
| `genome` | str | Species (`"homo_sapiens"` or `"mus_musculus"`) |
| `chrom` | str | Chromosome (Ensembl naming: `"1"`, `"X"`, etc.) |
| `start` | int | 0-based start coordinate |
| `end` | int | End coordinate (exclusive) |
| `strand` | str | `"+"` (forward) or `"-"` (reverse complement) |
| `seq` | str | DNA sequence (255bp) |
| `label` | int | 1 = enhancer, 0 = non-enhancer |

### Method

**Positives**: ENCODE SCREEN cCREs (Registry V4) filtered to enhancer-like
classes (dELS, pELS), resized to 255bp windows centered on each element's
midpoint. Soft-masked genomes from Ensembl.

**Negatives**: Random 255bp windows sampled per-chromosome via
`bedtools shuffle -chrom`, excluding positives and undefined (N-rich) regions.
1:1 positive-to-negative ratio.

**Conservation filtering**: Dataset v3 filters enhancers by conservation score.
Conservation is computed over the center 150bp of each CRE (the minimum ENCODE
cCRE length), ensuring the score reflects the element's core rather than flanking
sequence. Enhancer counts by `conserved_bases` threshold:

| Source | n >= 0 | n >= 10 | n >= 20 |
|--------|--------|---------|---------|
| phyloP_241m | 1,718,669 | 404,522 (23.5%) | 283,393 (16.5%) |
| phastCons_43p | 1,718,669 | 403,966 (23.5%) | 310,674 (18.1%) |

**Splits**: Configured via composable named split configs. Default (split_v1):
human chr19 held out for validation; remaining human and all mouse for training.

**Augmentation**: Training splits include reverse complement sequences
(strand="-"). Non-training splits are subsampled to `max_samples`.

## Model

### Architecture

Binary classifier using AlphaGenome's convolutional encoder trunk (no
transformer/attention layers):

```
Input: (B, 255, 4) one-hot DNA
  → SequenceEncoder (DnaEmbedder + 6x DownResBlock+Pool)
  → (B, 1536, 2) encoder trunk output
  → AdaptiveAvgPool1d(1) → Flatten → Linear(1536, 1)
  → binary logit
```


### Model configs

The `default` model config in `config.yaml` defines shared hyperparameters.
Other configs inherit from `default` and override specific fields. See
`config.yaml` for current values.

### Experiments

The `experiments` section in `config.yaml` defines which (model, dataset) pairs
to train. Each experiment produces outputs under
`results/model/{model}/{dataset}/`:
- `best.ckpt` — Lightning checkpoint
- `metrics.json` — validation AUROC and AUPRC

Training logs to [W&B project `bolinas-enhancer-classification`](https://wandb.ai/gonzalobenegas/bolinas-enhancer-classification).

### Results and findings

Dataset v3, 1 epoch, cosine LR (10% warmup), lr=1e-4:

| Model | val_auroc | val_auprc |
|-------|-----------|-----------|
| default (frozen pretrained) | 0.939 | 0.926 |
| finetune (unfrozen pretrained) | **0.954** | **0.945** |
| random_init (unfrozen random) | 0.930 | 0.916 |
| finetune_mlp (unfrozen, MLP-256) | 0.955 | 0.946 |

- Frozen encoder converges within a single epoch; continued training overfits.
- Finetuning the pretrained backbone gives the best results.
- Pretrained encoder outperforms random init even when both are unfrozen.
- MLP head (LayerNorm → Linear → GELU → Dropout → Linear) adds negligible
  improvement over linear head (+0.001); not worth the extra complexity.
- At 90% precision, finetune/v4 achieves ~86% recall (threshold ≈ 0.69).
  Caveat: computed on the balanced 1:1 validation set — genome-wide precision
  would be lower at the same threshold since non-enhancers vastly outnumber
  enhancers.

### Code layout

| File | Description |
|------|-------------|
| `src/bolinas/enhancer_classification/dataset.py` | `EnhancerDataset` — PyTorch Dataset loading parquet splits |
| `src/bolinas/enhancer_classification/model.py` | `EnhancerClassifier` — Lightning module with AlphaGenome encoder + linear head |
| `src/bolinas/enhancer_classification/train.py` | CLI training script with argparse |
| `workflow/rules/model.smk` | Snakemake rule calling the train CLI |

## Prerequisites

- AWS credentials configured (EC2 IAM role or `aws configure`)
- **Single GPU required** for training (multi-GPU not supported)
- All available CPU cores are used for data loading (`threads: workflow.cores`)
- AlphaGenome pretrained weights are downloaded automatically from HuggingFace

## Usage

```bash
uv sync --group enhancer-classification
uv run snakemake
```

### Top misclassified regions (finetune / v4)

**Top 10 false positives** (negatives predicted as enhancers):

| Coordinates | Probability | Notes |
|-------------|-------------|-------|
| 19:617254-617509 | 0.995 | Overlaps half of a dELS that also overlaps CDS, pretty weird |
| 19:28577254-28577509 | 0.994 | Should investigate, overlaps dELS with borderline conservation |
| 19:57162184-57162439 | 0.993 | Non-conserved dELS |
| 19:2185844-2186099 | 0.993 | Some CDS + some conserved non-coding on flank of dELS; might be functional |
| 19:11426852-11427107 | 0.989 | |
| 19:6931821-6932076 | 0.989 | |
| 19:55146866-55147121 | 0.989 | |
| 19:6309612-6309867 | 0.988 | |
| 19:17974001-17974256 | 0.987 | |
| 19:55227392-55227647 | 0.986 | |

**Top 10 false negatives** (enhancers predicted as non-enhancers):

| Coordinates | Probability | Notes |
|-------------|-------------|-------|
| 19:56313324-56313579 | 0.001 | pELS, moderate conservation, repeat |
| 19:17496740-17496995 | 0.005 | dELS, moderate conservation, repeat |
| 19:40365411-40365666 | 0.006 | dELS, moderate conservation, repeat |
| 19:49678736-49678991 | 0.008 | pELS, moderate conservation, repeat |
| 19:53552424-53552679 | 0.008 | |
| 19:42339179-42339434 | 0.009 | |
| 19:41260989-41261244 | 0.010 | |
| 19:45742723-45742978 | 0.011 | |
| 19:39936861-39937116 | 0.013 | |
| 19:13100078-13100333 | 0.014 | |

## Configuration

See `config/config.yaml`.
