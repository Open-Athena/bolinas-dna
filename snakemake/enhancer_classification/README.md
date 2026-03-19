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

Uses `bf16-mixed` precision and `torch.compile`.

### Model configs

The `default` model config in `config.yaml` defines shared hyperparameters.
Other configs inherit from `default` and override specific fields. See
`config.yaml` for current values.

### Experiments

The `experiments` section in `config.yaml` defines which (model, dataset) pairs
to train. Each experiment produces outputs under
`results/model/{model}/{dataset}/`:
- `best.ckpt` — best checkpoint (by val AUROC)
- `metrics.json` — best validation AUROC and epoch number

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
- Gradient norm is noisy (4–44) with gradient_clip_val=1.0, meaning clipping
  fires on most steps. Consider raising or removing the clip value.

### Code layout

| File | Description |
|------|-------------|
| `src/bolinas/enhancer_classification/dataset.py` | `EnhancerDataset` — PyTorch Dataset loading parquet splits |
| `src/bolinas/enhancer_classification/model.py` | `EnhancerClassifier` — Lightning module with AlphaGenome encoder + linear head |
| `src/bolinas/enhancer_classification/train.py` | CLI training script with argparse |
| `workflow/rules/model.smk` | Snakemake rule calling the train CLI |

## Prerequisites

- AWS credentials configured (EC2 IAM role or `aws configure`)
- **Single GPU required** for training — hardcoded to `devices=1`, multi-GPU
  not supported. Uses bf16-mixed precision and `torch.compile`.
- All available CPU cores are used for data loading (`threads: workflow.cores`)
- AlphaGenome pretrained weights are downloaded automatically from HuggingFace

## Usage

```bash
# Install dependencies (from repo root)
uv sync --group enhancer-classification

# Dry run
uv run snakemake -n

# Build datasets only
uv run snakemake

# Train a specific experiment
uv run snakemake results/model/debug/v3/metrics.json

# Train directly without Snakemake (e.g. for debugging)
uv run python -m bolinas.enhancer_classification.train \
    --train-parquet results/dataset/v3/train.parquet \
    --val-parquet results/dataset/v3/validation.parquet \
    --weights-path weights/model_all_folds.safetensors \
    --output-dir results/model/debug/v3 \
    --learning-rate 1e-3 \
    --batch-size 16 \
    --max-epochs 2 \
    --freeze-backbone \
    --wandb-run debug-v3

# Run tests (no GPU needed)
uv run pytest tests/enhancer_classification/ -v
```

## Configuration

See `config/config.yaml` for all parameters:

- `datasets`: versioned dataset configs, each referencing a split config and
  interval type (e.g., ELS, ELS_conserved_20)
- `splits`: named chromosome split configs
- `max_samples`: per-split subsampling caps
- `models`: named model training configs (freeze_backbone, lr, etc.)
- `experiments`: list of `{model, dataset}` pairs to train
- `alphagenome_weights_path`: path to pretrained AlphaGenome weights
  (download from [HuggingFace](https://huggingface.co/gtca/alphagenome_pytorch))
- `window_size`, `seed`: core parameters
