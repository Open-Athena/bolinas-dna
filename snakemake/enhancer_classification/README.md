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

Uses `bf16-mixed` precision and supports `torch.compile`.

### Model configs

| Config | Backbone | LR | Batch size | Epochs |
|--------|----------|-----|-----------|--------|
| `debug` | frozen | 1e-3 | 16 | 2 |
| `linear_probe` | frozen | 1e-3 | 256 | 50 |
| `finetune` | trainable | 1e-5 | 64 | 20 |

### Outputs

Each model produces under `results/model/{model}/`:
- `best.ckpt` — best checkpoint (by val AUROC)
- `metrics.json` — best validation AUROC and epoch number

### Code layout

| File | Description |
|------|-------------|
| `src/bolinas/enhancer_classification/dataset.py` | `EnhancerDataset` — PyTorch Dataset loading parquet splits |
| `src/bolinas/enhancer_classification/model.py` | `EnhancerClassifier` — Lightning module with AlphaGenome encoder + linear head |
| `src/bolinas/enhancer_classification/train.py` | CLI training script with argparse |
| `workflow/rules/model.smk` | Snakemake rule calling the train CLI |

## Prerequisites

- AWS credentials configured (EC2 IAM role or `aws configure`)
- **GPU required** for training (model uses bf16-mixed precision)
- AlphaGenome pretrained weights (set `alphagenome_weights_path` in config)

## Usage

```bash
# Install dependencies (from repo root)
uv sync --group enhancer-classification

# Dry run
uv run snakemake -n

# Build datasets only
uv run snakemake

# Train a specific model
uv run snakemake results/model/debug/metrics.json

# Train directly without Snakemake (e.g. for debugging)
uv run python -m bolinas.enhancer_classification.train \
    --train-parquet results/dataset/v1/train.parquet \
    --val-parquet results/dataset/v1/validation.parquet \
    --weights-path /path/to/alphagenome.pth \
    --output-dir results/model/debug \
    --learning-rate 1e-3 \
    --batch-size 16 \
    --max-epochs 2 \
    --freeze-backbone

# Run tests (no GPU needed)
uv run pytest tests/enhancer_classification/ -v
```

## Verification on GPU

The following steps require a GPU and were NOT run during initial implementation.
An agent picking this up on a GPU machine should:

1. **Unit tests** (already verified on CPU):
   ```bash
   uv run pytest tests/enhancer_classification/ -v
   ```

2. **Debug training run with random weights** (no pretrained weights needed):
   ```bash
   # Generate a small synthetic parquet for testing
   uv run python -c "
   import random, polars as pl
   random.seed(0)
   seqs = [''.join(random.choices('ACGT', k=255)) for _ in range(64)]
   labels = [i % 2 for i in range(64)]
   pl.DataFrame({'seq': seqs, 'label': labels}).write_parquet('/tmp/test_train.parquet')
   pl.DataFrame({'seq': seqs[:16], 'label': labels[:16]}).write_parquet('/tmp/test_val.parquet')
   "

   uv run python -m bolinas.enhancer_classification.train \
       --train-parquet /tmp/test_train.parquet \
       --val-parquet /tmp/test_val.parquet \
       --output-dir /tmp/test_model \
       --max-epochs 2 \
       --batch-size 8 \
       --num-workers 0 \
       --no-compile \
       --no-freeze-backbone

   # Verify outputs exist
   ls -la /tmp/test_model/best.ckpt /tmp/test_model/metrics.json
   cat /tmp/test_model/metrics.json
   ```

3. **Checkpoint reload smoke test**:
   ```bash
   uv run python -c "
   import torch
   from bolinas.enhancer_classification.model import EnhancerClassifier
   model = EnhancerClassifier.load_from_checkpoint('/tmp/test_model/best.ckpt')
   model.eval()
   x = torch.nn.functional.one_hot(torch.randint(0, 4, (1, 255)), 4).float()
   with torch.no_grad():
       logit = model(x)
   print(f'Logit shape: {logit.shape}, value: {logit.item():.4f}')
   "
   ```

4. **Full smoke test with pretrained weights** (if available):
   ```bash
   # Set the weights path in config, then:
   cd snakemake/enhancer_classification
   uv run snakemake results/model/debug/metrics.json
   ```

5. **Verify torch.compile works** (add `--compile` to step 2 above).

## Configuration

See `config/config.yaml` for all parameters:

- `datasets`: versioned dataset configs, each referencing a split config and
  interval type (e.g., ELS, ELS_conserved_20)
- `splits`: named chromosome split configs
- `max_samples`: per-split subsampling caps
- `models`: named model training configs (dataset, freeze_backbone, lr, etc.)
- `alphagenome_weights_path`: path to pretrained AlphaGenome weights
- `window_size`, `seed`: core parameters
