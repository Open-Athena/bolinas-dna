# Bolinas-DNA Experiments

This directory contains experiment utilities and examples for training genomic language models (gLMs) using the Marin/Levanter framework.

## Structure

- `__init__.py` - Package initialization
- `simple_train_config.py` - Training configuration dataclass (from Marin)
- `defaults.py` - Simplified experiment utilities for tokenization and training
- `exp0_plantcad_example.py` - Example PlantCAD training experiment

## Quick Start

### Running the Example Experiment

```bash
# From the bolinas-dna root directory
uv run python experiments/exp0_plantcad_example.py
```

### Creating a New Experiment

1. Import the necessary modules:

```python
from experiments.defaults import default_tokenize, default_train
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import GpuConfig
from levanter.data.text import TextLmDatasetFormat
from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main
```

2. Define your dataset and tokenizer:

```python
dataset_path = "your-org/your-dataset"
tokenizer_path = "your-org/your-tokenizer"

tokenized = default_tokenize(
    name="my-experiment",
    dataset=dataset_path,
    tokenizer=tokenizer_path,
    format=TextLmDatasetFormat(text_key="seq"),  # For DNA sequences
)
```

3. Configure your model:

```python
model_config = LlamaConfig(
    seq_len=512,
    hidden_dim=128,
    intermediate_dim=512,
    num_heads=4,
    num_kv_heads=4,
    num_layers=4,
)
```

4. Configure training:

```python
train_config = SimpleTrainConfig(
    resources=GpuConfig(gpu_count=1),
    train_batch_size=16,
    num_train_steps=1000,
    learning_rate=3e-4,
    weight_decay=0.1,
    steps_per_export=100,
    max_eval_batches=4,
)
```

5. Create the training step and run:

```python
model = default_train(
    name="my-model",
    tokenized=tokenized,
    model_config=model_config,
    train_config=train_config,
    eval_harness_tasks=[],  # No eval harness for DNA
    use_default_validation=False,  # No default validation for DNA
)

if __name__ == "__main__":
    executor_main(steps=[model])
```

## DNA-Specific Considerations

### Text Key
DNA sequences in HuggingFace datasets typically use the `seq` field instead of `text`. Make sure to specify this:

```python
format=TextLmDatasetFormat(text_key="seq")
```

### Validation
For DNA experiments, we typically disable default validation sets (which are designed for natural language):

```python
use_default_validation=False
eval_harness_tasks=[]
```

### Tokenizers
DNA tokenizers typically have very small vocabularies (e.g., PlantCaduceus uses 7 tokens for [A, C, G, T, N, PAD, UNK]).

## Scaling Up

The example experiment uses a small model for testing. For production runs, you can scale up:

### Larger Models

```python
model_config = LlamaConfig(
    seq_len=512,
    hidden_dim=1408,      # Scale up from 128
    intermediate_dim=5632, # Scale up from 512
    num_heads=22,          # Scale up from 4
    num_kv_heads=22,       # Scale up from 4
    num_layers=24,         # Scale up from 4
)
```

### More Training Steps

```python
train_config = SimpleTrainConfig(
    resources=GpuConfig(gpu_count=8),  # More GPUs
    train_batch_size=128,              # Larger batches
    num_train_steps=100000,            # More steps
    learning_rate=3e-4,
    weight_decay=0.1,
    steps_per_export=1000,             # Less frequent exports
)
```

## Differences from Full Marin

This `defaults.py` is a simplified version that removes dependencies on other Marin experiments modules. The main differences:

1. **No default validation sets** - Paloma and other NLP validation sets are not included
2. **No eval harness tasks** - CORE_TASKS and MMLU_TASKS are not available
3. **Simplified parameter counting** - Uses a basic estimation instead of exact counting
4. **No annealing/SFT** - `default_anneal()` and `default_sft()` are not included

If you need these features, you can either:
- Copy additional files from the Marin repo
- Import directly from the Marin monorepo (requires full repo access)
- Implement DNA-specific versions of these features

## Further Reading

- [Marin Documentation](https://github.com/marin-community/marin)
- [Levanter Documentation](https://levanter.readthedocs.io/)
- [PlantCAD Paper](https://github.com/PlantCaduceus)
