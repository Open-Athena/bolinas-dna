"""
Example training experiment.

This demonstrates the pattern for running DNA language model training using
the Marin/Levanter framework.
"""

import jax
import logging
from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main, versioned
from experiments.defaults import default_tokenize, default_train
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger("ray")

if (backend := jax.default_backend()) not in {"gpu", "cpu"}:
    raise NotImplementedError(f"Only GPU and CPU backends supported, not {backend=}")

# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
run_number = 0
num_gpus = len(jax.devices("gpu")) if backend == "gpu" else 1
tokenizer_path = "songlab/tokenizer-dna-clm"
dataset_path = "songlab/gpn-animal-promoter-dataset"
learning_rate = 3e-4
per_device_eval_parallelism = 16  # Adjust based on GPU memory
train_batch_size = per_device_eval_parallelism * num_gpus
num_train_steps = 1000
steps_per_export = 100
steps_per_cycle = 500
steps_per_eval = 100

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
model_config = LlamaConfig(
    max_seq_len=512,
    hidden_dim=128,
    intermediate_dim=512,
    num_heads=4,
    num_kv_heads=4,
    num_layers=4,
)

# -----------------------------------------------------------------------------
# Dataset configuration
# -----------------------------------------------------------------------------
data_tokenized = default_tokenize(
    name="gpn-animal-promoter",
    dataset=versioned(dataset_path),
    tokenizer=tokenizer_path,
    # DNA sequences are in `seq`, not `text`
    format=TextLmDatasetFormat(text_key="seq"),
)

# -----------------------------------------------------------------------------
# Training configuration
# -----------------------------------------------------------------------------
train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_gpu("T4", count=num_gpus),
    train_batch_size=train_batch_size,
    per_device_eval_parallelism=per_device_eval_parallelism,
    learning_rate=learning_rate,
    lr_schedule="inv",
    warmup=0.05,
    decay=0.1,
    cycle_length=steps_per_cycle,
    steps_per_eval=steps_per_eval,
    num_train_steps=num_train_steps,
    steps_per_export=steps_per_export,
    data_seed=42,
)

training_step = default_train(
    name=f"gpn-animal-promoter-r{run_number:02d}",
    tokenized=data_tokenized,
    model_config=model_config,
    train_config=train_config,
    tags=["dna", "gpn", "training"],
    eval_harness_tasks=[],
    use_default_validation=False,
)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("🧬 DNA Training Experiment")
    logger.info("=" * 64)
    logger.info(f"Model:              {model_config}")
    logger.info(f"Learning rate:      {learning_rate}")
    logger.info(f"Global batch size:  {train_batch_size}")
    logger.info(f"Micro batch size:   {per_device_eval_parallelism}")
    logger.info(f"Training steps:     {num_train_steps:,}")
    logger.info(f"Steps per export:   {steps_per_export:,}")
    logger.info(f"Steps per eval:     {steps_per_eval:,}")
    logger.info("=" * 64)

    executor_main(steps=[training_step])
