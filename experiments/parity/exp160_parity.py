# Copyright The Marin Authors / Bolinas Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity-check rerun of exp160 against its existing WandB baseline.

Validates the bolinas-dna marin-consumer infrastructure end-to-end (issue #168)
by re-running ~1000 steps of exp160's ``zoonomia_v1`` arm and comparing against
the reference run on dna-dev:

- WandB group: ``exp160-zoonomia-v1-v2-v0.1``
- Reference run name: ``dna-bolinas-zoonomia-v1-v2-v0.1-zoonomia_v1``

This is a one-time validation script, not a maintained experiment. The
scientific content (model, hparams, data, eval setup) MUST stay identical to
exp160; only imports are rewired to use bolinas-dna's modules. Specifically,
``NUM_TRAIN_STEPS`` stays at 10_000 so the LR schedule (warmup=0.1 → step 1000,
decay=0.2) produces the same hparams as the reference; the run is killed
manually shortly after the step-1000 eval lands.

Source of truth (do not change scientific content here without changing it
upstream too): ``marin-community/marin@dna-dev:experiments/dna/exp160_zoonomia_v1_v2.py``.

Caveats to flag during the comparison:
- RNG seed: verify the reference exp160 pinned a deterministic seed; if not,
  step-by-step train-loss will drift between runs but eval numbers should
  still be close.
- Tokenization cache: aim to hit ``bolinas-zoonomia-v1-zoonomia_v1-char-bos``
  so training inputs match the reference run exactly.
- Marin version drift: the locked marin wheel is newer than dna-dev's
  vendored levanter; any behavior change in the training loop / eval harness
  will surface here.
"""

import dataclasses
import logging
import os
from datetime import timedelta
from functools import lru_cache

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

# bolinas-dna ports — see issue #168 / src/bolinas/levanter and
# src/bolinas/evals/lm_eval. Importing DNALmDatasetFormat triggers the
# @LmDatasetFormatBase.register_subclass("dna") decorator at module load,
# so "dna" is a valid format choice anywhere downstream.
from bolinas.evals.lm_eval.task_configs import TRAITGYM_MENDELIAN_V2_255
from bolinas.levanter.defaults import dna_effective_seq_len
from bolinas.levanter.formats import DNALmDatasetFormat

# Stays in marin: ``convert_to_levanter_task_config`` lived in
# ``experiments.evals.task_configs`` on dna-dev but moved to
# ``marin.evaluation.evaluation_config`` on current main — refresh this import
# if marin reorganizes again.
#
# NOTE on `experiments.*` imports: marin's `experiments/` package only ships
# in the `marin-root` git source, which carries a ``[tool.uv.sources]
# marin-* = { workspace = true }`` block that overrides find-links across
# the consumer's whole resolve. That cascade leaves `iris._build_info.BUILD_DATE`
# empty and trips the controller's freshness check. So we vendor the two
# `experiments.*` symbols we use (`default_tokenize`, `qwen3_0_6b_hd128`)
# at the bottom of this file instead of importing them. Re-evaluate once
# marin publishes a `marin-root-latest` wheel.
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.utils import fsspec_utils
from marin.evaluation.evaluation_config import convert_to_levanter_task_config
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    VersionedValue,
    ensure_versioned,
    executor_main,
    this_output_path,
)
from marin.execution.remote import remote
from marin.processing.tokenize import (
    HfDatasetSpec,
    TokenizeConfig,
    lm_mixture_data_config,
    tokenize,
)
from marin.processing.tokenize.tokenize import HfTokenizeConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

# =============================================================================
# Constants — preserved verbatim from exp160 (do not edit scientific content).
# =============================================================================

VERSION = "v0.1"
TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_BASE_SEQ_LEN = 255  # bp (256 - 1 for BOS)

# Two arms of the zoonomia projection pipeline. The parity check below sweeps
# only ``zoonomia_v1`` by default (set SWEEP_DATASETS to override).
TRAIN_DATASETS = {
    "zoonomia_v1": "bolinas-dna/zoonomia-v1-v1",  # whole-genome cohort
    "zoonomia_v2": "bolinas-dna/zoonomia-v1-v2",  # TSS-proximal subset of v1
}

# Four region-specific validation sets, each tokenized functional + nonfunctional
# for the LL-gap signal (genomes-v5 region IDs).
VAL_DATASETS: tuple[tuple[str, str], ...] = (
    ("v30", "bolinas-dna/genomes-v5-validation-intervals-v30_255_255"),  # enhancers
    ("v5", "bolinas-dna/genomes-v5-validation-intervals-v5_255_255"),  # CDS
    (
        "v1",
        "bolinas-dna/genomes-v5-validation-intervals-v1_255_255",
    ),  # upstream (promoter)
    (
        "v15",
        "bolinas-dna/genomes-v5-validation-intervals-v15_255_255",
    ),  # downstream (3'-UTR)
)

# Training masks lowercase positions to 1% loss weight. Zoonomia datasets use
# ``sequence`` as the text field (vs the older ``seq`` default for genomes-v5).
TRAIN_FORMAT = DNALmDatasetFormat(text_key="sequence", lowercase_weight=0.01)

# Validation tokenization specs — the two terms of the LL gap.
VAL_SPECS: tuple[tuple[str, DNALmDatasetFormat], ...] = (
    ("functional", DNALmDatasetFormat(uppercase_weight=1.0, lowercase_weight=0.0)),
    ("nonfunctional", DNALmDatasetFormat(uppercase_weight=0.0, lowercase_weight=1.0)),
)

BATCH_SIZE = 4096
TPU_TYPES: tuple[str, ...] = ("v5p-8",)

# Optimizer (AdamConfig defaults; schedule shape from exp_bolinas_4b_sweep.py).
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
EPSILON = 1e-8
MAX_GRAD_NORM = 1.0
WARMUP_FRACTION = 0.1
DECAY_FRACTION = 0.2
LR_SCHEDULE = "linear"
MIN_LR_RATIO = 0.0

# Training horizon. **MUST stay 10_000** so the LR schedule (warmup=0.1 →
# step 1000, decay=0.2) is identical to the exp160 reference run; reducing
# this would change the schedule shape and invalidate the comparison. Stop
# the parity run manually once the step-1000 eval logs land.
NUM_TRAIN_STEPS = 10_000

# Eval cadence and checkpoint policy.
EVALS_PER_RUN = 10
CHECKPOINTS_PER_RUN = 3
CHECKPOINT_TIME_INTERVAL = timedelta(hours=1)

WANDB_PROJECT = "marin"

_EXPECTED_VOCAB_SIZE_WARNING = (
    f"Tokenizer {TOKENIZER!r} not found in _KNOWN_VOCAB_SIZES"
)
logging.getLogger("marin.processing.tokenize.data_configs").addFilter(
    lambda record: _EXPECTED_VOCAB_SIZE_WARNING not in record.getMessage()
)


# =============================================================================
# Environment overrides
# =============================================================================


def _selected_datasets() -> dict[str, str]:
    """Return the subset of TRAIN_DATASETS named in SWEEP_DATASETS (or all if unset)."""
    raw = os.getenv("SWEEP_DATASETS")
    if not raw:
        return dict(TRAIN_DATASETS)
    requested = tuple(s.strip() for s in raw.split(","))
    invalid = [n for n in requested if n not in TRAIN_DATASETS]
    if invalid:
        raise ValueError(
            f"Invalid SWEEP_DATASETS {invalid}; available: {sorted(TRAIN_DATASETS)}"
        )
    return {n: TRAIN_DATASETS[n] for n in requested}


# =============================================================================
# Builders
# =============================================================================


@lru_cache(maxsize=1)
def _model_seq_len() -> int:
    """Model context size = base DNA seq len + special tokens (BOS)."""
    return dna_effective_seq_len(DNA_BASE_SEQ_LEN, TOKENIZER)


# Same orchestrator footprint as exp160 — heavy work runs on zephyr workers.
_TOKENIZE_RESOURCES = ResourceConfig.with_cpu(cpu=1, ram="12g", disk="10g")


def _tokenize(
    name: str, dataset: str, dataset_format: DNALmDatasetFormat
) -> ExecutorStep:
    return default_tokenize(
        name=name,
        dataset=dataset,
        tokenizer=TOKENIZER,
        format=dataset_format,
        resources=_TOKENIZE_RESOURCES,
    )


def _build_data_mixture(strategy: str, dataset: str) -> LmDataConfig:
    """One training component + four validation sets each tokenized per VAL_SPEC."""
    components: dict[str, ExecutorStep] = {
        strategy: _tokenize(
            f"bolinas-zoonomia-v1-{strategy}-char-bos", dataset, TRAIN_FORMAT
        ),
    }
    for region, val_dataset in VAL_DATASETS:
        for suffix, fmt in VAL_SPECS:
            key = f"val_{region}_{suffix}"
            components[key] = _tokenize(f"bolinas-v5-{key}-char-bos", val_dataset, fmt)
    return lm_mixture_data_config(
        components=components,
        weights={strategy: 1.0},
    )


def _build_model_config() -> Qwen3Config:
    return dataclasses.replace(qwen3_0_6b_hd128, max_seq_len=_model_seq_len())


def _build_optimizer() -> AdamConfig:
    return AdamConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        max_grad_norm=MAX_GRAD_NORM,
        warmup=WARMUP_FRACTION,
        decay=DECAY_FRACTION,
        lr_schedule=LR_SCHEDULE,
        min_lr_ratio=MIN_LR_RATIO,
    )


def _eval_harness_config() -> LmEvalHarnessConfig:
    # Importing ``bolinas.evals.lm_eval.task_configs`` (above) loads the
    # package and runs its ``_install_task_manager_patch()`` side effect,
    # which monkeypatches ``lm_eval.tasks.TaskManager`` so it always includes
    # the bolinas-dna custom-task directory on its search path. Marin's
    # current ``LmEvalHarnessConfig`` no longer accepts ``include_path``;
    # without that patch, ``traitgym_mendelian_v2_255`` would be unreachable.
    return LmEvalHarnessConfig(
        task_spec=convert_to_levanter_task_config([TRAITGYM_MENDELIAN_V2_255]),
    )


def _checkpointer(num_train_steps: int) -> CheckpointerConfig:
    return CheckpointerConfig(
        save_interval=CHECKPOINT_TIME_INTERVAL,
        keep=[dict(every=max(1, num_train_steps // CHECKPOINTS_PER_RUN))],
    )


def _build_train_step(strategy: str, dataset: str) -> ExecutorStep:
    steps_per_eval = max(1, NUM_TRAIN_STEPS // EVALS_PER_RUN)
    run_name = f"dna-bolinas-zoonomia-v1-v2-{VERSION}-{strategy}"
    tags = ("dna", "exp160-parity", "zoonomia_v1_v2", VERSION, f"strategy={strategy}")

    inner = TrainLmConfig(
        data=_build_data_mixture(strategy, dataset),
        model=_build_model_config(),
        train_seq_len=_model_seq_len(),
        optimizer=_build_optimizer(),
        eval_harness=_eval_harness_config(),
        eval_harness_steps=steps_per_eval,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=list(tags),
                group=f"exp160-zoonomia-v1-v2-{VERSION}",
                name=run_name,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=NUM_TRAIN_STEPS,
            steps_per_eval=steps_per_eval,
            checkpointer=_checkpointer(NUM_TRAIN_STEPS),
            mesh=MeshConfig(axes={"replica": 1, "data": -1, "model": 1}),
            allow_nondivisible_batch_size=True,
        ),
    )
    pod_config = TrainLmOnPodConfig(
        train_config=inner,
        resources=ResourceConfig.with_tpu(TPU_TYPES),
        output_path=this_output_path(),
    )
    return ExecutorStep(
        name=os.path.join("checkpoints", run_name),
        fn=remote(run_levanter_train_lm, resources=ResourceConfig.with_cpu()),
        config=pod_config,
    )


def main() -> None:
    selected = _selected_datasets()
    steps = [
        _build_train_step(strategy, dataset) for strategy, dataset in selected.items()
    ]
    executor_main(
        steps=steps,
        description=f"exp160 parity check (bolinas-dna #168) — zoonomia v1/v2 {VERSION}",
    )


# =============================================================================
# Vendored from marin's `experiments/` package (see import comment above).
# =============================================================================

# `qwen3_0_6b_hd128` from marin/experiments/qwen3.py — head_dim=128 variant of
# Qwen3-0.6B (matches HF "Qwen/Qwen3-0.6B"). Vendored verbatim.
qwen3_0_6b_hd128 = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=1024,
    intermediate_dim=3072,
    num_heads=16,
    num_kv_heads=8,
    num_layers=28,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=True,
    head_dim=128,
)


# Subset of marin/experiments/defaults.py:default_tokenize that we actually
# exercise (HF dataset path: a single ``"<owner>/<name>"`` string). The full
# upstream version also handles GCS paths, ``HfDatasetSpec``, and
# ``ExecutorStep`` inputs; not needed for the parity check's HF-only calls.
_HF_BUCKET_URI_PREFIX = "hf://"
_HF_BUCKET_PATH_PREFIX = "gs://"  # marin's HF_BUCKET_PATH_PREFIX is gs://, used for cached HF datasets


def _is_hf_bucket_path(path: str) -> bool:
    return path.startswith(_HF_BUCKET_URI_PREFIX) or path.startswith(_HF_BUCKET_PATH_PREFIX)


def default_tokenize(
    name: str,
    dataset: InputName | ExecutorStep | str | HfDatasetSpec,
    tokenizer: str,
    format: DNALmDatasetFormat,  # narrowed from LmDatasetFormatBase for the parity check
    *,
    sample_count: int | VersionedValue[int] | None = None,
    is_validation: bool = False,
    levanter_batch_size: int | None = None,
    tags: tuple[str, ...] = (),
    resources=None,
    worker_resources=None,
) -> ExecutorStep:
    extra_kwargs: dict = {}
    if worker_resources is not None:
        extra_kwargs["worker_resources"] = worker_resources

    if isinstance(dataset, HfDatasetSpec):
        config = HfTokenizeConfig(
            id=dataset.id,
            name=dataset.name,
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            tags=[*tags],
            **extra_kwargs,
        )
    elif (
        isinstance(dataset, str)
        and not _is_hf_bucket_path(dataset)
        and dataset.count("/") == 1
        and not fsspec_utils.exists(dataset)
    ):
        config = HfTokenizeConfig(
            id=dataset,
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            tags=[*tags],
            **extra_kwargs,
        )
    else:
        config = TokenizeConfig(
            train_paths=[dataset] if not is_validation else [],
            validation_paths=[dataset] if is_validation else [],
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            tags=[*tags],
            **extra_kwargs,
        )

    return ExecutorStep(
        name=os.path.join("tokenized", name),
        description=f"Tokenize raw text using the {tokenizer} tokenizer.",
        fn=remote(
            tokenize,
            resources=resources or ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"),
            # Upstream uses ``["cpu"]`` here, which maps to marin's own ``cpu``
            # extra (marin + jax-cpu + torch-cpu). bolinas-dna doesn't define a
            # ``cpu`` extra, so we route to the ``marin`` extra instead — it
            # transitively installs marin + jax + jmp + tokenizers, which is
            # what tokenization actually needs.
            pip_dependency_groups=["marin"],
            env_vars={
                "TRANSFORMERS_NO_TORCH": "1",
                "TRANSFORMERS_NO_TORCHVISION": "1",
                "USE_TORCH": "0",
                "TORCH_DISABLE_GLOBAL_DEPS": "1",
                # huggingface_hub's default `read_timeout=10s` is too short for
                # this dataset (`bolinas-dna/zoonomia-v1-v2`, ~107M rows) — the
                # API call to fetch the parquet manifest times out and fails
                # the whole tokenization. Bump to 120s.
                "HF_HUB_DOWNLOAD_TIMEOUT": "120",
                # Many concurrent zephyr workers on a single VM all run
                # `uv sync --extra marin`, which git-builds `lm-eval` (no PyPI
                # release of marin-levanter's pinned fork). uv's per-cache-key
                # lock has a 300s default; under contention, several workers
                # time out and fail the whole pipeline. Bump to 30min so the
                # lock just queues quietly while the first worker builds.
                "UV_LOCK_TIMEOUT": "1800",
            },
        ),
        config=config,
    )


if __name__ == "__main__":
    main()
