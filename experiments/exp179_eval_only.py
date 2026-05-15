# Copyright The Marin Authors / Bolinas Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval-only run of the online lm_eval VEP scorer (mendelian_traits_255)
on exp166-p1B — parity check for #179.

Loads the HF checkpoint ``bolinas-dna/exp166-p1B-step-16398`` on a TPU pod and
runs the ``mendelian_traits_255`` task from ``bolinas.pipelines.evals.lm_eval``.
The task scores every (FWD, RC) row of ``bolinas-dna/evals_mendelian_traits_harness_255``,
emits per-subset PairwiseAccuracy split into FWD / RC / AVG, and reports
``_global_/avg/pairwise_accuracy`` as the headline scalar. The same model +
dataset are scored offline by ``snakemake/analysis/evals_v2/`` with
``inference.rc_avg=true``; the two AVG numbers should match within numerical
precision.

Why we bypass marin's ``LevanterLmEvalEvaluator``: that wrapper calls
``HFCheckpointConverter.from_hf(model_path)`` which walks levanter's model
registry and instantiates each candidate's ``hf_checkpoint_converter()``.
GemmaConfig's defaults to ``reference_checkpoint="google/gemma-2b"`` (gated
HF repo); the ``HFCheckpointConverter.__init__`` `_infer_tokenizer` call
fetches the gated tokenizer and 401s. So we construct the
``EvalHarnessMainConfig`` directly with an explicit ``model=Qwen3Config(...)``
matched to exp166's geometry, and call ``run_eval_harness_main`` ourselves —
the same shape the dna-branch precedent
``marin@dna:experiments/dna/smoke_tests/eval_traitgym.py`` uses.

Launch (from a CPU box with iris CLI authed to the marin cluster):

    uv run iris --cluster=marin job run \\
        --no-wait \\
        --user gonzalo \\
        --job-name exp179-eval-only \\
        --cpu 1 --memory 2g \\
        --extra marin \\
        --region us-east5 \\
        -e WANDB_API_KEY "$(grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}')" \\
        -e HF_HUB_DOWNLOAD_TIMEOUT 120 \\
        -e UV_LOCK_TIMEOUT 7200 \\
        -- python experiments/exp179_eval_only.py

Then follow with ``iris job logs <id> -f``.
"""

import dataclasses
import logging
import os

import jmp
import levanter.eval_harness as eval_harness
from fray.cluster import ResourceConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.evaluation.evaluation_config import convert_to_levanter_task_config
from marin.execution.executor import ExecutorStep, executor_main
from marin.execution.remote import remote

# Importing this package triggers ``_install_task_manager_patch()``, which
# monkeypatches ``lm_eval.tasks.TaskManager`` so the bolinas-dna custom-task
# directory is on its search path. Without that patch ``mendelian_traits_255``
# wouldn't be reachable from the iris-spawned worker.
from bolinas.levanter.defaults import dna_effective_seq_len
from bolinas.pipelines.evals.lm_eval.task_configs import MENDELIAN_TRAITS_255

logger = logging.getLogger(__name__)


# `bolinas-dna/exp166-p1B-step-16398` is the same checkpoint
# `snakemake/analysis/evals_v2/config/config.yaml` scores offline (entry
# `exp166-p1B`). Comparing the two PairwiseAccuracy numbers is the parity
# check this script exists for.
MODEL_NAME = "exp166-p1B-step-16398"
MODEL_PATH = "bolinas-dna/exp166-p1B-step-16398"
TOKENIZER = "bolinas-dna/tokenizer-char-bos"

# Qwen3 1B geometry derived in marin@dna:experiments/dna/exp166_zoonomia_1ep_scaling.py
# via CompletedAdamHHeuristic._build_model_config(hidden_size=1920). Verified
# inline so the eval doesn't need to import heuristic / scaling-sweep code.
DNA_BASE_SEQ_LEN = 255
HIDDEN_DIM = 1920
INTERMEDIATE_DIM = HIDDEN_DIM * 4  # mlp_ratio=4
NUM_HEADS = HIDDEN_DIM // 128  # hidden_head_ratio=128 → 15 heads
NUM_LAYERS = 19  # round(1920 / (64 + log2(1920)*4 - 9)) = 19

# v6e-4 has its own quota pool (separate from v5p-preemptible). Switched on
# 2026-05-15 because v5p-preemptible was stuck in the bad-node-signature
# retry loop in BOTH us-east5 and us-central1 across 3 attempts (r7/r8/r9).
# 1B-param eval inference fits in v6e-4 HBM (~31 GB/chip) comfortably.
TPU_TYPES: tuple[str, ...] = ("v6e-4",)

# bolinas-dna's `marin` extra transitively pulls lm-eval, levanter, jax,
# transformers — same substitution exp160_parity.py uses for its tokenize step
# (substituting marin's EVAL_DEPENDENCY_GROUPS=["eval","vllm","tpu"] which
# bolinas-dna doesn't define).
_EVAL_DEPENDENCY_GROUPS = ["marin"]

WANDB_PROJECT = "marin"
WANDB_RUN_NAME = "dna-exp179-mendelian-traits-rc-eval-only"
WANDB_TAGS = ("dna", "exp179", "mendelian-traits-rc-eval-only", "parity")


def _build_model_config(seq_len: int) -> Qwen3Config:
    """Qwen3 1B config matched to exp166-p1B's training geometry."""
    return Qwen3Config(
        hidden_dim=HIDDEN_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
        max_seq_len=seq_len,
        rope=Llama3RotaryEmbeddingsConfig(),
    )


@dataclasses.dataclass(frozen=True)
class _EvalConfig:
    """Empty placeholder — ExecutorStep requires a config dataclass; the eval
    invariants (model path, tokenizer, geometry) are captured as module-level
    constants and built by ``_run_eval_harness_only`` below.
    """


def _run_eval_harness_only(_config: _EvalConfig) -> None:
    """Worker-side function: construct EvalHarnessMainConfig with explicit
    Qwen3Config and run levanter's standalone eval-harness entrypoint.

    Runs on the TPU pod (called via remote()). Bypasses
    ``marin.evaluation.run.evaluate`` so we avoid the
    ``HFCheckpointConverter.from_hf`` registry walk that 401s on gated
    google/gemma-2b (see module docstring).
    """
    seq_len = dna_effective_seq_len(DNA_BASE_SEQ_LEN, TOKENIZER)
    eval_config = eval_harness.EvalHarnessMainConfig(
        eval_harness=eval_harness.LmEvalHarnessConfig(
            task_spec=convert_to_levanter_task_config([MENDELIAN_TRAITS_255]),
            log_samples=False,
        ),
        tokenizer=MODEL_PATH,  # exp166's HF repo bundles its tokenizer
        checkpoint_path=MODEL_PATH,
        checkpoint_is_hf=True,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=list(WANDB_TAGS),
                name=WANDB_RUN_NAME,
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            per_device_eval_parallelism=64,
        ),
        model=_build_model_config(seq_len),
    )
    eval_harness.run_eval_harness_main(eval_config)


def main() -> None:
    # Coordinator forwards these to the iris-spawned TPU worker; bolinas-dna
    # marin workers need them per `experiments/README.md` (HF parquet-manifest
    # fetch is slow at the default 10s timeout, and many concurrent uv syncs
    # on a single VM serialize on a uv lock with a 300s default).
    env_vars: dict[str, str] = {
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "UV_LOCK_TIMEOUT": "7200",
    }
    if "WANDB_API_KEY" in os.environ:
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    step = ExecutorStep(
        name=f"evaluation/lm_evaluation_harness_levanter/exp179_{MODEL_NAME}",
        fn=remote(
            _run_eval_harness_only,
            resources=ResourceConfig.with_tpu(TPU_TYPES),
            pip_dependency_groups=_EVAL_DEPENDENCY_GROUPS,
            env_vars=env_vars,
        ),
        config=_EvalConfig(),
    )
    executor_main(
        steps=[step],
        description=(
            "exp179 eval-only — mendelian_traits_255 (FWD/RC/AVG) on "
            "exp166-p1B; parity-check vs snakemake/analysis/evals_v2/."
        ),
    )


if __name__ == "__main__":
    main()
