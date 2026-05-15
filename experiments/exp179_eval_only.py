# Copyright The Marin Authors / Bolinas Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval-only run of the online lm_eval VEP scorer (mendelian_traits_255)
on exp166-p1B — parity check for #179.

Loads the HF checkpoint ``bolinas-dna/exp166-p1B-step-16398`` on a TPU pod and
runs the ``mendelian_traits_255`` task from ``bolinas.pipelines.evals.lm_eval``.
The task scores every (FWD, RC) row of ``bolinas-dna/evals_mendelian_traits_harness_255``,
averages per ``(chrom, pos, ref, alt)``, and reports per-subset PairwiseAccuracy.
The same model + dataset are scored offline by ``snakemake/analysis/evals_v2/``
with ``inference.rc_avg=true``; the two PairwiseAccuracy numbers should match
within numerical precision.

Launch (from a CPU box with iris IAP tunnel open per ``experiments/README.md``):

    uv run iris --controller-url=http://localhost:10000 --cluster=marin job run \\
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

Then follow with ``iris job logs <id> -f``. Results land in WandB
(``dna-exp179-mendelian-traits-rc-eval-only`` group) and as a JSON artifact
``lm_eval_harness_results``.
"""

import logging
import os

from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    this_output_path,
    versioned,
)
from marin.execution.remote import remote

# Importing this package triggers ``_install_task_manager_patch()``, which
# monkeypatches ``lm_eval.tasks.TaskManager`` so the bolinas-dna custom-task
# directory is on its search path. Without that patch ``mendelian_traits_255``
# wouldn't be reachable from the marin-launched LevanterLmEvalEvaluator.
from bolinas.pipelines.evals.lm_eval.task_configs import MENDELIAN_TRAITS_255

logger = logging.getLogger(__name__)


# `bolinas-dna/exp166-p1B-step-16398` is the same checkpoint
# `snakemake/analysis/evals_v2/config/config.yaml` scores offline (entry
# `exp166-p1B`). Comparing the two PairwiseAccuracy numbers is the parity
# check this script exists for.
MODEL_NAME = "exp166-p1B-step-16398"
MODEL_PATH = "bolinas-dna/exp166-p1B-step-16398"

# v4-8 is the marin default for levanter eval (`default_eval` →
# `ResourceConfig.with_tpu("v4-8")`) and is broadly available across v4 quota
# pools. Switched away from v5p-8 (which exp160_parity used for training) on
# 2026-05-15 because v5p capacity was contested across both us-east5 and
# us-central1 — eval is inference-only so the bigger v5p chips aren't needed.
TPU_TYPES: tuple[str, ...] = ("v4-8",)


# Vendored from `marin-community/marin@main:experiments/evals/evals.py:142-169`.
# Marin's `experiments/` package isn't shipped (see exp160_parity.py:65-73 for
# why), and its `EVAL_DEPENDENCY_GROUPS = ["eval", "vllm", "tpu"]` references
# extras bolinas-dna doesn't define — substitute `["marin"]`, which transitively
# pulls lm-eval, levanter, jax, transformers (same as exp160_parity._tokenize).
_EVAL_DEPENDENCY_GROUPS = ["marin"]


def _evaluate_levanter_lm_evaluation_harness(
    *,
    model_name: str,
    model_path: str,
    evals,
    resource_config: ResourceConfig,
    env_vars: dict[str, str] | None = None,
) -> ExecutorStep:
    """Vendored helper: wrap ``marin.evaluation.run.evaluate`` in an ``ExecutorStep``
    that runs on TPU under iris.
    """
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness_levanter/exp179_{model_name}",
        fn=remote(
            evaluate,
            resources=resource_config,
            pip_dependency_groups=_EVAL_DEPENDENCY_GROUPS,
            env_vars=env_vars,
        ),
        config=EvaluationConfig(
            evaluator="levanter_lm_evaluation_harness",
            model_name=None,  # imputed by run.py from model_path
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=versioned(evals),
            discover_latest_checkpoint=False,  # HF revision is fixed by repo name
            resource_config=resource_config,
            wandb_tags=[
                "dna",
                "exp179",
                "mendelian-traits-rc-eval-only",
                "parity",
            ],
        ),
    )


def main() -> None:
    # Coordinator forwards these to iris-spawned workers; bolinas-dna marin
    # workers need them per `experiments/README.md` (HF parquet-manifest fetch
    # is slow at the default 10s timeout, and many concurrent uv syncs on a
    # single VM serialize on a single uv lock with a 300s default).
    env_vars: dict[str, str] = {
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "UV_LOCK_TIMEOUT": "7200",
    }
    if "WANDB_API_KEY" in os.environ:
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    step = _evaluate_levanter_lm_evaluation_harness(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        evals=[MENDELIAN_TRAITS_255],
        resource_config=ResourceConfig.with_tpu(TPU_TYPES),
        env_vars=env_vars,
    )
    executor_main(
        steps=[step],
        description=(
            "exp179 eval-only — mendelian_traits_255 (RC-strand averaged) on "
            "exp166-p1B; parity-check vs snakemake/analysis/evals_v2/."
        ),
    )


if __name__ == "__main__":
    main()
