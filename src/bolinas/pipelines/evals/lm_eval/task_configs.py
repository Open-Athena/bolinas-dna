# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DNA variant-effect-prediction (VEP) ``EvalTaskConfig`` constants.

Ported from ``marin-community/marin@dna-dev``: the
``TRAITGYM_MENDELIAN_V2_255`` constant in ``experiments/evals/task_configs.py``.
The general-purpose ``EvalTaskConfig`` and ``convert_to_levanter_task_config``
remain in marin (see ``experiments.evals.task_configs``).
"""

from marin.evaluation.evaluation_config import EvalTaskConfig

TRAITGYM_MENDELIAN_V2_255 = EvalTaskConfig(
    "traitgym_mendelian_v2_255", 0, task_alias="traitgym_mendelian_v2_255"
)
