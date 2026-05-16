"""Custom lm-eval-harness tasks for DNA experiments.

This package's directory is the ``include_path`` passed to lm-eval at runtime.
``importlib.resources.files("bolinas.pipelines.evals.lm_eval")`` resolves to the on-disk
location of this directory (uv installs bolinas in editable mode), letting
lm-eval scan it for task YAMLs and import the referenced ``!function`` classes.

Importing this module patches ``lm_eval.tasks.TaskManager`` so it always
includes this directory on its search path. Marin's current
``LmEvalHarnessConfig`` no longer accepts ``include_path`` (the kwarg was
dropped between dna-dev and main), so without this patch the custom tasks
defined here are unreachable from a marin-launched run. The patch applies
once, idempotently, only when lm-eval is installed (i.e. under the ``marin``
dependency group).

Also patches ``levanter.eval_harness.LmEvalHarnessConfig._rename_tasks_for_eval_harness``
to handle plain ``lm_eval.api.task.Task`` subclasses (rather than only
``ConfigurableTask`` + dict). Our ``DnaVepLlrEvalTask`` extends ``Task``
directly because ``ConfigurableTask``'s ``__init__`` eagerly walks fewshot
docs and hangs on large datasets; the rename logic for plain Tasks is a
no-op (our ``self._task_name`` is already correct, no per-task aliasing
needed).
"""

import importlib.resources
from functools import wraps


def _install_task_manager_patch() -> None:
    try:
        from lm_eval.tasks import TaskManager
    except ImportError:
        # lm-eval not installed (default `uv sync`). Nothing to patch.
        return

    if getattr(TaskManager, "_bolinas_dna_patched", False):
        return

    bolinas_dna_path = str(importlib.resources.files(__name__))
    original_init = TaskManager.__init__

    @wraps(original_init)
    def patched_init(self, *args, include_path=None, **kwargs):
        if include_path is None:
            merged = [bolinas_dna_path]
        elif isinstance(include_path, str):
            merged = [include_path, bolinas_dna_path]
        else:
            merged = list(include_path) + [bolinas_dna_path]
        return original_init(self, *args, include_path=merged, **kwargs)

    TaskManager.__init__ = patched_init
    TaskManager._bolinas_dna_patched = True


def _install_levanter_rename_patch() -> None:
    """Teach ``levanter.eval_harness.LmEvalHarnessConfig._rename_tasks_for_eval_harness``
    about plain ``Task`` subclasses.

    The upstream method only handles ``dict`` and ``ConfigurableTask`` and raises
    ``ValueError: Unknown task type`` for anything else — which traps any
    custom task class (like ``DnaVepLlrEvalTask``) that extends ``Task``
    directly. Wrap it so a plain ``Task`` instance falls through as a no-op
    (no rename needed — the task already has the right name via its YAML
    ``task:`` field, and we don't use multi-instance fewshot variants).
    """
    try:
        from levanter.eval_harness import LmEvalHarnessConfig
        from lm_eval.api.task import Task
    except ImportError:
        return

    if getattr(LmEvalHarnessConfig, "_bolinas_dna_rename_patched", False):
        return

    original_rename = LmEvalHarnessConfig._rename_tasks_for_eval_harness

    @wraps(original_rename)
    def patched_rename(self, this_task, lm_eval_task_name, our_name):
        # Plain Task subclasses (non-ConfigurableTask) pass through unchanged.
        # Our DnaVepLlrEvalTask doesn't have a `config.task` to rename and the
        # rename hack is only needed for multi-instance fewshot variants of
        # the same lm-eval task — which we don't use.
        from lm_eval.api.task import ConfigurableTask

        if (
            not isinstance(this_task, dict)
            and isinstance(this_task, Task)
            and not isinstance(this_task, ConfigurableTask)
        ):
            return this_task
        return original_rename(self, this_task, lm_eval_task_name, our_name)

    LmEvalHarnessConfig._rename_tasks_for_eval_harness = patched_rename
    LmEvalHarnessConfig._bolinas_dna_rename_patched = True


_install_task_manager_patch()
_install_levanter_rename_patch()
