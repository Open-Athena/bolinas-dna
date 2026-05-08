"""Custom lm-eval-harness tasks for DNA experiments.

This package's directory is the ``include_path`` passed to lm-eval at runtime.
``importlib.resources.files("bolinas.evals.lm_eval")`` resolves to the on-disk
location of this directory (uv installs bolinas in editable mode), letting
lm-eval scan it for task YAMLs and import the referenced ``!function`` classes.

Importing this module patches ``lm_eval.tasks.TaskManager`` so it always
includes this directory on its search path. Marin's current
``LmEvalHarnessConfig`` no longer accepts ``include_path`` (the kwarg was
dropped between dna-dev and main), so without this patch the custom tasks
defined here are unreachable from a marin-launched run. The patch applies
once, idempotently, only when lm-eval is installed (i.e. under the ``marin``
dependency group).
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
            merged: list[str] = [bolinas_dna_path]
        elif isinstance(include_path, str):
            merged = [include_path, bolinas_dna_path]
        else:
            merged = list(include_path) + [bolinas_dna_path]
        return original_init(self, *args, include_path=merged, **kwargs)

    TaskManager.__init__ = patched_init
    TaskManager._bolinas_dna_patched = True


_install_task_manager_patch()
