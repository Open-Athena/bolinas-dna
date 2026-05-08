"""Custom lm-eval-harness tasks for DNA experiments.

This package's directory is the ``include_path`` passed to lm-eval at runtime.
``importlib.resources.files("bolinas.evals.lm_eval")`` resolves to the on-disk
location of this directory (uv installs bolinas in editable mode), letting
lm-eval scan it for task YAMLs and import the referenced ``!function`` classes.
"""
