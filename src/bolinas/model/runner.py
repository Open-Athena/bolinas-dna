"""HF inference harness for CLM variant/sequence scoring.

Vendored from biofoundation/inference.py at commit 834dd4c (May 2026),
CLM-only and rewritten to operate directly on HF objects — no
``Tokenizer`` / ``CausalLM`` abstract base classes, no adapter wrappers.

**Expected model/tokenizer interface (duck-typed):**

- ``model``: a callable that, given ``input_ids``, returns an object with
  a ``.logits`` attribute (shape ``[B, L, V]``). When called with
  ``output_hidden_states=True``, must also return ``.hidden_states`` — a
  tuple of layer outputs indexable with ``[-1]`` (last) and
  ``[len // 2]`` (middle). HF ``AutoModelForCausalLM`` satisfies this
  natively. Non-HF models (e.g. Evo2) can be wrapped to expose the same
  surface — see ``pipelines/evals/evo2.py`` for an example.

- ``tokenizer``: any object exposing ``.encode(text) -> list[int]``,
  ``.bos_token_id``, ``.eos_token_id``. HF ``PreTrainedTokenizerBase``
  satisfies this. The tokenizer is only used by the data transform step
  (``bolinas.data.transforms``), not the forward pass.
"""

from __future__ import annotations

import tempfile
from functools import partial
from typing import Any, Callable, Literal

import datasets
import numpy as np
import torch.nn as nn
from transformers import Trainer, TrainingArguments

import torch

from bolinas.data.dna import NUCLEOTIDES
from bolinas.data.genome import Genome
from bolinas.data.transforms import (
    _get_nucleotide_token_ids,
    transform_ll_clm,
    transform_llr_clm,
)
from bolinas.model.scoring import (
    compute_ll_clm,
    compute_llr_clm,
    compute_variant_score_bundle,
)


def run_inference(
    model: nn.Module,
    tokenizer: Any,
    dataset: datasets.Dataset,
    compute_fn: Callable[..., Any],
    data_transform_fn: Callable[..., dict[str, Any]] | None = None,
    data_transform_on_the_fly: bool = False,
    data_transform_kwargs: dict[str, Any] | None = None,
    inference_kwargs: dict[str, Any] | None = None,
) -> Any:
    processed_dataset = _process_dataset(
        dataset,
        tokenizer,
        data_transform_fn,
        data_transform_on_the_fly,
        data_transform_kwargs,
    )
    return _run_inference(
        _ModelComputeFnWrapper(model, compute_fn),
        processed_dataset,
        **(inference_kwargs or {}),
    )


def _run_strand_aware(
    model: nn.Module,
    tokenizer: Any,
    dataset: datasets.Dataset,
    *,
    compute_fn: Callable[..., Any],
    transform_fn: Callable[..., dict[str, Any]],
    transform_kwargs: dict[str, Any] | None = None,
    rc_avg: bool = False,
    **kwargs: Any,
) -> Any:
    """Run inference once on the forward strand; if ``rc_avg=True``, also run
    on the reverse-complemented strand and return the element-wise mean.

    ``strand`` is bound into ``transform_fn`` via ``partial``. Averaging is
    element-wise on numpy arrays, so it works for shape ``[N]`` (e.g.
    ``run_llr_clm``) and ``[N, 4]`` (``run_variant_score_bundle``).
    """

    def _one(strand: Literal["+", "-"]) -> Any:
        return run_inference(
            model,
            tokenizer,
            dataset,
            compute_fn=compute_fn,
            data_transform_fn=partial(
                transform_fn, strand=strand, **(transform_kwargs or {})
            ),
            **kwargs,
        )

    fwd = _one("+")
    if not rc_avg:
        return fwd
    rc = _one("-")
    return (np.asarray(fwd) + np.asarray(rc)) / 2


run_ll_clm = partial(
    run_inference,
    compute_fn=compute_ll_clm,
    data_transform_fn=transform_ll_clm,
)


def run_llr_clm(
    model: nn.Module,
    tokenizer: Any,
    dataset: datasets.Dataset,
    genome: Genome,
    window_size: int,
    rc_avg: bool = False,
    **kwargs: Any,
) -> Any:
    return _run_strand_aware(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_llr_clm,
        transform_fn=transform_llr_clm,
        transform_kwargs=dict(genome=genome, window_size=window_size),
        rc_avg=rc_avg,
        **kwargs,
    )


def run_variant_score_bundle(
    model: nn.Module,
    tokenizer: Any,
    dataset: datasets.Dataset,
    genome: Genome,
    window_size: int,
    rc_avg: bool = False,
    **kwargs: Any,
) -> Any:
    """Run the variant-score bundle (LLR + embedding distances + next-token JSD).

    Computes all four scores in a single forward pass:

    - LLR (log-likelihood ratio)
    - Last-layer embedding L2 distance
    - Middle-layer embedding L2 distance
    - ``next_token_jsd_mean`` — per-position 4-nucleotide softmax JSD between
      REF and ALT next-token predictions, averaged over downstream positions
      (called ``down_jsd_mean`` in Open-Athena/bolinas-dna#175).

    Nucleotide token IDs are derived from the tokenizer via
    ``_get_nucleotide_token_ids`` (which handles BOS/n_prefix offset) and
    bound into the compute function as a tensor in ``NUCLEOTIDES`` order.

    Args:
        model: HF-shaped causal LM (see module docstring for interface).
        tokenizer: Tokenizer for the model.
        dataset: Dataset with variant information (chrom, pos, ref, alt).
        genome: Genome object for sequence extraction.
        window_size: Window size for sequence context.
        rc_avg: If True, also score the reverse-complemented window for
            each variant and return the element-wise average of FWD and
            RC predictions (shape ``[N, 4]``). Doubles inference cost.
            Under RC, the AR mask runs in token order which is reversed
            on the RC strand — so FWD captures the genomic-downstream
            half of the variant's effect footprint and RC captures the
            genomic-upstream half (see issue #175 conclusion 9). The
            average is bidirectional nuc-dep despite the unidirectional
            AR model.
        **kwargs: Additional arguments passed to run_inference.

    Returns:
        Numpy array with shape [B, 4] where columns are:
            - [:, 0]: LLR
            - [:, 1]: Last-layer embedding distance
            - [:, 2]: Middle-layer embedding distance
            - [:, 3]: next_token_jsd_mean
    """
    nuc_ids_dict = _get_nucleotide_token_ids(tokenizer)
    nuc_token_ids = torch.tensor(
        [nuc_ids_dict[nuc] for nuc in NUCLEOTIDES], dtype=torch.long
    )
    return _run_strand_aware(
        model,
        tokenizer,
        dataset,
        compute_fn=partial(compute_variant_score_bundle, nuc_token_ids=nuc_token_ids),
        transform_fn=transform_llr_clm,
        transform_kwargs=dict(genome=genome, window_size=window_size),
        rc_avg=rc_avg,
        **kwargs,
    )


def _run_inference(
    model: nn.Module,
    dataset: datasets.Dataset,
    **kwargs: Any,
) -> Any:
    """Run inference on a dataset using a trained model via HF Trainer.

    Args:
        model: A trained PyTorch model that can be used with the HuggingFace Trainer.
        dataset: HuggingFace dataset to run inference on. The dataset should be
            compatible with the model's expected input format.
        **kwargs: Additional keyword arguments to pass to TrainingArguments.
            Common options include:
            - per_device_eval_batch_size: Batch size for evaluation
            - dataloader_num_workers: Number of workers for data loading
            - torch_compile: Whether to use torch.compile for faster inference
            - bf16_full_eval: Whether to use bf16 for evaluation

    Returns:
        The model's predictions on the dataset. The exact format depends on the
        model and dataset, but typically includes probabilities or embeddings.
    """
    # HF Trainer requires an output_dir even for .predict() (it never
    # writes to it on the predict path). Use a tempdir scoped to this
    # call so it's cleaned up deterministically.
    with tempfile.TemporaryDirectory() as output_dir:
        training_args = TrainingArguments(
            output_dir=output_dir,
            **(kwargs or {}),
        )
        trainer = Trainer(model=model, args=training_args)
        return trainer.predict(test_dataset=dataset).predictions


class _ModelComputeFnWrapper(nn.Module):
    """Adapt ``compute_fn(model, batch)`` into an ``nn.Module.forward`` so
    HF Trainer can call it as if it were the model."""

    def __init__(self, model: nn.Module, compute_fn: Callable[..., Any]):
        super().__init__()
        self.model = model
        self.compute_fn = compute_fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.compute_fn(self.model, *args, **kwargs)


def _process_dataset(
    dataset: datasets.Dataset,
    tokenizer: Any,
    data_transform_fn: Callable[..., dict[str, Any]] | None = None,
    data_transform_on_the_fly: bool = False,
    data_transform_kwargs: dict[str, Any] | None = None,
) -> datasets.Dataset:
    if data_transform_fn is None:
        return dataset
    if data_transform_kwargs is None:
        data_transform_kwargs = {}
    data_transform_fn = partial(data_transform_fn, tokenizer=tokenizer)
    if data_transform_on_the_fly:
        return dataset.with_transform(
            _make_batch_transform(data_transform_fn),
            **data_transform_kwargs,
        )
    return dataset.map(
        data_transform_fn,
        **data_transform_kwargs,
    )


def _make_batch_transform(
    transform_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, list[Any]]], dict[str, list[Any]]]:
    def batch_transform_fn(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # Convert batch format to list of examples
        examples = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        # Apply transform to each example
        transformed_examples = [transform_fn(example) for example in examples]
        # Convert back to batch format
        return {
            key: [ex[key] for ex in transformed_examples]
            for key in transformed_examples[0].keys()
        }

    return batch_transform_fn
