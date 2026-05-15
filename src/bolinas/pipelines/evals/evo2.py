"""Evo2 inference wrappers reusing bolinas.model.runner.

Provides:

- ``compute_evo2_llr`` — per-variant log-likelihood ratio (alt - ref) over a
  genome-anchored window. Used by issue #131's TraitGym VEP eval.
- ``compute_evo2_ll`` — per-sequence log-likelihood with case-based breakdown
  (uppercase=phyloP-functional, lowercase=non-functional). Used by the LL-gap
  follow-up eval.

Evo2 quack-wrappers live here (not in ``bolinas.model.*``): they shim Evo2's
native API into the duck-typed interface that ``bolinas.model.runner``
expects (a callable returning an object with ``.logits``, plus a tokenizer
with ``.encode/.bos_token_id/.eos_token_id``). Core stays HF-only.

Note: the LOCAL_RANK -> CUDA_VISIBLE_DEVICES guard from
biofoundation/examples/evo2_llr.py MUST run before importing torch / evo2.
Do it at the top of the entry script, not here.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch.nn as nn

if TYPE_CHECKING:
    from datasets import Dataset


# Single source of truth for argparse `choices` in scripts/evo2_eval/. Display
# ordering for the metrics aggregator (size-sorted) is a separate concern;
# see scripts/evo2_eval/traitgym_v2_metrics.py:MODELS.
EVO2_MODEL_CHOICES: tuple[str, ...] = (
    "evo2_1b_base",
    "evo2_7b",
    "evo2_7b_base",
    "evo2_40b",
    "evo2_40b_base",
    "evo2_20b",
)


class _Evo2QuackModel(nn.Module):
    """Wrap Evo2 so it satisfies ``bolinas.model.runner``'s duck-typed model
    interface: ``model(input_ids)`` returns an object with ``.logits``.

    Also handles sharded (multi-GPU) Vortex models where the embedding layer
    can land on a non-``cuda:0`` device — routes ``input_ids`` to the embed
    device on entry and logits back to the caller's device on exit.
    """

    def __init__(self, evo2_model: Any):
        super().__init__()
        self._model = evo2_model
        try:
            self._embed_device = evo2_model.model.embedding_layer.weight.device
        except AttributeError:
            self._embed_device = None

    def forward(self, input_ids: Any, **kwargs: Any) -> SimpleNamespace:
        caller_device = input_ids.device
        if self._embed_device is not None and input_ids.device != self._embed_device:
            input_ids = input_ids.to(self._embed_device)
        # Evo2 returns (outputs, embeddings) tuple; outputs[0] is logits.
        outputs, _ = self._model(input_ids)
        logits = outputs[0]
        if logits.device != caller_device:
            logits = logits.to(caller_device)
        return SimpleNamespace(logits=logits)


class _Evo2QuackTokenizer:
    """Wrap vortex's ``CharLevelTokenizer`` so it satisfies
    ``bolinas.data.transforms``'s duck-typed tokenizer interface.

    CharLevelTokenizer has no BOS/EOS — we expose them as ``None``, which
    ``_get_special_token_counts`` handles correctly (counts=0).
    """

    def __init__(self, char_tokenizer: Any):
        self._t = char_tokenizer

    def encode(self, text: str) -> list[int]:
        return list(map(int, self._t.tokenize(text)))

    @property
    def bos_token_id(self) -> int | None:
        return None

    @property
    def eos_token_id(self) -> int | None:
        return None


def find_max_batch_size(
    model: nn.Module,
    window_size: int = 8192,
    start: int = 64,
    vocab_size: int = 512,
    seq_factor: int = 2,
) -> int:
    """Largest batch size in ``{start, start//2, start//4, ...}`` that
    survives a probe forward pass — *not* a true maximum. We halve from
    ``start`` on each CUDA OOM, so the result can be up to 2× smaller
    than the true maximum. HF Trainer's ``auto_find_batch_size`` is a
    no-op for ``predict()``, so we tune explicitly.

    ``seq_factor`` is the number of sequences per logical "row" the
    downstream pipeline pushes through the model in one batch:

    - ``2`` (default) for ``compute_variant_score_bundle`` — one ref-suffix
      and one alt-suffix forwarded per variant (the shared prefix is
      forwarded once via KV-cache; only the suffixes hit the per-row
      multiplier).
    - ``1`` for ``compute_ll_clm`` — one sequence per row.
    """
    import torch

    # For sharded Vortex models (40B on 2 GPUs) the embedding layer may live
    # on a non-cuda:0 device. Send the probe input to whichever device the
    # first parameter is on, matching what the inference pathway does.
    try:
        probe_device = next(model.parameters()).device
    except StopIteration:
        probe_device = torch.device("cuda:0")
    bs = start
    while bs >= 1:
        try:
            x = torch.randint(
                0, vocab_size, (bs * seq_factor, window_size), device=probe_device
            )
            with torch.inference_mode():
                _ = model(x)
            torch.cuda.empty_cache()
            return bs
        except RuntimeError as e:
            # Catch both OutOfMemoryError *and* 32-bit index overflow
            # (canUse32BitIndexMath). The latter kicks in around bs*L*hidden ≳ 2^31
            # — e.g. bs=64 × 8192ctx × 2048hidden for 1B. Either way, halve.
            msg = str(e)
            if "out of memory" not in msg.lower() and "32BitIndexMath" not in msg:
                raise
            torch.cuda.empty_cache()
            bs //= 2
    raise RuntimeError("Even batch_size=1 doesn't fit — check model/GPU.")


def _load_evo2_for_inference(
    model_name: str,
) -> tuple[_Evo2QuackModel, _Evo2QuackTokenizer]:
    """Construct ``(model, tokenizer)`` ready for ``bolinas.model.runner``.

    Both wrappers shim Evo2 into the HF-shaped duck-typed interface that
    ``bolinas.model.runner.run_variant_score_bundle`` / ``run_ll_clm`` expect; the model
    wrapper also handles sharded multi-GPU device routing transparently.
    """
    from evo2 import Evo2

    _model = Evo2(model_name)
    model = _Evo2QuackModel(_model)
    tokenizer = _Evo2QuackTokenizer(_model.tokenizer)
    return model, tokenizer


def compute_evo2_llr(
    model_name: str,
    dataset: pd.DataFrame,
    genome_path: str | Path,
    window_size: int = 8192,
    batch_size: int | None = None,
    num_workers: int = 8,
    data_transform_on_the_fly: bool = True,
    tune_start: int = 64,
) -> np.ndarray:
    """Compute per-variant LLR using an Evo2 checkpoint.

    Args:
        model_name: One of ``evo2_1b_base``, ``evo2_7b``, ``evo2_40b`` (or any
            name accepted by ``evo2.Evo2``).
        dataset: DataFrame with columns ``[chrom, pos, ref, alt]``. Row order
            of the output aligns with row order of this input.
        genome_path: Path to genome reference FASTA (.fa / .fa.gz).
        window_size: Context length in bp. Fixed to 8192 for the issue #131
            eval per user request.
        batch_size: Per-device eval batch size. If ``None`` (default), run an
            OOM-descent tune starting from ``tune_start`` and pick the largest
            that survives a forward pass.
        num_workers: Dataloader workers.
        data_transform_on_the_fly: Forwarded to ``run_variant_score_bundle``.
        tune_start: Initial batch size to probe when ``batch_size is None``.

    Returns:
        1-D float numpy array of LLR values (alt_logprob - ref_logprob),
        length equal to ``len(dataset)``.
    """
    from datasets import Dataset

    from bolinas.data.genome import Genome
    from bolinas.model.runner import run_variant_score_bundle

    genome_path = Path(genome_path)
    assert genome_path.exists(), f"genome not found: {genome_path}"

    model, tokenizer = _load_evo2_for_inference(model_name)
    genome = Genome(str(genome_path))

    if batch_size is None:
        batch_size = find_max_batch_size(
            model, window_size=window_size, start=tune_start, seq_factor=2
        )
        print(f"[evo2] tuned batch_size={batch_size} (start={tune_start})")

    hf_dataset = Dataset.from_pandas(dataset, preserve_index=False)

    # Bundle returns [N, 2] = (LLR, next_token_jsd_mean); we only need LLR.
    bundle = run_variant_score_bundle(
        model,
        tokenizer,
        hf_dataset,
        genome,
        window_size,
        data_transform_on_the_fly=data_transform_on_the_fly,
        inference_kwargs=dict(
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=num_workers,
            remove_unused_columns=False,
        ),
    )

    llr = np.asarray(bundle)[:, 0]
    assert llr.shape == (len(dataset),), (
        f"LLR shape mismatch: got {llr.shape}, expected ({len(dataset)},)"
    )
    assert np.isfinite(llr).all(), "non-finite LLR values produced by Evo2"
    return llr


def scores_dataframe(llr: np.ndarray) -> pd.DataFrame:
    """Expand a 1-D LLR array into the score DataFrame the rest of the eval code expects."""
    llr = np.asarray(llr).reshape(-1)
    return pd.DataFrame(
        {
            "llr": llr,
            "minus_llr": -llr,
            "abs_llr": np.abs(llr),
        }
    )


def compute_evo2_ll(
    model_name: str,
    dataset: "Dataset",
    window_size: int = 255,
    batch_size: int | None = None,
    num_workers: int = 4,
    tune_start: int = 512,
) -> np.ndarray:
    """Compute per-sequence log-likelihood (with case-based breakdown) using
    an Evo2 checkpoint.

    Wraps ``bolinas.model.runner.run_ll_clm``. The dataset's ``seq``
    column drives ``transform_ll_clm``, which uppercases before tokenizing
    and emits an ``is_upper`` mask aligned to source positions; the
    downstream ``compute_ll_clm`` slices that mask to target positions and
    bucketises log-probs into upper/lower sums + counts.

    Args:
        model_name: One of ``evo2_1b_base``, ``evo2_7b_base``, etc.
        dataset: An ``datasets.Dataset`` with at minimum a ``seq`` column
          of mixed-case DNA strings. Row order of the output aligns with
          row order of this input.
        window_size: Used only by the OOM-descent batch-size tuner. The
          actual sequence length is whatever ``transform_ll_clm`` produces
          — body length plus a BOS/EOS *only if the tokenizer defines
          them*. Evo2's vortex CharLevelTokenizer does not, so for our CDS
          dataset (255-bp ``seq``) the input_ids land at length 255. After
          the causal shift, that's 254 target tokens per row. Default 255
          matches that.
        batch_size: Per-device eval batch size. ``None`` triggers
          OOM-descent tuning starting from ``tune_start``.
        num_workers: Dataloader workers.
        tune_start: Initial batch size for the tuner. Default 512 because
          this eval uses a short context (257 tokens) — the OOM-descent
          tuner only halves, never grows, so seeding it low here would
          settle at a wastefully small batch.

    Returns:
        ``[N, 4]`` float numpy array of per-row
        ``(ll_sum_upper, ll_sum_lower, n_upper, n_lower)``. Counts are
        token-level on the *target* positions (length ``L-1`` after the
        causal shift). Aggregate across rows with ``aggregate_ll_gap``.
    """
    from bolinas.model.runner import run_ll_clm

    model, tokenizer = _load_evo2_for_inference(model_name)

    if batch_size is None:
        batch_size = find_max_batch_size(
            model, window_size=window_size, start=tune_start, seq_factor=1
        )
        print(f"[evo2] tuned batch_size={batch_size} (start={tune_start})")

    pred = run_ll_clm(
        model,
        tokenizer,
        dataset,
        data_transform_on_the_fly=True,
        inference_kwargs=dict(
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=num_workers,
            remove_unused_columns=False,
        ),
    )

    pred = np.asarray(pred)
    # On some HF Trainer / device combinations (observed on GH200 / aarch64
    # with evo2_40b) ``run_ll_clm`` returns a flat ``[N*4]`` array instead
    # of ``[N, 4]`` — likely a per-batch ``[1, 4]`` output that gets
    # squeezed to ``[4]`` somewhere along the gather path. Row-major
    # reshape preserves per-row order.
    if pred.ndim == 1 and pred.shape[0] == len(dataset) * 4:
        print(
            f"[evo2] WARNING: run_ll_clm returned flat shape {pred.shape}, "
            f"reshaping to ({len(dataset)}, 4). If this recurs, investigate."
        )
        pred = pred.reshape(len(dataset), 4)
    assert pred.ndim == 2 and pred.shape == (len(dataset), 4), (
        f"LL pred shape mismatch: got {pred.shape}, expected ({len(dataset)}, 4)"
    )
    assert np.isfinite(pred).all(), "non-finite values in Evo2 LL prediction"
    return pred


def aggregate_ll_gap(pred: np.ndarray) -> dict[str, float]:
    """Collapse a ``[N, 4]`` per-row LL prediction into dataset-wide
    token-weighted means and the LL gap.

    Cast to fp64 *before* summing — fp32 accumulation drift over ~10^6
    target tokens is non-trivial.

    Args:
        pred: ``[N, 4]`` of ``(ll_sum_upper, ll_sum_lower, n_upper, n_lower)``.

    Returns:
        Dict with ``LL_all``, ``LL_upper``, ``LL_lower``, ``gap``,
        ``n_upper``, ``n_lower``. ``LL_*`` are mean log-likelihoods per
        target token (negative; closer to 0 is better — ``compute_ll_clm``
        returns raw ``log p``, not NLL). ``gap = LL_upper - LL_lower``,
        positive when uppercase (functional) bases are easier to predict
        than lowercase.
    """
    pred = np.asarray(pred)
    assert pred.ndim == 2 and pred.shape[1] == 4, (
        f"expected [N, 4] pred, got {pred.shape}"
    )
    S_u, S_l, n_u, n_l = pred.astype(np.float64).sum(axis=0)
    assert n_u > 0, "no upper (functional) target tokens — check case mask"
    assert n_l > 0, "no lower (non-functional) target tokens — check case mask"
    LL_upper = float(S_u / n_u)
    LL_lower = float(S_l / n_l)
    LL_all = float((S_u + S_l) / (n_u + n_l))
    return {
        "LL_all": LL_all,
        "LL_upper": LL_upper,
        "LL_lower": LL_lower,
        "gap": LL_upper - LL_lower,
        "n_upper": int(n_u),
        "n_lower": int(n_l),
    }
