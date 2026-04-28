"""Evo2 inference wrappers via biofoundation's run_llr_clm / run_ll_clm.

Thin wrappers around the Evo2CausalLM + Evo2Tokenizer adapters:

- ``compute_evo2_llr`` — per-variant log-likelihood ratio (alt - ref) over a
  genome-anchored window. Used by issue #131's TraitGym VEP eval.
- ``compute_evo2_ll`` — per-sequence log-likelihood with case-based breakdown
  (uppercase=phyloP-functional, lowercase=non-functional), via biofoundation's
  PR #18 ``run_ll_clm``. Used by the LL-gap follow-up eval.

Note: the LOCAL_RANK -> CUDA_VISIBLE_DEVICES guard from
biofoundation/examples/evo2_llr.py MUST run before importing torch / evo2 /
biofoundation. Do it at the top of the entry script, not here.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def find_max_batch_size(
    model,
    window_size: int = 8192,
    start: int = 64,
    vocab_size: int = 512,
    seq_factor: int = 2,
) -> int:
    """OOM-descent: start at ``start``, halve on CUDA OOM until a forward
    pass through ``model`` survives. HF Trainer's ``auto_find_batch_size`` is
    a no-op for ``predict()``, so we tune explicitly.

    ``seq_factor`` is the number of sequences per logical "row" the
    downstream pipeline pushes through the model in one batch:

    - ``2`` (default) for ``compute_llr_clm`` — flattens ``[B, 2, L]``
      (ref+alt per variant) into ``[B*2, L]`` before the fwd pass.
    - ``1`` for ``compute_ll_clm`` — one sequence per row.
    """
    import torch

    # For sharded Vortex models (40B on 2 GPUs) the embedding layer may live
    # on a non-cuda:0 device. Send the probe input to whichever device the
    # first parameter is on, matching what biofoundation's Trainer pathway
    # does at inference time.
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


def _load_evo2_for_inference(model_name: str):
    """Construct ``(model, tokenizer)`` ready for biofoundation inference.

    Wraps the Evo2CausalLM ``forward`` so input_ids are routed to the
    embedding layer's device and logits are returned on the caller's device.
    On a single GPU this is a no-op; on a sharded Vortex model (40B across
    multiple GPUs) the embedding layer can land on a non-cuda:0 device and
    the logits emerge on the last shard's device — biofoundation's scoring
    code expects them collocated, so we shim the boundary here.
    """
    from biofoundation.model.adapters.evo2 import Evo2CausalLM, Evo2Tokenizer
    from evo2 import Evo2

    _model = Evo2(model_name)
    model = Evo2CausalLM(_model)

    _embed_device = _model.model.embedding_layer.weight.device
    _orig_forward = model.forward

    def _sharded_forward(input_ids):
        caller_device = input_ids.device
        if input_ids.device != _embed_device:
            input_ids = input_ids.to(_embed_device)
        logits = _orig_forward(input_ids)
        if logits.device != caller_device:
            logits = logits.to(caller_device)
        return logits

    model.forward = _sharded_forward
    tokenizer = Evo2Tokenizer(_model.tokenizer)
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
        data_transform_on_the_fly: Forwarded to ``run_llr_clm``.
        tune_start: Initial batch size to probe when ``batch_size is None``.

    Returns:
        1-D float numpy array of LLR values (alt_logprob - ref_logprob),
        length equal to ``len(dataset)``.
    """
    from biofoundation.data import Genome
    from biofoundation.inference import run_llr_clm
    from datasets import Dataset

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

    llr = run_llr_clm(
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

    llr = np.asarray(llr).reshape(-1)
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
    dataset,
    window_size: int = 255,
    batch_size: int | None = None,
    num_workers: int = 4,
    tune_start: int = 512,
) -> np.ndarray:
    """Compute per-sequence log-likelihood (with case-based breakdown) using
    an Evo2 checkpoint.

    Wraps biofoundation's ``run_ll_clm`` (PR #18). The dataset's ``seq``
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
    from biofoundation.inference import run_ll_clm

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
    # squeezed to ``[4]`` somewhere along the gather path. Reshape if
    # flat; row-major reshape preserves the per-row order.
    if pred.ndim == 1 and pred.shape[0] == len(dataset) * 4:
        pred = pred.reshape(len(dataset), 4)
    assert pred.ndim == 2 and pred.shape == (len(dataset), 4), (
        f"LL pred shape mismatch: got {pred.shape}, expected ({len(dataset)}, 4)"
    )
    assert np.isfinite(pred).all(), "non-finite values in Evo2 LL prediction"
    return pred


def aggregate_ll_gap(pred: np.ndarray) -> dict[str, float]:
    """Collapse a ``[N, 4]`` per-row LL prediction into dataset-wide
    token-weighted means and the LL gap.

    Per biofoundation's PR #18 example: cast to fp64 *before* summing —
    fp32 accumulation drift over ~10^6 target tokens is non-trivial.

    Args:
        pred: ``[N, 4]`` of ``(ll_sum_upper, ll_sum_lower, n_upper, n_lower)``.

    Returns:
        Dict with ``LL_all``, ``LL_upper``, ``LL_lower``, ``gap``,
        ``n_upper``, ``n_lower``. ``LL_*`` are mean log-likelihoods per
        target token (negative; closer to 0 is better — biofoundation's
        ``compute_ll_clm`` returns raw ``log p``, not NLL). The gap follows
        biofoundation's example convention: ``gap = LL_upper - LL_lower``,
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
