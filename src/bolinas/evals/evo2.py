"""Variant LLR scoring for Evo2 models via biofoundation's run_llr_clm.

Thin wrapper that mirrors src/bolinas/evals/inference.py:compute_variant_scores
but swaps the HF AutoModelForCausalLM loader for the Evo2CausalLM adapter.
Returns LLR only (embeddings are out of scope for this pass).

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
) -> int:
    """OOM-descent: start at ``start``, halve on CUDA OOM until a forward
    pass through ``model`` survives. HF Trainer's ``auto_find_batch_size`` is
    a no-op for ``predict()``, so we tune explicitly.

    The dummy input has shape ``[B*2, L]`` to match ``compute_llr_clm`` which
    flattens the ``[B, 2, L]`` (ref+alt per variant) tensor before the fwd
    pass.
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
            x = torch.randint(0, vocab_size, (bs * 2, window_size), device=probe_device)
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
    from biofoundation.model.adapters.evo2 import Evo2CausalLM, Evo2Tokenizer
    from datasets import Dataset
    from evo2 import Evo2

    genome_path = Path(genome_path)
    assert genome_path.exists(), f"genome not found: {genome_path}"

    _model = Evo2(model_name)
    model = Evo2CausalLM(_model)

    # For sharded Vortex models (40B on 2 GPUs), the embedding layer may
    # land on a non-cuda:0 device, AND the model's output logits come out on
    # the last shard's device. HF Trainer / biofoundation's scoring fns
    # expect input_ids and logits to live on the same device (torch.gather).
    # Wrap the adapter's forward so:
    #   1. input_ids are moved to the embedding's device (input side)
    #   2. logits are moved back to the caller's device (output side)
    # Both are no-ops in the single-GPU case.
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
    genome = Genome(str(genome_path))

    if batch_size is None:
        batch_size = find_max_batch_size(model, window_size=window_size, start=tune_start)
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
