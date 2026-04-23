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


def compute_evo2_llr(
    model_name: str,
    dataset: pd.DataFrame,
    genome_path: str | Path,
    window_size: int = 8192,
    batch_size: int = 8,
    num_workers: int = 8,
    data_transform_on_the_fly: bool = True,
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
        batch_size: Per-device eval batch size passed through to HF Trainer.
        num_workers: Dataloader workers.
        data_transform_on_the_fly: Forwarded to ``run_llr_clm``.

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
    tokenizer = Evo2Tokenizer(_model.tokenizer)
    genome = Genome(str(genome_path))

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
