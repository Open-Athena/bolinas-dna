"""Bolinas wrapper around ``compute_llr_and_pooled_embeddings``.

Returns a DataFrame with one row per variant (input order preserved) and the
columns documented in ``compute_pooled_features``.
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from biofoundation.data import Genome, transform_llr_clm
from biofoundation.inference import run_inference
from biofoundation.model.adapters.hf import HFCausalLMWithEmbeddings, HFTokenizer
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bolinas.supervised.scoring import compute_llr_and_pooled_embeddings


# Scalar score columns produced before the dense feature blocks. Two scalars,
# then three D-dim blocks (mean_ref, mean_alt, traitgym_innerprod).
NUM_SCALAR_FEATURES = 2


def compute_pooled_features(
    checkpoint_path: str | Path,
    dataset: pd.DataFrame,
    genome_path: str | Path,
    context_size: int,
    batch_size: int = 64,
    num_workers: int = 4,
    data_transform_on_the_fly: bool = True,
    torch_compile: bool = False,
) -> pd.DataFrame:
    """Extract per-variant zero-shot scalars + mean-pooled embeddings + TraitGym innerprod.

    Args:
        checkpoint_path: HF repo id (``bolinas-dna/exp166-p1B-step-16398``) or
            local path. Loaded via ``AutoModelForCausalLM.from_pretrained``.
        dataset: DataFrame with at minimum ``[chrom, pos, ref, alt]``. Row order
            is preserved and aligned with the returned DataFrame.
        genome_path: Path to a (g)zipped FASTA reference (GRCh38 release 113 in
            the bolinas evals pipeline).
        context_size: DNA window in bp passed to ``transform_llr_clm``
            (``255`` for BOS-using checkpoints like exp166-p1B; the tokenizer
            prepends BOS to make 256 tokens).
        batch_size: forward-pass batch size.
        num_workers: dataloader workers.
        data_transform_on_the_fly: when True, sequence extraction happens
            inside the Trainer iteration (no pre-materialised cache).
        torch_compile: whether to ``torch.compile`` the wrapped model.

    Returns:
        DataFrame with columns:

        * ``llr``: log-likelihood ratio ``alt - ref`` (signed)
        * ``minus_llr``, ``abs_llr``: derived scalars (convenience for downstream)
        * ``embed_last_l2``: flattened-sequence Euclidean distance, last hidden
        * ``mean_ref``, ``mean_alt``: D-dim list columns (mean over tokens)
        * ``traitgym_innerprod``: D-dim list column ``(emb_ref ⊙ emb_alt).sum(seq)``
    """
    checkpoint_path = (
        Path(checkpoint_path)
        if Path(str(checkpoint_path)).exists()
        else str(checkpoint_path)
    )
    genome_path = Path(genome_path)

    genome = Genome(genome_path)

    tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(checkpoint_path))
    model = HFCausalLMWithEmbeddings(
        AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
    )

    hf_dataset = Dataset.from_pandas(dataset, preserve_index=False)

    raw = run_inference(
        model,
        tokenizer,
        hf_dataset,
        compute_fn=compute_llr_and_pooled_embeddings,
        data_transform_fn=partial(
            transform_llr_clm, genome=genome, window_size=context_size
        ),
        data_transform_on_the_fly=data_transform_on_the_fly,
        inference_kwargs={
            "per_device_eval_batch_size": batch_size,
            "torch_compile": torch_compile,
            "bf16_full_eval": True,
            "dataloader_num_workers": num_workers,
            "remove_unused_columns": False,
        },
    )

    raw = np.asarray(raw)
    assert raw.ndim == 2 and raw.shape[0] == len(dataset), (
        f"unexpected raw shape {raw.shape} vs n_variants={len(dataset)}"
    )

    total_dense = raw.shape[1] - NUM_SCALAR_FEATURES
    assert total_dense > 0 and total_dense % 3 == 0, (
        f"feature width {raw.shape[1]} is not 2 + 3*D for any integer D"
    )
    d = total_dense // 3

    llr = raw[:, 0].astype(np.float32)
    embed_last_l2 = raw[:, 1].astype(np.float32)
    mean_ref = raw[:, 2 : 2 + d].astype(np.float32)
    mean_alt = raw[:, 2 + d : 2 + 2 * d].astype(np.float32)
    innerprod = raw[:, 2 + 2 * d : 2 + 3 * d].astype(np.float32)

    return pd.DataFrame(
        {
            "llr": llr,
            "minus_llr": -llr,
            "abs_llr": np.abs(llr),
            "embed_last_l2": embed_last_l2,
            "mean_ref": list(mean_ref),
            "mean_alt": list(mean_alt),
            "traitgym_innerprod": list(innerprod),
        }
    )
