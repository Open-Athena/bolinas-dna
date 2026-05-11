"""4-pass forward-inference feature extractor for zero-shot VEP scoring.

For each variant we build **4 candidate sequences** (one per A/C/G/T at the
variant center) and run a single forward pass with ``output_hidden_states=True``.
From these we cache:

- joint sequence log-probability for each of the 4 candidates (drives all 6
  likelihood scores via softmax-normalization over the 4)
- per-position log-probability for each of the 4 candidates (optional; cheap
  and supports future per-position effect scoring)
- per-position hidden states (last + middle layer) from the REF and ALT
  candidate passes only (the other 2 passes' embeddings are discarded — we
  only ever compare REF vs ALT)

The forward-pass + window-construction logic is **inlined** from biofoundation
(``transform_llr_clm`` in ``biofoundation.data``, ``_logits_to_logprobs`` /
``compute_llr_and_embedding_distance`` in ``biofoundation.model.scoring``,
``HFCausalLMWithEmbeddings.forward`` in ``biofoundation.model.adapters.hf``)
so we can change layer indices, output extras, or memory tradeoffs without
plumbing through biofoundation's Trainer wrapper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from biofoundation.data import Genome
from transformers import AutoModelForCausalLM, AutoTokenizer

from bolinas.zeroshot_vep.scores import NUC_TO_IDX, NUCLEOTIDES


# ---------------------------------------------------------------------------
# Window construction (inlined from biofoundation.data._get_variant_window).
# ---------------------------------------------------------------------------


def _get_centered_window(
    genome: Genome, chrom: str, pos: int, ref: str, window_size: int
) -> tuple[str, int]:
    """Extract a window centered on a variant.

    The window splits as ``left | REF | right`` with ``left = window_size // 2``.
    For even ``window_size`` the left flank gets the extra base (e.g.
    128 + 1 + 127 = 256). For odd it's symmetric (e.g. 127 + 1 + 127 = 255).

    Returns ``(sequence, var_idx)`` where ``var_idx`` is the 0-based offset of
    the variant base within the window (= ``window_size // 2``).
    """
    center_index = pos - 1  # 1-based VCF coord → 0-based python
    var_idx = window_size // 2
    start = center_index - var_idx
    end = start + window_size
    seq = genome(chrom, start, end).upper()
    assert len(seq) == window_size
    assert seq[var_idx] == ref, (
        f"genome ref {seq[var_idx]!r} != variant ref {ref!r} at {chrom}:{pos} "
        f"(window_size={window_size}). Possible build mismatch or wrong chromosome naming."
    )
    return seq, var_idx


def _build_candidate_sequences(seq: str, var_idx: int) -> list[str]:
    """Return 4 sequences with A/C/G/T at ``var_idx``, in :data:`NUCLEOTIDES` order."""
    return [seq[:var_idx] + nuc + seq[var_idx + 1 :] for nuc in NUCLEOTIDES]


# ---------------------------------------------------------------------------
# Tokenizer probing (inlined from biofoundation.data._get_special_token_counts
# / _get_nucleotide_token_ids).
# ---------------------------------------------------------------------------


def _get_special_token_counts(tokenizer) -> tuple[int, int]:
    """``(n_prefix, n_suffix)`` — auto-prepended / appended special tokens.

    Probes the tokenizer behaviourally by encoding a single "A": some
    tokenizers declare ``bos_token_id`` but don't auto-insert it.
    """
    try:
        bos_id = tokenizer.bos_token_id
    except AttributeError:
        bos_id = None
    try:
        eos_id = tokenizer.eos_token_id
    except AttributeError:
        eos_id = None
    encoded = tokenizer.encode("A")
    n_prefix = 1 if bos_id is not None and encoded[:1] == [bos_id] else 0
    n_suffix = 1 if eos_id is not None and encoded[-1:] == [eos_id] else 0
    return n_prefix, n_suffix


def _get_nucleotide_token_ids(tokenizer) -> dict[str, int]:
    n_prefix, _ = _get_special_token_counts(tokenizer)
    return {nuc: tokenizer.encode(nuc)[n_prefix] for nuc in NUCLEOTIDES}


# ---------------------------------------------------------------------------
# Per-position log-prob (inlined from biofoundation.model.scoring._logits_to_logprobs)
# but memory-efficient: uses cross_entropy to avoid materializing the full
# log_softmax tensor (which can be 10+ GB for Qwen3-class vocabs at T=512).
# ---------------------------------------------------------------------------


def _per_position_logprobs(
    logits: torch.Tensor,  # (B, T, V)
    input_ids: torch.Tensor,  # (B, T)
) -> torch.Tensor:
    """Per-position log-prob of the input token at the next index.

    Returns shape ``(B, T-1)`` where entry ``[b, i]`` is
    ``log p(input_ids[b, i+1] | input_ids[b, :i+1])``.

    Uses ``F.cross_entropy(reduction='none')`` so we never materialize the full
    ``log_softmax(logits)`` tensor — important at large vocab × long sequence.
    """
    # (B, T-1, V) view → permute to (B, V, T-1) for cross_entropy's expected layout.
    pred_logits = logits[:, :-1, :].transpose(1, 2)  # (B, V, T-1)
    targets = input_ids[:, 1:]  # (B, T-1)
    nll = F.cross_entropy(pred_logits, targets, reduction="none")  # (B, T-1)
    return -nll


# ---------------------------------------------------------------------------
# Feature extraction driver.
# ---------------------------------------------------------------------------


def extract_features(
    checkpoint_path: str | Path,
    dataset: pd.DataFrame,
    genome_path: str | Path,
    window_size: int,
    batch_size: int = 16,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    store_pos_logprob: bool = True,
) -> dict[str, np.ndarray]:
    """Run 4-pass inference over a dataset of variants and return a cache dict.

    Each variant contributes 4 forward-pass rows; per-batch input tensor shape
    is ``(4 * batch_size, T)`` where ``T = window_size + n_prefix``.

    Args:
        checkpoint_path: HuggingFace checkpoint directory (must contain config,
            tokenizer, model weights).
        dataset: DataFrame with at least ``chrom``, ``pos``, ``ref``, ``alt``
            columns. Row order is preserved in the cache.
        genome_path: FASTA reference (loaded via :class:`biofoundation.data.Genome`).
        window_size: DNA window length around the variant. Token length is
            ``window_size + n_prefix`` where ``n_prefix`` is 1 for BOS tokenizers.
        batch_size: Number of *variants* per batch (sequence batch will be 4×).
        device: ``"cuda"`` / ``"cpu"``. Defaults to cuda if available.
        dtype: Model compute dtype. ``torch.bfloat16`` matches the evals_v2
            convention for A10G.
        store_pos_logprob: Also save per-position log-probs to the cache.
            Cheap (~MB) and useful for future per-position scoring experiments.

    Returns:
        Cache dict with keys (all numpy arrays):
            ``seq_logprob`` (N, 4) fp32 — joint seq log-prob per candidate
            ``ref_idx``, ``alt_idx`` (N,) int8 — index in {0,1,2,3}
            ``emb_ref_last``, ``emb_ref_middle`` (N, T, D) fp16
            ``emb_alt_last``, ``emb_alt_middle`` (N, T, D) fp16
            ``var_pos`` (scalar int) — token-index of variant center
            ``window_size``, ``n_prefix``, ``n_suffix`` (scalar int)
            ``row_idx`` (N,) int32 — index into source dataset
            ``pos_logprob`` (N, 4, T-1) fp32 — only if store_pos_logprob
    """
    checkpoint_path = Path(checkpoint_path)
    genome_path = Path(genome_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()

    genome = Genome(genome_path)

    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    assert n_suffix == 0, (
        f"unexpected suffix tokens (n_suffix={n_suffix}); only prefix-BOS tokenizers "
        f"are supported here. Adapt the var_pos calculation if you hit this."
    )

    # Probe T and D using the first variant.
    probe_row = dataset.iloc[0]
    probe_seq, dna_var_idx = _get_centered_window(
        genome, probe_row["chrom"], int(probe_row["pos"]),
        probe_row["ref"], window_size,
    )
    probe_ids = torch.tensor(tokenizer.encode(probe_seq), device=device)
    T_tok = probe_ids.shape[0]
    tok_var_pos = dna_var_idx + n_prefix
    assert T_tok == window_size + n_prefix, (
        f"unexpected token length {T_tok} for window_size={window_size}, n_prefix={n_prefix}"
    )

    # max_position_embeddings is the *training* context. With rope_scaling
    # (e.g. llama3 with factor=8.0 on exp55-mammals), RoPE can extrapolate
    # to ``factor * original_max_position_embeddings`` with degradation. We
    # do NOT hard-fail here — the user may want to compare scores under
    # extrapolation explicitly. Warn loudly so the regime is visible in logs.
    max_pos = getattr(model.config, "max_position_embeddings", None)
    rope_scaling = getattr(model.config, "rope_scaling", None) or {}
    rope_factor = float(rope_scaling.get("factor", 1.0)) if rope_scaling else 1.0
    rope_orig = int(rope_scaling.get("original_max_position_embeddings", max_pos or 0)) if rope_scaling else (max_pos or 0)
    effective_max = max(max_pos or 0, int(rope_factor * (rope_orig or 0)))
    if max_pos is not None and T_tok > max_pos:
        print(
            f"[features] WARN: T_tok={T_tok} exceeds model.config.max_position_embeddings={max_pos}; "
            f"rope_scaling={rope_scaling} → effective_max≈{effective_max}. "
            f"Running in extrapolation regime.",
            flush=True,
        )
        assert effective_max == 0 or T_tok <= effective_max, (
            f"token length {T_tok} exceeds rope-scaling effective_max={effective_max}; "
            f"model will not produce meaningful output"
        )

    with torch.no_grad():
        probe_out = model(probe_ids.unsqueeze(0), output_hidden_states=True)
        n_hidden_states = len(probe_out.hidden_states)
        D = probe_out.hidden_states[-1].shape[-1]
        middle_idx = n_hidden_states // 2  # matches biofoundation
    del probe_out

    print(
        f"[features] checkpoint={checkpoint_path.name} window={window_size} "
        f"T_tok={T_tok} D={D} n_hidden_states={n_hidden_states} "
        f"middle_idx={middle_idx} var_pos_tok={tok_var_pos} dtype={dtype}",
        flush=True,
    )

    N = len(dataset)
    cache: dict[str, np.ndarray] = {
        "seq_logprob": np.empty((N, 4), dtype=np.float32),
        "ref_idx": np.empty(N, dtype=np.int8),
        "alt_idx": np.empty(N, dtype=np.int8),
        "emb_ref_last": np.empty((N, T_tok, D), dtype=np.float16),
        "emb_ref_middle": np.empty((N, T_tok, D), dtype=np.float16),
        "emb_alt_last": np.empty((N, T_tok, D), dtype=np.float16),
        "emb_alt_middle": np.empty((N, T_tok, D), dtype=np.float16),
        "row_idx": np.empty(N, dtype=np.int32),
    }
    if store_pos_logprob:
        cache["pos_logprob"] = np.empty((N, 4, T_tok - 1), dtype=np.float32)

    b_arange = torch.arange(batch_size, device=device)  # capped below per-batch

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_rows = dataset.iloc[batch_start:batch_end]
        B = len(batch_rows)

        batch_input_ids: list[torch.Tensor] = []
        batch_ref_idx: list[int] = []
        batch_alt_idx: list[int] = []
        for _, row in batch_rows.iterrows():
            seq, var_idx = _get_centered_window(
                genome, row["chrom"], int(row["pos"]), row["ref"], window_size
            )
            sequences = _build_candidate_sequences(seq, var_idx)
            ids = torch.stack([
                torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in sequences
            ])  # (4, T)
            batch_input_ids.append(ids)
            batch_ref_idx.append(NUC_TO_IDX[row["ref"]])
            batch_alt_idx.append(NUC_TO_IDX[row["alt"]])

        input_ids = torch.stack(batch_input_ids).to(device)  # (B, 4, T)
        input_ids_flat = input_ids.reshape(B * 4, T_tok)
        ref_idx_np = np.asarray(batch_ref_idx, dtype=np.int8)
        alt_idx_np = np.asarray(batch_alt_idx, dtype=np.int8)

        with torch.no_grad():
            out = model(input_ids_flat, output_hidden_states=True)

        # Compute per-position log-probs in fp32 for numerical stability,
        # then reshape back to (B, 4, T-1) and sum → (B, 4) joint log-probs.
        logits = out.logits  # (B*4, T, V) in compute dtype
        pos_logp = _per_position_logprobs(logits.float(), input_ids_flat)  # (B*4, T-1)
        pos_logp_b = pos_logp.reshape(B, 4, T_tok - 1)
        seq_logp_b = pos_logp_b.sum(dim=-1)
        del logits  # free vocab-sized memory ASAP

        last_hidden_b = out.hidden_states[-1].reshape(B, 4, T_tok, D)
        middle_hidden_b = out.hidden_states[middle_idx].reshape(B, 4, T_tok, D)
        del out

        cache["seq_logprob"][batch_start:batch_end] = seq_logp_b.cpu().numpy()
        if store_pos_logprob:
            cache["pos_logprob"][batch_start:batch_end] = pos_logp_b.cpu().numpy()
        cache["ref_idx"][batch_start:batch_end] = ref_idx_np
        cache["alt_idx"][batch_start:batch_end] = alt_idx_np

        ref_idx_dev = torch.as_tensor(ref_idx_np, device=device, dtype=torch.long)
        alt_idx_dev = torch.as_tensor(alt_idx_np, device=device, dtype=torch.long)
        b_dev = b_arange[:B]
        emb_ref_last = last_hidden_b[b_dev, ref_idx_dev]      # (B, T, D)
        emb_alt_last = last_hidden_b[b_dev, alt_idx_dev]
        emb_ref_mid = middle_hidden_b[b_dev, ref_idx_dev]
        emb_alt_mid = middle_hidden_b[b_dev, alt_idx_dev]
        cache["emb_ref_last"][batch_start:batch_end]   = emb_ref_last.to(torch.float16).cpu().numpy()
        cache["emb_alt_last"][batch_start:batch_end]   = emb_alt_last.to(torch.float16).cpu().numpy()
        cache["emb_ref_middle"][batch_start:batch_end] = emb_ref_mid.to(torch.float16).cpu().numpy()
        cache["emb_alt_middle"][batch_start:batch_end] = emb_alt_mid.to(torch.float16).cpu().numpy()
        cache["row_idx"][batch_start:batch_end] = np.arange(batch_start, batch_end, dtype=np.int32)

        if batch_start % (batch_size * 50) == 0:
            print(f"[features]   {batch_end}/{N} variants", flush=True)

    cache["var_pos"] = np.asarray(int(tok_var_pos), dtype=np.int32)
    cache["n_prefix"] = np.asarray(int(n_prefix), dtype=np.int32)
    cache["n_suffix"] = np.asarray(int(n_suffix), dtype=np.int32)
    cache["window_size"] = np.asarray(int(window_size), dtype=np.int32)

    assert not np.isnan(cache["seq_logprob"]).any(), "NaN in seq_logprob — investigate"
    return cache


def write_cache(path: str | Path, cache: dict[str, np.ndarray]) -> None:
    """Save cache to ``path`` via ``np.savez_compressed``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cache)


def read_cache(path: str | Path) -> dict[str, np.ndarray]:
    """Load all arrays from an npz cache into a dict (materializes into RAM)."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
