"""CLM scoring kernels.

Vendored from biofoundation/model/scoring.py at commit 834dd4c (May 2026),
CLM-only (MLM compute functions dropped). Rewritten to call HF model
methods directly (``model(input_ids).logits``, ``model(input_ids,
output_hidden_states=True).hidden_states[i]``) — no ``CausalLM`` /
``CausalLMWithEmbeddings`` / ``EmbeddingModel`` abstract base classes.

The ``model`` argument is duck-typed: any callable whose output exposes
``.logits`` (and ``.hidden_states`` when ``output_hidden_states=True`` is
passed) works. HF ``AutoModelForCausalLM`` satisfies this natively;
non-HF models (e.g. Evo2) should be wrapped to expose the same surface —
see ``pipelines/evals/evo2.py`` for an example.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor


# https://github.com/ArcInstitute/evo2/blob/4c3c8522dc99d2dc14b5b5a07cd65f2b67e6f457/evo2/scoring.py#L37
def _logits_to_logprobs(
    logits: Float[Tensor, "B L V"],
    input_ids: Int[Tensor, "B L"],
) -> Float[Tensor, "B L-1"]:
    """Per-token log-likelihoods of the provided sequence at each position.

    Takes logits ``[B, L, V]`` and uses ``input_ids`` to index into the log-
    likelihoods, returning ``[B, L-1]``.

    Logits are cast to fp32 before log_softmax: bf16 log_softmax has
    ~10^-3 per-token rounding error that compounds across the sequence
    sum in ``_clm_seq_logprob`` (see biofoundation issue #21).
    """
    softmax_logprobs = torch.log_softmax(logits.float(), dim=-1)
    softmax_logprobs = softmax_logprobs[:, :-1]
    input_ids = input_ids[:, 1:]
    assert softmax_logprobs.shape[1] == input_ids.shape[1]

    logprobs = torch.gather(
        softmax_logprobs,  # Gather likelihoods...
        2,  # along the vocab dimension...
        input_ids.unsqueeze(-1),  # using the token ids to index.
    ).squeeze(-1)

    return logprobs


def _clm_seq_logprob(
    logits: Float[Tensor, "B L V"],
    input_ids: Int[Tensor, "B L"],
) -> Float[Tensor, " B"]:
    log_probs = _logits_to_logprobs(logits, input_ids)
    return reduce(log_probs.float(), "B L -> B", "sum")


def compute_llr_clm(
    model: Any,
    input_ids: Int[Tensor, "B 2 L"],
) -> Float[Tensor, " B"]:
    """Compute log-likelihood ratio for causal language models.

    Args:
        model: HF-shaped causal LM; ``model(input_ids).logits`` must
            return logits of shape ``[B, L, V]``.
        input_ids: Input sequences with shape [B, 2, L] where the 2 sequences are [ref, alt]

    Returns:
        Log-likelihood ratio (alt_logprob - ref_logprob) with shape [B]
    """
    B = input_ids.shape[0]
    input_ids = rearrange(input_ids, "B V L -> (B V) L")

    logits = model(input_ids).logits
    log_prob = _clm_seq_logprob(logits, input_ids)
    log_prob = rearrange(log_prob, "(B V) -> B V", B=B)

    llr = log_prob[:, 1] - log_prob[:, 0]  # alt - ref
    return llr


def compute_reflogprob_clm(
    model: Any,
    input_ids: Int[Tensor, "B 4 L"],
    ref: Int[Tensor, " B"],
) -> Float[Tensor, " B"]:
    B = input_ids.shape[0]
    batch_indices = torch.arange(B)
    input_ids = rearrange(input_ids, "B V L -> (B V) L")
    logits = model(input_ids).logits
    log_prob = _clm_seq_logprob(logits, input_ids)
    log_prob = rearrange(log_prob, "(B V) -> B V", B=B)
    # marginal log-probability of each of the 4 alleles
    marginal_log_prob = torch.log_softmax(log_prob, dim=-1)
    ref_log_prob = marginal_log_prob[batch_indices, ref]
    return ref_log_prob


def compute_ll_clm(
    model: Any,
    input_ids: Int[Tensor, "B L"],
    is_upper: Bool[Tensor, "B L"] | None = None,
) -> Float[Tensor, "B 2"] | Float[Tensor, "B 4"]:
    """Per-sequence log-likelihood sums and target counts under a CLM.

    Returns sums and counts (not means) so callers can aggregate to a
    dataset-wide token-weighted mean LL by summing across rows then
    dividing — correct even for sequences that are all-upper or
    all-lower in their case mask.

    ``_logits_to_logprobs`` returns ``[B, L-1]`` where entry ``[b, i]``
    is ``log p(input_ids[b, i+1] | input_ids[b, :i+1])``, so when an
    ``is_upper`` mask is supplied, the relevant case is the case of the
    *target* ``input_ids[i+1]`` — we slice ``is_upper[:, 1:]`` to align
    with the L-1 log-probs.

    Output:

    - Without ``is_upper``: ``[B, 2]`` of ``(ll_sum, n)`` per row.
      ``n = L - 1`` for every row.
    - With ``is_upper``: ``[B, 4]`` of
      ``(ll_sum_upper, ll_sum_lower, n_upper, n_lower)`` per row.
      Invariants: ``ll_sum_upper + ll_sum_lower`` is the total sum,
      ``n_upper + n_lower = L - 1``. Special-token target positions
      (``is_upper = False``) fall into the "lower" bucket.

    Per-row sums are fp32; aggregating across many rows can exceed fp32
    precision (~0.5 absolute error at totals of ~10^6, reachable on a
    16k-row eval set), so cast to fp64 before the cross-row sum.
    """
    logits = model(input_ids).logits
    logp = _logits_to_logprobs(logits, input_ids).float()  # [B, L-1]
    L_minus_1 = logp.shape[-1]
    ll_sum_total = logp.sum(dim=-1)
    if is_upper is None:
        n = torch.full_like(ll_sum_total, float(L_minus_1))
        return torch.stack([ll_sum_total, n], dim=-1)
    upper_t = is_upper[:, 1:].float()
    n_upper = upper_t.sum(dim=-1)
    ll_sum_upper = (logp * upper_t).sum(dim=-1)
    ll_sum_lower = ll_sum_total - ll_sum_upper
    n_lower = float(L_minus_1) - n_upper
    return torch.stack([ll_sum_upper, ll_sum_lower, n_upper, n_lower], dim=-1)


def compute_euclidean_distance(
    model: Any,
    input_ids: Int[Tensor, "B 2 L"],
) -> Float[Tensor, " B"]:
    """Compute Euclidean distance between reference and alternate embeddings.

    ``model`` must be a callable that, given ``input_ids`` ``[B*2, L]``,
    returns an embeddings tensor of shape ``[B*2, L, D]``. For HF, this
    means passing the base ``AutoModel`` (not the causal-LM head).

    Returns euclidean distance of shape ``[B]``.
    """
    B = input_ids.shape[0]
    input_ids = rearrange(input_ids, "B V L -> (B V) L")
    embeddings = model(input_ids)
    embeddings = rearrange(embeddings, "(B V) L D -> B V (L D)", B=B)
    ref_emb = embeddings[:, 0, :]
    alt_emb = embeddings[:, 1, :]
    return F.pairwise_distance(ref_emb, alt_emb)


def compute_variant_score_bundle(
    model: Any,
    input_ids: Int[Tensor, "B 2 L"],
    nuc_token_ids: Int[Tensor, " 4"],
) -> Float[Tensor, "B 2"]:
    """Compute LLR and per-position next-token JSD using prefix-sharing.

    For each (ref, alt) pair the input_ids are identical at every position
    *before* the variant and differ only at ``var_pos`` onward (SNV
    invariant). The shared prefix is forwarded once with ``use_cache=True``;
    the divergent suffixes (one for ref, one for alt) are forwarded with
    the cached prefix as past key/values. Roughly halves the prefix's
    forward-pass FLOPs (≈25% wall-clock saving on a 1B Qwen3 with the
    variant at the window center), same idea lm-eval-harness and inference
    servers like vLLM use.

    Token-level ``var_pos`` is inferred from the input_ids diff, which
    automatically accounts for any BOS the tokenizer prepends — and any
    EOS appended at the end is irrelevant since the variant lands well
    before the suffix end. Even and odd ``window_size`` are both fine
    (``transform_llr_clm`` controls the in-sequence pos; the token-level
    pos is just that plus ``n_prefix``).

    Within a single inference batch ``var_pos`` is constant by
    construction (every variant lands at ``window_size // 2`` for FWD or
    ``window_size - 1 - window_size // 2`` for RC, plus ``n_prefix``;
    ``window_size`` is a per-checkpoint constant). We assert this so a
    batched dataset that violated the assumption would fail loud rather
    than silently produce stitched-together logits from mismatched splits.

    The JSD column (``next_token_jsd_mean``, called ``down_jsd_mean`` in
    Open-Athena/bolinas-dna#175) is the per-position 4-nucleotide softmax
    Jensen-Shannon divergence between REF-context and ALT-context next-token
    predictions, averaged over the AR positions where the variant is in
    context (i.e. ``var_pos <= t <= L-2``). Computed from the stitched
    logits — same numerical path as the un-optimized kernel.

    Embedding-distance columns are dropped (#175 conclusion 9: JSD has
    Spearman ρ ≈ 0.90 with last-layer L2 within mendelian subsets, so the
    signal is largely preserved without the activation-memory cost).

    Args:
        model: HF-shaped causal LM. ``model(input_ids, use_cache=True)``
            must return ``.logits`` of shape ``[B, L, V]`` and
            ``.past_key_values`` (a HF ``Cache`` or legacy
            tuple-of-(K, V)-pairs); ``model(input_ids, past_key_values=...)``
            must accept the cache and produce position-correct logits.
        input_ids: Input sequences with shape [B, 2, L] where the 2 sequences are [ref, alt].
        nuc_token_ids: Length-4 tensor of token IDs for the 4 DNA nucleotides
            in canonical ``NUCLEOTIDES`` (= "ACGT") order. Used to slice the
            full-vocab logits down to a 4-class softmax for the JSD column.

    Returns:
        Tensor with shape [B, 2] where columns are:
            - [:, 0]: LLR (log-likelihood ratio: alt_logprob - ref_logprob)
            - [:, 1]: next_token_jsd_mean (mean per-position 4-nuc JSD over downstream positions)
    """
    B, _, L = input_ids.shape

    # Detect var_pos at the token level; this is BOS-aware automatically.
    diff = input_ids[:, 0] != input_ids[:, 1]  # [B, L]
    assert diff.any(dim=-1).all(), "found rows with identical ref/alt input_ids"
    var_pos_each = diff.float().argmax(dim=-1)  # [B]
    p = int(var_pos_each[0].item())
    assert (var_pos_each == p).all(), (
        f"prefix-sharing requires constant var_pos within the batch; got "
        f"unique values {var_pos_each.unique().tolist()}"
    )
    assert 0 < p < L - 1, (
        f"variant at token position {p} of length-{L} sequence has no shared "
        f"prefix or no downstream prediction; expected 0 < var_pos < L-1"
    )

    # Split shared prefix [B, p] from divergent suffixes [B*2, L-p].
    prefix = input_ids[:, 0, :p].contiguous()
    suffixes_flat = rearrange(input_ids[:, :, p:], "B V L -> (B V) L").contiguous()

    # 1. Forward the shared prefix once and capture its KV cache.
    prefix_out = model(prefix, use_cache=True)
    prefix_logits = prefix_out.logits  # [B, p, V]
    past_kv = prefix_out.past_key_values

    # 2. Duplicate the cache so each (ref, alt) pair sees its own copy.
    past_kv_doubled = _repeat_interleave_kv_cache(past_kv, 2)

    # 3. Forward the divergent suffixes with the cached prefix as context.
    #    HF derives position_ids for the suffix from cache length, so the
    #    suffix tokens land at absolute positions [p, L-1] correctly.
    suffix_logits = model(
        suffixes_flat, past_key_values=past_kv_doubled, use_cache=False
    ).logits  # [B*2, L-p, V]

    # 4. Stitch into [B*2, L, V] for the LLR + JSD downstream math.
    prefix_logits_doubled = prefix_logits.repeat_interleave(2, dim=0)
    logits = torch.cat([prefix_logits_doubled, suffix_logits], dim=1)

    input_ids_flat = rearrange(input_ids, "B V L -> (B V) L")

    log_prob = _clm_seq_logprob(logits, input_ids_flat)
    log_prob = rearrange(log_prob, "(B V) -> B V", B=B)
    llr = log_prob[:, 1] - log_prob[:, 0]  # alt - ref

    next_token_jsd_mean = _compute_next_token_jsd_mean(
        logits, input_ids, nuc_token_ids
    )

    return torch.stack([llr, next_token_jsd_mean], dim=1)


def _repeat_interleave_kv_cache(past_kv: Any, n: int) -> Any:
    """Repeat each layer's K and V along the batch dim by ``n``.

    Always returns an HF ``DynamicCache`` (constructing one from a legacy
    tuple if needed). Modern Qwen3/Llama-style models call
    ``past_key_values.get_seq_length()`` internally — the legacy
    tuple-of-(K, V)-pairs format normally auto-converts, but under
    ``torch.compile`` the conversion can be skipped and the method call
    raises ``AttributeError: 'tuple' object has no attribute
    'get_seq_length'``. Returning a real ``DynamicCache`` sidesteps that.

    Mutates an input ``Cache`` in place — caller doesn't reuse the original.
    """
    from transformers.cache_utils import DynamicCache

    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        for i in range(len(past_kv.key_cache)):
            past_kv.key_cache[i] = past_kv.key_cache[i].repeat_interleave(n, dim=0)
            past_kv.value_cache[i] = past_kv.value_cache[i].repeat_interleave(n, dim=0)
        return past_kv

    # Legacy tuple format → coerce to DynamicCache.
    new_cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(past_kv):
        new_cache.update(
            k.repeat_interleave(n, dim=0),
            v.repeat_interleave(n, dim=0),
            layer_idx=layer_idx,
        )
    return new_cache


def _compute_next_token_jsd_mean(
    logits: Float[Tensor, "Bx2 L V"],
    input_ids: Int[Tensor, "B 2 L"],
    nuc_token_ids: Int[Tensor, " 4"],
) -> Float[Tensor, " B"]:
    """Mean per-position 4-nuc next-token JSD over downstream positions.

    For each batch row, compute the Jensen-Shannon divergence between the
    REF and ALT next-token distributions (restricted to A/C/G/T) at every
    position whose left context includes the variant, then average.
    """
    B, _, L = input_ids.shape

    # SNV invariants: exactly one differing position per row, not at the last
    # position (since logits[L-1] predicts off the end and is dropped).
    diff = input_ids[:, 0] != input_ids[:, 1]  # [B, L]
    assert diff.any(dim=-1).all(), "found rows with identical ref/alt input_ids"
    var_pos = diff.float().argmax(dim=-1)  # [B]
    assert (var_pos < L - 1).all(), (
        f"variant at last position has no downstream prediction; var_pos={var_pos.tolist()}"
    )

    # Restrict to 4-nuc softmax + cast to fp32 (bf16 log_softmax has the
    # same rounding-error issue flagged in _logits_to_logprobs).
    nuc_logits = logits[..., nuc_token_ids.to(logits.device)].float()
    log_p = F.log_softmax(nuc_logits, dim=-1)  # [B*2, L, 4]
    log_p = rearrange(log_p, "(B V) L C -> B V L C", B=B)
    log_p_ref = log_p[:, 0]  # [B, L, 4]
    log_p_alt = log_p[:, 1]  # [B, L, 4]

    # log_M = log(0.5 * (P + Q)) computed exactly via logaddexp (no clamp).
    log_m = torch.logaddexp(log_p_ref, log_p_alt) - math.log(2.0)
    p_ref = log_p_ref.exp()
    p_alt = log_p_alt.exp()
    kl_ref_m = (p_ref * (log_p_ref - log_m)).sum(dim=-1)  # [B, L]
    kl_alt_m = (p_alt * (log_p_alt - log_m)).sum(dim=-1)  # [B, L]
    jsd_per_pos = 0.5 * (kl_ref_m + kl_alt_m)  # [B, L]

    # Mask: t in [var_pos, L-2]. logits[t] predicts input_ids[t+1] from
    # context input_ids[:t+1]; at t=var_pos the variant has just entered
    # context (first downstream-affected prediction). Drop t=L-1 to match
    # the _logits_to_logprobs convention (predicts off the end).
    pos = torch.arange(L, device=jsd_per_pos.device)
    mask = (pos.unsqueeze(0) >= var_pos.unsqueeze(1)) & (pos.unsqueeze(0) <= L - 2)
    n_pos = mask.sum(dim=-1).float()  # > 0 by the assertion above
    return (jsd_per_pos * mask).sum(dim=-1) / n_pos
