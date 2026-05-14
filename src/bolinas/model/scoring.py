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


def compute_llr_and_embedding_distance(
    model: Any,
    input_ids: Int[Tensor, "B 2 L"],
) -> Float[Tensor, "B 3"]:
    """Compute LLR, last-layer distance, and middle-layer distance in one pass.

    Args:
        model: HF-shaped causal LM; ``model(input_ids, output_hidden_states=True)``
            must return an object with ``.logits`` and ``.hidden_states``
            (tuple of layer outputs, last layer at ``[-1]``).
        input_ids: Input sequences with shape [B, 2, L] where the 2 sequences are [ref, alt]

    Returns:
        Tensor with shape [B, 3] where columns are:
            - [:, 0]: LLR (log-likelihood ratio: alt_logprob - ref_logprob)
            - [:, 1]: Euclidean distance between last-layer embeddings
            - [:, 2]: Euclidean distance between middle-layer embeddings
    """
    B = input_ids.shape[0]
    input_ids_flat = rearrange(input_ids, "B V L -> (B V) L")

    output = model(input_ids_flat, output_hidden_states=True)
    logits = output.logits
    last_emb = output.hidden_states[-1]
    middle_idx = len(output.hidden_states) // 2
    middle_emb = output.hidden_states[middle_idx]

    log_prob = _clm_seq_logprob(logits, input_ids_flat)
    log_prob = rearrange(log_prob, "(B V) -> B V", B=B)
    llr = log_prob[:, 1] - log_prob[:, 0]  # alt - ref

    last_emb = rearrange(last_emb, "(B V) L D -> B V (L D)", B=B)
    last_distance = F.pairwise_distance(last_emb[:, 0], last_emb[:, 1])

    middle_emb = rearrange(middle_emb, "(B V) L D -> B V (L D)", B=B)
    middle_distance = F.pairwise_distance(middle_emb[:, 0], middle_emb[:, 1])

    return torch.stack([llr, last_distance, middle_distance], dim=1)
