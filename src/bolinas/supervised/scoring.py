"""GPU-side scoring helpers used by the supervised-VEP feature-extraction pipeline.

The compute function here is plugged into ``biofoundation.inference.run_inference``
as the ``compute_fn``; it consumes the same ``transform_llr_clm`` data transform
(``input_ids`` of shape ``[B, 2, L]``) as the zero-shot pipeline.

Layout of the returned tensor (columns):

    [:, 0]                  llr (alt_logprob - ref_logprob)
    [:, 1]                  embed_last_l2 (flattened-sequence Euclidean distance, last hidden)
    [:, 2 : 2+D]            mean_ref (mean of last hidden over token positions, ref context)
    [:, 2+D : 2+2*D]        mean_alt (same, alt context)
    [:, 2+2*D : 2+3*D]      traitgym_innerprod = (last_ref ⊙ last_alt).sum(seq_axis), per channel
"""

from einops import rearrange, reduce
import torch
import torch.nn.functional as F
from torch import Tensor

from biofoundation.model.base import CausalLMWithEmbeddings
from biofoundation.model.scoring import _clm_seq_logprob


def compute_llr_and_pooled_embeddings(
    model: CausalLMWithEmbeddings,
    input_ids: Tensor,
) -> Tensor:
    """One forward pass yielding LLR + last-layer L2 + mean-pooled ref/alt + TraitGym innerprod.

    Sequence-level invariants:

    * The pooled / innerprod features use ONLY the last hidden state. Middle-layer
      features are not currently produced; #175 found the last layer
      dominates for the 1B generalist.
    * The TraitGym ``innerprod`` feature is per-channel (D-dim, summed across
      tokens), not per-token. It cannot be reconstructed from ``mean_ref`` and
      ``mean_alt`` alone, so it is computed here.

    Args:
        model: ``CausalLMWithEmbeddings`` returning ``(logits, last_hidden,
            middle_hidden)`` from a single forward pass.
        input_ids: ``[B, 2, L]`` with the V=2 dimension ordered as
            ``[ref_context, alt_context]`` (matches ``transform_llr_clm``).

    Returns:
        ``[B, 2 + 3*D]`` tensor laid out as documented in the module docstring.
        D is the model's hidden-state width and is fixed across the run.
    """
    B = input_ids.shape[0]
    input_ids_flat = rearrange(input_ids, "B V L -> (B V) L")

    logits, last_emb, _middle_emb = model(input_ids_flat)

    log_prob = _clm_seq_logprob(logits, input_ids_flat)
    log_prob = rearrange(log_prob, "(B V) -> B V", B=B)
    llr = log_prob[:, 1] - log_prob[:, 0]  # alt - ref

    last_flat = rearrange(last_emb, "(B V) L D -> B V (L D)", B=B)
    last_distance = F.pairwise_distance(last_flat[:, 0], last_flat[:, 1])

    last_pair = rearrange(last_emb, "(B V) L D -> B V L D", B=B)
    mean_pooled = reduce(last_pair, "B V L D -> B V D", "mean")
    mean_ref = mean_pooled[:, 0]
    mean_alt = mean_pooled[:, 1]

    innerprod = reduce(last_pair[:, 0] * last_pair[:, 1], "B L D -> B D", "sum")

    return torch.cat(
        [
            llr.unsqueeze(-1).float(),
            last_distance.unsqueeze(-1).float(),
            mean_ref.float(),
            mean_alt.float(),
            innerprod.float(),
        ],
        dim=-1,
    )
