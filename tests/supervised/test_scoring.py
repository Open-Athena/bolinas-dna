"""Unit tests for ``bolinas.supervised.scoring.compute_llr_and_pooled_embeddings``.

The function consumes a ``CausalLMWithEmbeddings`` and ``input_ids`` of shape
``[B, 2, L]``. We mock the model with a tiny deterministic stub so the test is
fast and independent of any real checkpoint.
"""

import pytest
import torch
import torch.nn as nn
from einops import rearrange, reduce

from bolinas.supervised.scoring import compute_llr_and_pooled_embeddings


class _StubCLMWithEmbeddings(nn.Module):
    """Deterministic stub: maps input_ids to fixed logits + hidden states.

    Hidden state ``h[b, l, d] = input_ids[b, l].float() + d * 0.01`` makes the
    embedding patterns easy to reason about while still being non-trivial
    (varies per token id, per position, and per channel). Logits are a learned
    linear projection of the hidden state.
    """

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        # Deterministic linear from hidden -> logits.
        torch.manual_seed(0)
        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        D = self.hidden_dim
        positions = torch.arange(D, dtype=torch.float32, device=input_ids.device) * 0.01
        hidden = input_ids.float().unsqueeze(-1) + positions
        logits = self.proj(hidden)
        return logits, hidden, hidden


@pytest.fixture
def stub_inputs():
    """Small batch of (ref, alt) pairs with deliberately distinct content."""
    B, L = 3, 8
    torch.manual_seed(42)
    # input_ids shape [B, 2, L]. ref and alt differ at the center position.
    ref = torch.randint(low=4, high=20, size=(B, L))
    alt = ref.clone()
    alt[:, L // 2] = (alt[:, L // 2] + 5) % 20 + 4  # mutate the center token
    return torch.stack([ref, alt], dim=1)


def test_output_shape_is_2_plus_3D(stub_inputs):
    D = 6
    model = _StubCLMWithEmbeddings(vocab_size=24, hidden_dim=D)
    out = compute_llr_and_pooled_embeddings(model, stub_inputs)
    B = stub_inputs.shape[0]
    assert out.shape == (B, 2 + 3 * D)
    assert out.dtype == torch.float32


def test_llr_matches_independent_computation(stub_inputs):
    """Reconstruct LLR from the model's logits and confirm the helper agrees."""
    D = 6
    model = _StubCLMWithEmbeddings(vocab_size=24, hidden_dim=D)
    out = compute_llr_and_pooled_embeddings(model, stub_inputs)

    B = stub_inputs.shape[0]
    ids_flat = rearrange(stub_inputs, "B V L -> (B V) L")
    logits, _, _ = model(ids_flat)
    log_probs_full = torch.log_softmax(logits.float(), dim=-1)
    log_probs_targets = torch.gather(
        log_probs_full[:, :-1],
        2,
        ids_flat[:, 1:].unsqueeze(-1),
    ).squeeze(-1)
    seq_logp = log_probs_targets.float().sum(dim=-1)
    seq_logp = rearrange(seq_logp, "(B V) -> B V", B=B)
    expected_llr = seq_logp[:, 1] - seq_logp[:, 0]

    torch.testing.assert_close(out[:, 0], expected_llr, rtol=1e-4, atol=1e-4)


def test_mean_pool_matches_independent_computation(stub_inputs):
    D = 6
    model = _StubCLMWithEmbeddings(vocab_size=24, hidden_dim=D)
    out = compute_llr_and_pooled_embeddings(model, stub_inputs)

    B = stub_inputs.shape[0]
    ids_flat = rearrange(stub_inputs, "B V L -> (B V) L")
    _, hidden, _ = model(ids_flat)
    hidden_pair = rearrange(hidden, "(B V) L D -> B V L D", B=B)
    mean_pooled = reduce(hidden_pair, "B V L D -> B V D", "mean")
    expected_mean_ref = mean_pooled[:, 0]
    expected_mean_alt = mean_pooled[:, 1]

    mean_ref = out[:, 2 : 2 + D]
    mean_alt = out[:, 2 + D : 2 + 2 * D]
    torch.testing.assert_close(mean_ref, expected_mean_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(mean_alt, expected_mean_alt, rtol=1e-5, atol=1e-5)


def test_traitgym_innerprod_is_per_channel_summed_over_tokens(stub_inputs):
    D = 6
    model = _StubCLMWithEmbeddings(vocab_size=24, hidden_dim=D)
    out = compute_llr_and_pooled_embeddings(model, stub_inputs)

    B = stub_inputs.shape[0]
    ids_flat = rearrange(stub_inputs, "B V L -> (B V) L")
    _, hidden, _ = model(ids_flat)
    hidden_pair = rearrange(hidden, "(B V) L D -> B V L D", B=B)
    expected_innerprod = (hidden_pair[:, 0] * hidden_pair[:, 1]).sum(dim=1)

    innerprod = out[:, 2 + 2 * D : 2 + 3 * D]
    torch.testing.assert_close(innerprod, expected_innerprod, rtol=1e-5, atol=1e-5)


def test_traitgym_innerprod_is_symmetric_under_ref_alt_swap(stub_inputs):
    """``(emb_ref ⊙ emb_alt).sum(seq)`` is commutative in (ref, alt)."""
    D = 6
    model = _StubCLMWithEmbeddings(vocab_size=24, hidden_dim=D)

    swapped = stub_inputs.flip(dims=[1])
    out_orig = compute_llr_and_pooled_embeddings(model, stub_inputs)
    out_swap = compute_llr_and_pooled_embeddings(model, swapped)

    inner_orig = out_orig[:, 2 + 2 * D : 2 + 3 * D]
    inner_swap = out_swap[:, 2 + 2 * D : 2 + 3 * D]
    torch.testing.assert_close(inner_orig, inner_swap, rtol=1e-5, atol=1e-5)


def test_mean_pools_swap_under_ref_alt_swap(stub_inputs):
    """mean_ref / mean_alt should exchange under ref↔alt swap (they're asymmetric)."""
    D = 6
    model = _StubCLMWithEmbeddings(vocab_size=24, hidden_dim=D)

    swapped = stub_inputs.flip(dims=[1])
    out_orig = compute_llr_and_pooled_embeddings(model, stub_inputs)
    out_swap = compute_llr_and_pooled_embeddings(model, swapped)

    mean_ref_orig = out_orig[:, 2 : 2 + D]
    mean_alt_orig = out_orig[:, 2 + D : 2 + 2 * D]
    mean_ref_swap = out_swap[:, 2 : 2 + D]
    mean_alt_swap = out_swap[:, 2 + D : 2 + 2 * D]

    torch.testing.assert_close(mean_ref_swap, mean_alt_orig, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(mean_alt_swap, mean_ref_orig, rtol=1e-5, atol=1e-5)


def test_llr_flips_sign_under_ref_alt_swap(stub_inputs):
    """LLR = alt_logp - ref_logp negates when ref/alt swap."""
    D = 6
    model = _StubCLMWithEmbeddings(vocab_size=24, hidden_dim=D)

    swapped = stub_inputs.flip(dims=[1])
    llr_orig = compute_llr_and_pooled_embeddings(model, stub_inputs)[:, 0]
    llr_swap = compute_llr_and_pooled_embeddings(model, swapped)[:, 0]
    torch.testing.assert_close(llr_swap, -llr_orig, rtol=1e-4, atol=1e-4)


def test_embed_last_l2_is_symmetric_under_ref_alt_swap(stub_inputs):
    """Pairwise L2 distance is symmetric in its two arguments."""
    D = 6
    model = _StubCLMWithEmbeddings(vocab_size=24, hidden_dim=D)

    swapped = stub_inputs.flip(dims=[1])
    l2_orig = compute_llr_and_pooled_embeddings(model, stub_inputs)[:, 1]
    l2_swap = compute_llr_and_pooled_embeddings(model, swapped)[:, 1]
    torch.testing.assert_close(l2_swap, l2_orig, rtol=1e-5, atol=1e-5)
