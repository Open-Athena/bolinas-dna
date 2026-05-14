"""Tests for ``bolinas.model.scoring`` (CLM-only).

Vendored from biofoundation/tests/test_scoring.py at commit 834dd4c (May 2026).
MLM test (``test_run_llr_mlm_rc_avg_equals_mean_of_two_passes``) dropped —
bolinas-dna's vendored scoring module is CLM-only.

The helper test doubles (``_DeterministicCLM``,
``_DeterministicCausalLMWithEmbeddings``) are plain ``nn.Module`` subclasses
here. After the migration, ``bolinas.model.scoring`` calls
``model(input_ids).logits`` (and ``output.hidden_states[i]`` when
``output_hidden_states=True``) directly — no ``CausalLM`` /
``CausalLMWithEmbeddings`` abstract bases. The helpers return
``SimpleNamespace`` objects exposing the same attribute surface as HF
``CausalLMOutput``.
"""

import math
from functools import partial
from types import SimpleNamespace

import datasets
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from bolinas.data.genome import Genome
from bolinas.data.transforms import (
    transform_llr_clm,
    transform_reflogprob_clm,
)
from bolinas.model.runner import (
    run_inference,
    run_ll_clm,
    run_llr_and_embedding_distance,
    run_llr_clm,
)
from bolinas.model.scoring import (
    _logits_to_logprobs,
    compute_llr_and_embedding_distance,
    compute_llr_clm,
    compute_ll_clm,
    compute_reflogprob_clm,
)


TINY_CLM = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"


def _load_tiny_clm():
    return AutoModelForCausalLM.from_pretrained(TINY_CLM)


class _DeterministicCLM(nn.Module):
    """Test double whose forward returns a fixed logits tensor wrapped in an
    HF-style output object (``.logits``)."""

    def __init__(self, logits: Tensor):
        super().__init__()
        # Register as buffer so .to(device) works, but value is fixed.
        self.register_buffer("_logits", logits)

    def forward(self, input_ids):
        # Returns a *copy* sliced to the input batch/length so callers can
        # use any input_ids shape that matches.
        B, L = input_ids.shape
        assert self._logits.shape[0] >= B and self._logits.shape[1] >= L
        return SimpleNamespace(logits=self._logits[:B, :L].clone())


def test_compute_ll_clm_matches_hf_cross_entropy():
    """ll_sum / n  ==  -model(input_ids, labels=input_ids).loss.

    HF's CausalLM models compute loss as mean cross-entropy over the L-1
    shifted targets. Dividing our per-row ll_sum by n recovers the same
    quantity, with the standard sign flip.
    """
    torch.manual_seed(0)
    model = _load_tiny_clm()
    raw = AutoModelForCausalLM.from_pretrained(TINY_CLM)
    raw.eval()
    model.eval()

    vocab_size = raw.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (3, 17))

    with torch.no_grad():
        out = compute_ll_clm(model, input_ids)  # [B, 2]
        assert out.shape == (3, 2)
        ll_mean = out[:, 0] / out[:, 1]
        for i in range(input_ids.shape[0]):
            hf_loss = raw(input_ids[i : i + 1], labels=input_ids[i : i + 1]).loss
            assert math.isclose(
                ll_mean[i].item(), -hf_loss.item(), rel_tol=1e-5, abs_tol=1e-5
            ), f"row {i}: ours={ll_mean[i].item()} hf={-hf_loss.item()}"


def test_compute_ll_clm_hand_computed_two_token():
    """Smallest non-trivial off-by-one check with a known logits tensor."""
    # Vocab size 4, batch 1, length 3
    # logits[0, 0] predicts input_ids[0, 1]
    # logits[0, 1] predicts input_ids[0, 2]
    # logits[0, 2] is unused (last position)
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [9.0, 9.0, 9.0, 9.0],
            ]
        ]
    )
    input_ids = torch.tensor([[3, 1, 0]])
    model = _DeterministicCLM(logits)

    log_softmax_0 = torch.log_softmax(logits[0, 0], dim=-1)
    log_softmax_1 = torch.log_softmax(logits[0, 1], dim=-1)
    expected_sum = (log_softmax_0[1] + log_softmax_1[0]).item()

    out = compute_ll_clm(model, input_ids)
    assert out.shape == (1, 2)
    assert math.isclose(out[0, 0].item(), expected_sum, rel_tol=1e-6, abs_tol=1e-7)
    assert out[0, 1].item() == 2.0


def test_compute_ll_clm_target_side_shift():
    """is_upper applies to the *target* token (input_ids[i+1]), not the source."""
    torch.manual_seed(1)
    B, L, V = 1, 8, 5
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)

    # Source-aligned mask: positions [0..3] uppercase, [4..7] lowercase.
    is_upper = torch.tensor([[True, True, True, True, False, False, False, False]])

    out = compute_ll_clm(model, input_ids, is_upper)  # [B, 4]
    assert out.shape == (1, 4)
    ll_sum_upper, ll_sum_lower, n_upper, n_lower = out[0].tolist()

    # Manual computation
    log_softmax = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = input_ids[0, 1:]
    per_target_logp = log_softmax[torch.arange(L - 1), targets]
    # Target frame: is_upper[1:] = [T, T, T, F, F, F, F]  → 3 upper, 4 lower
    upper_mask = is_upper[0, 1:]
    lower_mask = ~upper_mask

    expected_sum_upper = per_target_logp[upper_mask].sum().item()
    expected_sum_lower = per_target_logp[lower_mask].sum().item()

    assert math.isclose(ll_sum_upper, expected_sum_upper, rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(ll_sum_lower, expected_sum_lower, rel_tol=1e-6, abs_tol=1e-7)
    assert n_upper == 3.0
    assert n_lower == 4.0

    # Sanity: had we mistakenly used the SOURCE frame [:-1] — would give
    # n_upper=4, n_lower=3 (different counts) and a different sum.
    assert n_upper != is_upper[0, :-1].sum().item()


def test_compute_ll_clm_invariants():
    """Per-row invariants of the [B, 4] output."""
    torch.manual_seed(2)
    B, L, V = 4, 11, 7
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)

    is_upper = torch.zeros(B, L, dtype=torch.bool)
    is_upper[0, :3] = True
    is_upper[1, :7] = True
    is_upper[2, :2] = True
    is_upper[3, :9] = True

    out = compute_ll_clm(model, input_ids, is_upper)  # [B, 4]
    ll_sum_upper, ll_sum_lower, n_upper, n_lower = out.unbind(-1)
    out_no_mask = compute_ll_clm(model, input_ids)  # [B, 2]
    ll_sum_total, n_total = out_no_mask.unbind(-1)

    # Sums partition the total
    assert torch.allclose(ll_sum_upper + ll_sum_lower, ll_sum_total, atol=1e-5)
    # Counts partition L-1
    assert torch.equal(n_upper + n_lower, n_total)
    assert torch.all(n_total == float(L - 1))


def test_compute_ll_clm_dataset_wide_token_weighted_mean():
    """The intended aggregation pattern works and beats avg-of-means
    when n_upper / n_lower vary across rows."""
    torch.manual_seed(5)
    B, L, V = 4, 11, 7
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)

    is_upper = torch.zeros(B, L, dtype=torch.bool)
    is_upper[0, :3] = True
    is_upper[1, :7] = True
    is_upper[2, :2] = True
    is_upper[3, :9] = True

    out = compute_ll_clm(
        model, input_ids, is_upper
    ).double()  # cast for fp64 accumulate
    S_u, S_l, n_u, n_l = out.sum(dim=0).unbind(-1)
    LL_all = ((S_u + S_l) / (n_u + n_l)).item()
    LL_upper = (S_u / n_u).item()
    LL_lower = (S_l / n_l).item()

    # Brute-force: gather *every* target logp across the whole batch and
    # split by mask.
    log_softmax = torch.log_softmax(logits[:, :-1], dim=-1)
    targets = input_ids[:, 1:]
    per_target_logp = torch.gather(log_softmax, 2, targets.unsqueeze(-1)).squeeze(-1)
    target_upper = is_upper[:, 1:]
    expected_LL_all = per_target_logp.mean().item()
    expected_LL_upper = per_target_logp[target_upper].mean().item()
    expected_LL_lower = per_target_logp[~target_upper].mean().item()

    assert math.isclose(LL_all, expected_LL_all, rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(LL_upper, expected_LL_upper, rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(LL_lower, expected_LL_lower, rel_tol=1e-6, abs_tol=1e-7)

    # Sanity that this differs from avg-of-per-sequence-means with
    # heterogeneous counts (the wrong way to aggregate).
    per_seq_upper = (out[:, 0] / out[:, 2]).double()
    naive = per_seq_upper.mean().item()
    assert not math.isclose(LL_upper, naive, rel_tol=1e-3, abs_tol=1e-3)


def test_compute_ll_clm_all_upper_or_all_lower_rows_aggregate_correctly():
    """All-upper / all-lower rows have n=0 in one bucket, ll_sum=0 there.
    They still contribute correctly when summing across the dataset (no
    NaN in the per-row tensor, no NaN gymnastics needed at aggregation)."""
    torch.manual_seed(8)
    B, L, V = 3, 6, 4
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)

    is_upper = torch.zeros(B, L, dtype=torch.bool)
    is_upper[0, :] = True  # row 0: all upper (target frame too)
    # row 1: all lower (default)
    is_upper[2, :3] = True  # row 2: mixed

    out = compute_ll_clm(model, input_ids, is_upper)
    assert out.shape == (B, 4)
    # Per-row primitive output is finite everywhere — no NaN to manage.
    assert torch.isfinite(out).all()
    # Row 0: n_lower == 0, ll_sum_lower == 0
    assert out[0, 1].item() == 0.0
    assert out[0, 3].item() == 0.0
    # Row 1: n_upper == 0, ll_sum_upper == 0
    assert out[1, 0].item() == 0.0
    assert out[1, 2].item() == 0.0

    # Aggregating across all 3 rows still gives meaningful global LLs
    S_u, S_l, n_u, n_l = out.double().sum(dim=0).tolist()
    assert n_u > 0 and n_l > 0  # because row 2 contributes to both
    LL_upper = S_u / n_u
    LL_lower = S_l / n_l
    assert math.isfinite(LL_upper) and math.isfinite(LL_lower)


def test_compute_ll_clm_shape_without_mask():
    torch.manual_seed(4)
    B, L, V = 2, 6, 4
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)
    out = compute_ll_clm(model, input_ids)
    assert out.shape == (B, 2)
    assert torch.all(out[:, 1] == float(L - 1))


def test_logits_to_logprobs_promotes_bf16_to_fp32():
    """Regression test for #21: log_softmax must run in fp32 even when
    the model returns bf16 logits, so per-token bf16 rounding error does
    not compound across the sequence sum.

    Starting from the same bf16-rounded logits, the fp32-internal path
    (the fix) must produce a sequence-summed log-prob closer to the full
    fp32 reference than the bf16-internal path (the unfixed code) does.
    """
    torch.manual_seed(0)
    B, L, V = 2, 256, 6  # T=256 from the issue's measurement
    fp32_logits = torch.randn(B, L, V) * 5
    bf16_logits = fp32_logits.to(torch.bfloat16)
    input_ids = torch.randint(0, V, (B, L))

    # The fix: fp32 internal even from bf16 input.
    logp_fixed = _logits_to_logprobs(bf16_logits, input_ids)
    assert logp_fixed.dtype == torch.float32

    # Full fp32 path (best attainable given fp32 logits).
    logp_fp32 = _logits_to_logprobs(fp32_logits, input_ids)

    # Unfixed path: log_softmax in bf16 (inline reproduction).
    softmax_bf16 = torch.log_softmax(bf16_logits, dim=-1)[:, :-1]
    targets = input_ids[:, 1:]
    logp_unfixed = (
        torch.gather(softmax_bf16, 2, targets.unsqueeze(-1)).squeeze(-1).float()
    )

    err_fixed = (logp_fixed.sum(-1) - logp_fp32.sum(-1)).abs().max().item()
    err_unfixed = (logp_unfixed.sum(-1) - logp_fp32.sum(-1)).abs().max().item()
    # The unfixed path compounds bf16 log_softmax error across L-1
    # positions on top of input-rounding error; the fix only carries the
    # latter.
    assert err_fixed < err_unfixed


def test_run_ll_clm_end_to_end():
    """Smoke test: run_ll_clm threads transform_ll_clm + compute_ll_clm
    through the HF Trainer batching pipeline and produces the expected
    [N, 4] shape. Catches any future regression in the wiring of the
    partial / data-transform / model-compute-fn pipeline."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    model = AutoModelForCausalLM.from_pretrained(TINY_CLM)

    seqs = ["ACGTAC", "AcGtAc", "acgtac", "ACGTAC"]
    dataset = datasets.Dataset.from_dict({"seq": seqs})

    pred = run_ll_clm(
        model,
        tokenizer,
        dataset,
        data_transform_on_the_fly=True,
        inference_kwargs=dict(
            per_device_eval_batch_size=2,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
        ),
    )

    assert pred.shape == (len(seqs), 4)
    # Sanity: per-row n_upper + n_lower = L - 1 (= 6 - 1 = 5; tokenizer has no specials)
    assert (pred[:, 2] + pred[:, 3] == 5).all()
    # Row 0 and row 3 are identical sequences — same outputs.
    assert (pred[0] == pred[3]).all()
    # Row 0 (all upper) and row 2 (all lower) hit different is_upper buckets
    # but since the tokenizer is case-insensitive, the *total* sum matches.
    total_0 = pred[0, 0] + pred[0, 1]
    total_2 = pred[2, 0] + pred[2, 1]
    assert math.isclose(total_0, total_2, rel_tol=1e-5, abs_tol=1e-5)


def _write_long_fasta(tmp_path):
    """400-bp FASTA — long enough for windows up to ~200bp without N-padding."""
    fasta = ">chr1\n" + ("ACGT" * 100) + "\n"
    path = tmp_path / "long.fa"
    path.write_text(fasta)
    return path


def _make_variant_dataset():
    """A handful of variants at mid-chrom positions so windows don't N-pad.

    Ref alleles match the underlying FASTA (``"ACGT" * 100``), so for 1-based
    VCF position ``N`` the genome base is ``"ACGT"[(N - 1) % 4]``.
    """
    return datasets.Dataset.from_dict(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "pos": [99, 100, 101, 102],  # G, T, A, C
            "ref": ["G", "T", "A", "C"],
            "alt": ["C", "A", "T", "G"],
        }
    )


_INFERENCE_KWARGS = dict(
    per_device_eval_batch_size=2,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    report_to="none",
)


def test_run_llr_clm_rc_avg_equals_mean_of_two_passes(tmp_path):
    """run_llr_clm(rc_avg=True) returns the element-wise mean of two single-
    strand runs. Catches regressions in the partial / transform / compute_fn
    wiring of the rc_avg path."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    model = AutoModelForCausalLM.from_pretrained(TINY_CLM)
    model.eval()
    genome = Genome(_write_long_fasta(tmp_path))
    dataset = _make_variant_dataset()
    window_size = 16

    fwd = run_llr_clm(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc_avg=False,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    rc = run_inference(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_llr_clm,
        data_transform_fn=partial(
            transform_llr_clm, genome=genome, window_size=window_size, strand="-"
        ),
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    avg = run_llr_clm(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc_avg=True,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )

    np.testing.assert_allclose(
        avg, (np.asarray(fwd) + np.asarray(rc)) / 2, rtol=1e-5, atol=1e-6
    )
    # And FWD != RC for at least one variant (otherwise the test is trivial)
    assert not np.allclose(fwd, rc, atol=1e-6)


def test_run_reflogprob_clm_fwd_and_rc_differ():
    """FWD and RC ``transform_reflogprob_clm`` passes through
    ``run_inference`` produce distinct outputs. ``bolinas.model.runner``
    doesn't export a ``run_reflogprob_clm`` wrapper, so this verifies the
    underlying ``run_inference`` + ``compute_reflogprob_clm`` +
    ``transform_reflogprob_clm`` wiring at least responds to strand —
    rc-averaging itself is covered by ``run_llr_clm`` and
    ``run_llr_and_embedding_distance`` tests above."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    model = AutoModelForCausalLM.from_pretrained(TINY_CLM)
    model.eval()
    seqs = ["ACGTACGTACGT", "TGCATGCATGCA", "AAACCCGGGTTT", "CGATCGATCGAT"]
    pos_list = [5, 4, 6, 7]
    dataset = datasets.Dataset.from_dict({"seq": seqs, "pos": pos_list})

    fwd = run_inference(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_reflogprob_clm,
        data_transform_fn=partial(transform_reflogprob_clm, strand="+"),
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    rc = run_inference(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_reflogprob_clm,
        data_transform_fn=partial(transform_reflogprob_clm, strand="-"),
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    assert not np.allclose(fwd, rc, atol=1e-6)


class _DeterministicCausalLMWithEmbeddings(nn.Module):
    """Test double returning content-dependent (logits, last_emb, middle_emb)
    so that two different input batches produce two different outputs —
    enough to verify that the rc_avg=True path of
    ``run_llr_and_embedding_distance`` correctly averages the FWD and RC
    [N, 3] predictions.

    Mimics HF ``CausalLMOutputWithHiddenStates``: returns an object with
    ``.logits`` and ``.hidden_states`` where ``hidden_states[-1]`` is the
    last-layer embedding and ``hidden_states[len // 2]`` is the middle one.
    """

    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

    def forward(self, input_ids, output_hidden_states: bool = False):
        x = input_ids.float()
        logits = x.unsqueeze(-1).repeat(1, 1, self.vocab_size).contiguous()
        logits = logits + torch.arange(self.vocab_size, dtype=torch.float)
        last = x.unsqueeze(-1).repeat(1, 1, self.emb_dim).contiguous()
        middle = (x.unsqueeze(-1) * 2).repeat(1, 1, self.emb_dim).contiguous()
        # ``compute_llr_and_embedding_distance`` indexes hidden_states[-1]
        # (last) and hidden_states[len // 2] (middle). A 2-tuple makes
        # ``len // 2 == 1``, so hidden_states[1] is ``last`` — but we
        # want index ``len // 2`` to map to ``middle``. Use 4 layers so
        # ``len // 2 == 2`` and ``[2] == middle`` while ``[-1] == last``.
        hidden_states = (last, last, middle, last)
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


def test_run_llr_and_embedding_distance_rc_avg_equals_mean_of_two_passes(tmp_path):
    """End-to-end smoke test for the [N, 3] return shape — the path
    biofoundation issue #24 specifically called out as the primary VEP
    entrypoint."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    model = _DeterministicCausalLMWithEmbeddings(vocab_size=8, emb_dim=4)
    model.eval()
    genome = Genome(_write_long_fasta(tmp_path))
    dataset = _make_variant_dataset()
    window_size = 16

    fwd = run_llr_and_embedding_distance(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc_avg=False,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    rc = run_inference(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_llr_and_embedding_distance,
        data_transform_fn=partial(
            transform_llr_clm, genome=genome, window_size=window_size, strand="-"
        ),
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )
    avg = run_llr_and_embedding_distance(
        model,
        tokenizer,
        dataset,
        genome,
        window_size,
        rc_avg=True,
        data_transform_on_the_fly=True,
        inference_kwargs=_INFERENCE_KWARGS,
    )

    assert fwd.shape == (4, 3)
    assert rc.shape == (4, 3)
    assert avg.shape == (4, 3)
    np.testing.assert_allclose(
        avg, (np.asarray(fwd) + np.asarray(rc)) / 2, rtol=1e-5, atol=1e-6
    )
    assert not np.allclose(fwd, rc, atol=1e-6)
