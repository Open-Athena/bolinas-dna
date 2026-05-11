🤖 Filed in the course of [Open-Athena/bolinas-dna#175](https://github.com/Open-Athena/bolinas-dna/issues/175) (zero-shot VEP scoring exploration), where I'm running an alternative implementation alongside the existing `compute_llr_clm` and comparing.

## Observation

`biofoundation.model.scoring._logits_to_logprobs` runs `torch.log_softmax` directly on the bf16 logits returned by the model (since `bf16_full_eval=True` is the default in `run_inference`). The per-position log-probs are kept in bf16 through the gather, then cast to fp32 only at the per-sequence sum:

https://github.com/Open-Athena/biofoundation/blob/df5bd4994ca4b2a5bbbc429ec7a592b08e5a22c3/biofoundation/model/scoring.py#L117-L139

```python
def _logits_to_logprobs(logits, input_ids):
    softmax_logprobs = torch.log_softmax(logits, dim=-1)   # bf16 in → bf16 out
    softmax_logprobs = softmax_logprobs[:, :-1]
    input_ids = input_ids[:, 1:]
    logprobs = torch.gather(softmax_logprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
    return logprobs                                        # bf16

def _clm_seq_logprob(logits, input_ids):
    log_probs = _logits_to_logprobs(logits, input_ids)
    return reduce(log_probs.float(), "B L -> B", "sum")    # fp32 only at sum
```

Doing the `log_softmax` in bf16 introduces a per-token relative error of ~10⁻³ that compounds across the T-1 positions of the sequence sum. For the genomic LM checkpoint I tested (`exp136-proj_v30`, V=6, T=256, bf16 forward), this comes out to:

- Median absolute LLR diff vs. an fp32-log_softmax reimplementation: **0.055**
- Worst-case absolute LLR diff: **0.625** (at \|LLR_ref\| ≈ 0.79; relative 80% in the worst variant)
- LLR sign flips: **23 / 1128 variants** (mostly where \|LLR\| < 0.1)
- Matched-pair `|llr_pos| vs |llr_neg|` order flips: **19 / 564 pairs (3.4%)**

Pearson correlation between bf16 and fp32 LLR remains 0.9996 (Spearman 0.998), so the *ranking* is mostly preserved — but in the `evals_v2` matched-pair PairwiseAccuracy use-case the 3.4% pair-flip rate moves per-subset values by 1–2 pairs in the small-N regime, which is the SE noise floor on those subsets.

## Proposed fix

Cast logits to fp32 before `log_softmax`:

```python
def _logits_to_logprobs(logits, input_ids):
    softmax_logprobs = torch.log_softmax(logits.float(), dim=-1)
    softmax_logprobs = softmax_logprobs[:, :-1]
    ...
```

## Tradeoff

The fp32 cast materializes the `(B, T-1, V)` softmax tensor in fp32 instead of bf16. For the small-vocab DNA tokenizers used in the bolinas gLMs (V=6), this is negligible memory: at T=256, B=64, V=6, the cost is ~0.4 MB.

For larger-vocab models (e.g. Qwen3-base V=151K) the cost is ~10 GB at the same B,T — there `F.cross_entropy(logits, targets, reduction='none')` is a drop-in alternative that achieves the fp32 stability without materializing the full softmax tensor.

If a single code path is preferred regardless of vocab, the `F.cross_entropy` form covers both:

```python
def _per_position_logprobs(logits, input_ids):
    pred_logits = logits[:, :-1, :].float().transpose(1, 2)  # (B, V, T-1)
    targets = input_ids[:, 1:]
    nll = F.cross_entropy(pred_logits, targets, reduction='none')
    return -nll
```

(`F.cross_entropy` internally avoids materializing the full log-softmax via a fused kernel.)

## Side effect this fix would have on bolinas

`evals_v2`'s public leaderboards (Open-Athena/bolinas-dna#161 / #162 / #172) would shift slightly — typically by less than 1 pair per subset, but visible at small-N subsets. The `evals_v2` config locks the leaderboard scores to `minus_llr` (mendelian) / `abs_llr` (complex/eqtl) computed via biofoundation, so re-pinning bolinas to a biofoundation version with this fix would propagate the new numerics. Whether that's worth doing is a downstream decision.

## What I'm doing in zeroshot_vep

The new pipeline in [bolinas-dna#175](https://github.com/Open-Athena/bolinas-dna/issues/175) does the fp32 cast inline (it's a separate forward-pass implementation, not calling `compute_llr_clm`), so the numerics there are already the "improved" version. I'll flag the ~5% relative-error discrepancy vs. `evals_v2` in that issue's results table as a caveat rather than a divergence to chase.
