🤖 **Correction on the eqtl-null framing.** I called it a matched-pair construction artifact ("negatives too close to positives"); @gonzalobenegas pushed back — AlphaGenome does find signal on the same eqtl dataset, so the test set isn't the bottleneck. The real explanation is **selection strength**:

| dataset | positive class | selection pressure | sequence-likelihood signal |
|---|---|---|---|
| mendelian | HGMD/OMIM pathogenic | strong purifying | medium-large Cohen's d (0.3–0.95) |
| complex | fine-mapped GWAS (MAF > 0.001) | weak | tiny d (~0.05–0.15) on a handful of subsets |
| eqtl | GTEx fine-mapped expression | weakest / mixed-direction | d ≈ 0 across distal; wrong-sign for splicing |

Sequence-likelihood models (LLR, entropy under a natural-sequence prior) detect *constraint*. Mendelian pathogenic variants are heavily constrained → big LLR. eQTLs mostly aren't constrained at all — many are nearly neutral or even gain-of-function (e.g. variants creating canonical splice motifs). The model can't distinguish them from matched non-eQTLs because the relevant signal isn't *in* the model's training distribution.

AlphaGenome closes the gap because it's trained on functional readouts (chromatin, expression) — it learns regulatory function directly rather than inferring it from conservation. That's outside zero-shot bolinas gLM scope.

**Consistent with this framing in our data**:
- The only scores that work for eqtl globally are `minus_entropy` (macro 0.523) and `abs_llr` (0.516) — both are **magnitude / position-uncertainty** scores, not "is alt selected against?" scores. They detect "this position is unusual" without requiring directional selection on the variant. Right signal type for low-selection variants.
- The wrong-sign Cohen's d for `minus_llr` on splicing eQTLs (-0.624) is exactly what you'd expect when splicing eQTL positives include gain-of-function variants (alt is *more* likely than ref under the model because alt creates a canonical motif).

**Dropping from iter-3 plan**:
- ~~"Eqtl-distal matched-pair re-examination"~~ — the matching isn't the bottleneck; selection is.

**Adding to iter-3 plan**:
- **eqtl-via-magnitude-scores**: re-frame the eqtl comparison around `minus_entropy` + `abs_llr` + similar direction-agnostic scores instead of treating it as a leaderboard-style benchmark for `minus_llr`. Could even build a magnitude-only composite (`rk_minus_entropy_plus_abs_llr_plus_embed_l2_flat_last`) and see if it's a meaningful zero-shot baseline for eqtl-class variants.
