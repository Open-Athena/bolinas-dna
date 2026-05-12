🤖 **Closing — neither finding is a big issue in practice**

Final read after iters 1–5:

**N-stretches** — not a real problem.
- 0.155% of bases are N, 99.46% of windows are entirely N-free, 0.4% have ≥10% N. The OP's "125 leading N's" example was unlucky sampling (~7600 of ~1M rhino rows look like it).
- **F1 (post-extraction `n_frac` filter)** is still trivially worth landing — one config knob, drops ≤0.5% of rows, mechanism-agnostic. But not a quality emergency.

**Repeat-masking consistency** — variance exists but is small in practice and mostly intrinsic.
- Overall **9.3% lowercase** across v1 → mean effective per-base training weight **0.907** under the 100× downweight policy.
- Per-species effective weight ranges 0.78–0.98 (1.26× spread). 10 pp GCF/GCA gap is the only piece that's clearly *not* biology — it tracks Dfam 3.3 library coverage by lineage. Even there, the absolute impact is small: 1% of the gradient still gets through on masked bases, and 80–98% of bases per window are never downweighted.
- v1's mask is already what a uniform-RepeatMasker pipeline produces today (iter-5: Zoonomia ran Dfam 3.3 + cactus-preprocess RED). R2 was a non-starter (already done). R5 (upgrade to Dfam 3.9) might trim the gap by a few pp; not worth it unless training-side artifacts appear.

**Recommendation: R1 + optional F1.** Document both findings in the dataset card; no pipeline change required unless downstream evals surface a GCF/GCA training asymmetry. Updating the issue body to reflect this.

This audit was most useful as a record of *why R1 is fine*, not as a forcing function for a fix. The most misleading framing along the way was iter-2's "HAL strip-down" reading, which iter-4 and iter-5 walked back as the actual mechanism came into focus.

Closing the issue. Reopen if downstream eval data shows the variance actually matters.
