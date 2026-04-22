"""CIGAR-based exact coordinate lift between aligned sequences.

For a PAF alignment of query `[q_start, q_end)` against target `[t_start, t_end)`
with a CIGAR string (from minimap2's `cg:Z:` tag or FASTGA's `-pafx` output),
`cigar_lift` returns the exact target sub-interval corresponding to a given
query sub-interval.

This replaces the linear-proportional lift used in an earlier iteration, which
ignored gaps and could drift by hundreds of bp on gappy alignments — enough to
miss the mm10 cCRE boundary in per-cCRE ortholog evaluation.

PAF CIGAR ops:
- `M`, `=`, `X`: consume both query and target by 1
- `I`: consume query by 1 (insertion relative to target)
- `D`, `N`: consume target by 1 (deletion / skipped region)
- `S`, `H`, `P`: don't consume either for our purposes

For reverse-strand alignments (`strand == "-"`), PAF reports `q_start`/`q_end`
and `t_start`/`t_end` in forward-strand coords of each sequence. The alignment
walks query forward from `q_start` to `q_end` but maps to target REVERSE (from
`t_end` down to `t_start`). We handle this by walking target in the negative
direction from `t_end` for reverse strand.
"""

from __future__ import annotations

import re

_CIGAR_OP_RE = re.compile(r"(\d+)([MIDNSHPX=])")


def cigar_lift(
    q_start: int,
    q_end: int,
    t_start: int,
    t_end: int,
    strand: str,
    cigar: str,
    lift_q_start: int,
    lift_q_end: int,
) -> tuple[int, int] | None:
    """Return exact target coords for a query sub-interval within a PAF alignment.

    Parameters
    ----------
    q_start, q_end : int
        Alignment's query (hg38) interval. 0-based half-open.
    t_start, t_end : int
        Alignment's target (mm10) interval. 0-based half-open.
    strand : {"+", "-"}
        PAF strand column — "+" means query aligns forward to target; "-"
        means query was reverse-complemented before aligning to target.
    cigar : str
        CIGAR string, e.g. "10=2X5=1I3=1D4=". Ops supported: M, =, X, I, D,
        N, S, H, P.
    lift_q_start, lift_q_end : int
        Query sub-interval to lift. Assumed 0-based half-open. Clipped to
        the alignment span if it extends outside.

    Returns
    -------
    (t_start_lifted, t_end_lifted) : tuple[int, int] | None
        Target sub-interval (0-based half-open) corresponding exactly to
        the given query sub-interval, accounting for gaps. Returns `None`
        if the sub-interval falls entirely outside the alignment.
    """
    if lift_q_end <= q_start or lift_q_start >= q_end:
        return None
    # Clip to the alignment's query span so lift_q stays within [q_start, q_end].
    lift_q_start = max(lift_q_start, q_start)
    lift_q_end = min(lift_q_end, q_end)

    q_pos = q_start
    # For strand "-", walk target backward from t_end; final position is t_start.
    t_pos = t_start if strand == "+" else t_end
    t_dir = 1 if strand == "+" else -1

    t_at_lift_start: int | None = None
    t_at_lift_end: int | None = None

    for n_str, op in _CIGAR_OP_RE.findall(cigar):
        n = int(n_str)
        q_advance = op in "M=XI"
        t_advance = op in "M=XDN"
        q_step = n if q_advance else 0
        q_pos_next = q_pos + q_step

        # Does this op straddle the lift start boundary?
        if t_at_lift_start is None and q_pos <= lift_q_start < q_pos_next:
            # How many query bases into this op does lift_q_start sit?
            advance = lift_q_start - q_pos
            # If both advance, target advances 1:1. If only q advances (I op),
            # t stays at current t_pos. If only t advances (D/N), q wouldn't
            # straddle — impossible here since q_pos < q_pos_next.
            t_at_lift_start = t_pos + (t_dir * advance if t_advance else 0)
        # Does this op straddle the lift end boundary?
        if q_pos <= lift_q_end <= q_pos_next:
            advance = lift_q_end - q_pos
            t_at_lift_end = t_pos + (t_dir * advance if t_advance else 0)
            break

        q_pos = q_pos_next
        if t_advance:
            t_pos += t_dir * n

    # Defensive fallbacks — shouldn't normally trigger when lift_q_* is inside [q_start, q_end].
    if t_at_lift_start is None:
        t_at_lift_start = t_pos
    if t_at_lift_end is None:
        t_at_lift_end = t_pos

    # For reverse strand, walking query forward walked target backward, so
    # t_at_lift_end < t_at_lift_start. Swap for standard [start, end) order.
    if strand == "+":
        return t_at_lift_start, t_at_lift_end
    return t_at_lift_end, t_at_lift_start
