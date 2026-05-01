"""Resize a projected interval to a fixed length, centered on its midpoint.

Used post-projection to coerce variable-length orthologous regions back to a
fixed model-context size (255 bp). When centering would fall off either end of
the chromosome the interval is shifted inward to preserve the target length.
"""

from __future__ import annotations


def resize_to_length(
    start: int, end: int, target_len: int, chrom_size: int
) -> tuple[int, int]:
    """Resize ``[start, end)`` to length ``target_len`` around its midpoint.

    Args:
        start: 0-based half-open start coordinate (inclusive).
        end: 0-based half-open end coordinate (exclusive). Must be > start.
        target_len: desired output length (positive).
        chrom_size: chromosome size; output is clamped to ``[0, chrom_size]``.

    Returns:
        ``(new_start, new_end)`` such that ``new_end - new_start == target_len``
        and ``0 <= new_start <= new_end <= chrom_size``.

    Raises:
        ValueError: if ``chrom_size < target_len`` (cannot fit).
        ValueError: if ``end <= start`` or ``target_len <= 0``.
    """
    if target_len <= 0:
        raise ValueError(f"target_len must be positive: {target_len}")
    if end <= start:
        raise ValueError(f"empty interval: start={start} end={end}")
    if chrom_size < target_len:
        raise ValueError(
            f"chrom_size ({chrom_size}) < target_len ({target_len}); cannot fit"
        )

    midpoint = (start + end) // 2
    half = target_len // 2
    new_start = midpoint - half
    new_end = new_start + target_len

    if new_start < 0:
        shift = -new_start
        new_start += shift
        new_end += shift
    elif new_end > chrom_size:
        shift = new_end - chrom_size
        new_start -= shift
        new_end -= shift

    assert new_end - new_start == target_len
    assert 0 <= new_start <= new_end <= chrom_size
    return new_start, new_end
