"""Tests for ``bolinas.projection.resize``."""

from __future__ import annotations

import pytest

from bolinas.projection.resize import resize_to_length


def test_already_target_length_no_change() -> None:
    s, e = resize_to_length(1000, 1255, target_len=255, chrom_size=100000)
    assert (s, e) == (1000, 1255)


def test_longer_input_trimmed_around_midpoint() -> None:
    # input 400 bp, target 255 → trim to 255 around mid=1200
    s, e = resize_to_length(1000, 1400, target_len=255, chrom_size=100000)
    # midpoint = (1000+1400)//2 = 1200; half = 127; new_start = 1073
    assert (s, e) == (1073, 1328)


def test_shorter_input_padded_around_midpoint() -> None:
    # input 100 bp, target 255 → pad to 255 around mid=1050
    s, e = resize_to_length(1000, 1100, target_len=255, chrom_size=100000)
    # midpoint = 1050; half = 127; new_start = 923; new_end = 1178
    assert (s, e) == (923, 1178)


def test_left_edge_shifts_inward() -> None:
    # Window centered at chr-position 50 with target 255 would extend to -77
    s, e = resize_to_length(40, 60, target_len=255, chrom_size=100000)
    assert s == 0
    assert e - s == 255


def test_right_edge_shifts_inward() -> None:
    s, e = resize_to_length(99950, 99970, target_len=255, chrom_size=100000)
    assert e == 100000
    assert e - s == 255


def test_chrom_size_equal_to_target_len_works() -> None:
    s, e = resize_to_length(0, 100, target_len=255, chrom_size=255)
    assert (s, e) == (0, 255)


def test_chrom_size_smaller_than_target_raises() -> None:
    with pytest.raises(ValueError, match="cannot fit"):
        resize_to_length(0, 100, target_len=255, chrom_size=200)


def test_empty_interval_raises() -> None:
    with pytest.raises(ValueError, match="empty interval"):
        resize_to_length(100, 100, target_len=255, chrom_size=10000)
    with pytest.raises(ValueError, match="empty interval"):
        resize_to_length(100, 90, target_len=255, chrom_size=10000)


def test_nonpositive_target_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        resize_to_length(100, 200, target_len=0, chrom_size=10000)
    with pytest.raises(ValueError, match="positive"):
        resize_to_length(100, 200, target_len=-5, chrom_size=10000)


@pytest.mark.parametrize("target_len", [1, 2, 100, 255, 256, 999])
def test_output_length_exactly_target(target_len: int) -> None:
    s, e = resize_to_length(500, 600, target_len=target_len, chrom_size=10000)
    assert e - s == target_len
    assert 0 <= s <= e <= 10000
