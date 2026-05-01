import pytest

from bolinas.projection.resize import resize_to_length


def test_identity_when_already_target_length():
    assert resize_to_length(100, 355, 255, chrom_size=10_000) == (100, 355)


def test_trim_when_longer_than_target():
    # length 500 -> trim to 255 around midpoint 350
    s, e = resize_to_length(100, 600, 255, chrom_size=10_000)
    assert e - s == 255
    # midpoint preserved (mod parity)
    assert abs(((s + e) // 2) - 350) <= 1


def test_pad_when_shorter_than_target():
    # length 50 -> pad to 255 around midpoint 4025 (well clear of either end).
    s, e = resize_to_length(4_000, 4_050, 255, chrom_size=10_000)
    assert e - s == 255
    assert abs(((s + e) // 2) - 4_025) <= 1


def test_clamp_at_chrom_start_shifts_right():
    # Interval near 0; centering would go negative -> shift right.
    s, e = resize_to_length(0, 10, 255, chrom_size=10_000)
    assert s == 0
    assert e == 255


def test_clamp_at_chrom_end_shifts_left():
    # Interval near chrom end; centering would exceed -> shift left.
    s, e = resize_to_length(9_990, 10_000, 255, chrom_size=10_000)
    assert s == 9_745
    assert e == 10_000


def test_raises_when_chrom_smaller_than_target():
    with pytest.raises(ValueError, match="cannot fit"):
        resize_to_length(0, 50, 255, chrom_size=100)


def test_raises_on_empty_interval():
    with pytest.raises(ValueError, match="empty"):
        resize_to_length(100, 100, 255, chrom_size=10_000)


def test_raises_on_nonpositive_target():
    with pytest.raises(ValueError, match="positive"):
        resize_to_length(100, 200, 0, chrom_size=10_000)


@pytest.mark.parametrize(
    "start,end",
    [(0, 1), (50, 100), (5_000, 5_001), (9_745, 10_000)],
)
def test_output_invariants_random_inputs(start, end):
    s, e = resize_to_length(start, end, 255, chrom_size=10_000)
    assert e - s == 255
    assert 0 <= s <= e <= 10_000
