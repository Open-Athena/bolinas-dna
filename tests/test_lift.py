"""Tests for bolinas.lift.cigar_lift — exact coord lift via CIGAR walking."""

import pytest

from bolinas.lift import cigar_lift


def test_gapfree_forward_full_alignment() -> None:
    # 100 bp aligned, identical mapping; lifting the whole query should
    # return the whole target.
    assert cigar_lift(
        q_start=100, q_end=200, t_start=500, t_end=600,
        strand="+", cigar="100=",
        lift_q_start=100, lift_q_end=200,
    ) == (500, 600)


def test_gapfree_forward_subinterval() -> None:
    # Gap-free alignment: linear interpolation equals exact lift.
    assert cigar_lift(
        q_start=100, q_end=200, t_start=500, t_end=600,
        strand="+", cigar="100M",
        lift_q_start=120, lift_q_end=150,
    ) == (520, 550)


def test_forward_with_insertion_in_query() -> None:
    # Alignment: 50M 10I 40M — query has 10 bp insertion at query pos [150, 160)
    # Query total 100, target total 90.
    # q 100..150 maps to t 500..550
    # q 150..160 is query-only insertion (no target advance)
    # q 160..200 maps to t 550..590
    # Lifting q [140, 170) should give t [540, 560) — includes the insertion
    # (the insertion contributes 0 target bases).
    assert cigar_lift(
        q_start=100, q_end=200, t_start=500, t_end=590,
        strand="+", cigar="50M10I40M",
        lift_q_start=140, lift_q_end=170,
    ) == (540, 560)


def test_forward_with_deletion_in_query() -> None:
    # Alignment: 50M 10D 40M — target has 10 bp extra at target pos [550, 560)
    # Query total 90, target total 100.
    # q 100..150 maps to t 500..550
    # deletion (no query advance) from t 550 to 560
    # q 150..190 maps to t 560..600
    # Lifting q [140, 150) should give t [540, 550)
    # Lifting q [150, 160) should give t [560, 570) — spans the deletion
    assert cigar_lift(
        q_start=100, q_end=190, t_start=500, t_end=600,
        strand="+", cigar="50M10D40M",
        lift_q_start=140, lift_q_end=150,
    ) == (540, 550)
    assert cigar_lift(
        q_start=100, q_end=190, t_start=500, t_end=600,
        strand="+", cigar="50M10D40M",
        lift_q_start=150, lift_q_end=160,
    ) == (560, 570)


def test_reverse_strand_full_alignment() -> None:
    # 100 bp aligned, query reverse-complemented.
    # q 100..200 aligns to t 500..600 reversed.
    # Lifting the whole query should still return the whole target.
    assert cigar_lift(
        q_start=100, q_end=200, t_start=500, t_end=600,
        strand="-", cigar="100=",
        lift_q_start=100, lift_q_end=200,
    ) == (500, 600)


def test_reverse_strand_subinterval() -> None:
    # Reverse-complement alignment, gap-free.
    # q_start..q_end forward walks t_end..t_start backward.
    # q=100 maps to t=600, q=200 maps to t=500 (both exclusive end/start).
    # Subquery [120, 150) covers 30 bp at query offset [20, 50) into the
    # alignment. On target, these 30 bp are at offset [50, 80) from the
    # reverse end — i.e., at target [520, 550) (not [550, 580)!).
    # Walk: q=100 -> t=600, advance 20 query bases -> q=120, t=580.
    # This is where the lift starts (target end boundary).
    # Advance 30 more -> q=150, t=550. Lift end boundary.
    # Returned as (min, max) = (550, 580).
    assert cigar_lift(
        q_start=100, q_end=200, t_start=500, t_end=600,
        strand="-", cigar="100M",
        lift_q_start=120, lift_q_end=150,
    ) == (550, 580)


def test_lift_outside_alignment_returns_none() -> None:
    assert cigar_lift(
        q_start=100, q_end=200, t_start=500, t_end=600,
        strand="+", cigar="100M",
        lift_q_start=50, lift_q_end=90,
    ) is None
    assert cigar_lift(
        q_start=100, q_end=200, t_start=500, t_end=600,
        strand="+", cigar="100M",
        lift_q_start=250, lift_q_end=300,
    ) is None


def test_lift_clipped_to_alignment() -> None:
    # Lift interval extends beyond alignment; clipped at boundaries.
    # q 100..200 -> t 500..600, gap-free.
    # lift_q [50, 150) → clip to [100, 150) → t [500, 550).
    assert cigar_lift(
        q_start=100, q_end=200, t_start=500, t_end=600,
        strand="+", cigar="100M",
        lift_q_start=50, lift_q_end=150,
    ) == (500, 550)
