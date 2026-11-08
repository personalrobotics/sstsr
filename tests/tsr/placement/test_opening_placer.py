# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Oracle-backed release properties for ``OpeningPlacer`` (issue #200).

The release contract (``docs/ARCHITECTURE.md``) is a claim about **every** pose a region
admits, so the region is sampled -- midpoint, per-dimension extrema, corners, interior --
and each pose is certified by :func:`_placement_oracle.certify_release`, which reads
resting, containment and height off the posed geometry's support function. Completeness
is checked against the same oracle: every rest :meth:`StablePlacer.place` finds whose
footprint, swept through every yaw, fits inside the opening must be returned.

The mutants at the end are the defects this suite exists to catch; each must fail it.
"""

from __future__ import annotations

import contextlib
import logging
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import tsr.placement.opening_placer as opl
from tsr.placement import Cylinder, OpeningPlacer, Shape, StablePlacer

from .._hypothesis_strategies import transforms
from ._placement_matrix import SHAPE_KINDS, bw_samples, matrix_settings, oracle_shape, place_shape
from ._placement_oracle import _FAN, certify_release, length_atol


def _placer(opening, depth) -> OpeningPlacer:
    kind, extent = opening
    return OpeningPlacer(size=extent, depth=depth) if kind == "rectangle" else OpeningPlacer(radius=extent, depth=depth)


def _fits(prim, R, opening, clearance) -> bool:
    """Whether the object at rotation ``R``, its origin at the opening's centre, stays
    ``clearance`` inside the opening at every yaw: its footprint about the origin, the
    largest horizontal reach, within the opening's smallest inset half-extent."""
    footprint = float(prim.support(_FAN, R).max())
    kind, extent = opening
    room = min(extent) / 2 - clearance if kind == "rectangle" else extent - clearance
    return footprint <= room + length_atol(prim.scale)


def release_failures(shape, opening, depth, lowest, clearance, *, T=None, rng=None, templates=None):
    """Every admitted pose certified, and every fitting rest returned."""
    prim = oracle_shape(shape)
    if templates is None:
        templates = _placer(opening, depth).place(shape, lowest=lowest, clearance=clearance)
    T = np.eye(4) if T is None else T
    T_inv = np.linalg.inv(T)
    failures = []
    for t in templates:
        tsr = t.instantiate(T)
        for xi in bw_samples(t, rng=rng):
            w = certify_release(prim, T_inv @ tsr.to_transform(xi), opening=opening, lowest=lowest, clearance=clearance)
            if not w.ok:
                failures.append(f"[{t.variant}] xi={np.round(xi, 6).tolist()}: {w.failed}")
    returned = {t.variant for t in templates}
    for rest in StablePlacer(1.0e6, 1.0e6).place(shape):
        if _fits(prim, rest.Tw_e[:3, :3], opening, clearance) and rest.variant not in returned:
            failures.append(f"[{rest.variant}] fits the opening but was not returned")
    return failures


@st.composite
def releases(draw):
    scale = draw(st.sampled_from([1e-2, 0.1, 1.0]))
    shape = place_shape(draw(st.sampled_from(SHAPE_KINDS)), scale)
    if draw(st.booleans()):
        opening = ("rectangle", (draw(st.floats(0.2, 3.0)) * scale, draw(st.floats(0.2, 3.0)) * scale))
    else:
        opening = ("circle", draw(st.floats(0.1, 1.5)) * scale)
    clearance = draw(st.floats(0.0, 0.1)) * scale
    depth = draw(st.floats(0.1, 3.0)) * scale
    low = draw(st.floats(-1.0, 1.0)) * depth
    lowest = (low, low + draw(st.floats(0.0, 0.5)) * scale)
    return shape, opening, depth, lowest, clearance


@matrix_settings(300)
@given(case=releases(), T=transforms(), seed=st.integers(0, 2**32 - 1))
def test_every_admitted_pose_is_a_release_through_the_opening(case, T, seed):
    shape, opening, depth, lowest, clearance = case
    assert not release_failures(shape, opening, depth, lowest, clearance, T=T, rng=np.random.default_rng(seed))


#: A recycling bin's opening, as prl_assets declares Geodude's: 0.238 m square, 0.757 m
#: deep, with 10 mm of clearance and the 16 mm a release drifts sideways.
BIN = ("rectangle", (0.238, 0.238))
BIN_DEPTH = 0.757
BIN_CLEARANCE = 0.026


def test_a_tall_bottle_is_released_only_standing():
    """A 0.22 m bottle lying rolls, so it has no rest on its side, and standing it fits."""
    bottle = place_shape("bottle", 1.0)
    templates = OpeningPlacer(size=BIN[1], depth=BIN_DEPTH).place(bottle, lowest=(0.0, 0.05), clearance=BIN_CLEARANCE)
    assert templates
    for t in templates:
        up = t.Tw_e[:3, :3] @ np.array([0.0, 0.0, 1.0])
        assert abs(abs(up[2]) - 1.0) < 1e-9, t.variant
    assert not release_failures(bottle, BIN, BIN_DEPTH, (0.0, 0.05), BIN_CLEARANCE, templates=templates)


def test_an_object_wider_than_the_opening_is_an_empty_feasible_set(caplog):
    shape = place_shape("box", 1.0)  # 0.2 x 0.1 x 0.3 m: its narrowest rest reaches 0.112 m
    with caplog.at_level(logging.DEBUG, logger="tsr.placement.opening_placer"):
        assert OpeningPlacer(size=(0.2, 0.2), depth=0.5).place(shape, lowest=(0.0, 0.1)) == []
    assert "exceeds_opening" in caplog.text


def test_a_circle_admits_its_inscribed_square_exactly():
    """The origin of an upright cylinder of radius r slides in the square inscribed in the
    circle of radius R - c - r: at its corners the cylinder touches the clearance."""
    R, c, r = 0.10, 0.01, 0.03
    can = Shape([Cylinder(r, 0.12)])
    (t,) = [
        t for t in OpeningPlacer(radius=R, depth=0.3).place(can, lowest=(0.0, 0.0), clearance=c) if "-z" in t.variant
    ]
    corner = math.hypot(t.Bw[0, 1], t.Bw[1, 1])
    assert corner + r == pytest.approx(R - c, abs=1e-12)


def test_a_footprint_that_fits_only_at_the_centre_is_the_boundary():
    r, c = 0.05, 0.01
    can = Shape([Cylinder(r, 0.12)])
    # A circle pins the origin at its centre; a narrow rectangle pins only x.
    for placer, pinned in (
        (OpeningPlacer(radius=r + c, depth=0.3), 2),
        (OpeningPlacer(size=(2 * (r + c), 3.0), depth=0.3), 1),
    ):
        standing = placer.place(can, lowest=(0.0, 0.0), clearance=c)
        assert standing and all(np.allclose(t.Bw[:pinned], 0.0) for t in standing)
        assert placer.place(Shape([Cylinder(r * 1.01, 0.12)]), lowest=(0.0, 0.0), clearance=c) == []


@pytest.mark.parametrize(
    "build",
    [
        lambda: OpeningPlacer(depth=0.5),
        lambda: OpeningPlacer(size=(0.2, 0.2), radius=0.1, depth=0.5),
        lambda: OpeningPlacer(size=(0.2,), depth=0.5),
        lambda: OpeningPlacer(size=(0.2, float("nan")), depth=0.5),
        lambda: OpeningPlacer(radius=-0.1, depth=0.5),
        lambda: OpeningPlacer(radius=0.1, depth=0.0),
    ],
)
def test_a_malformed_opening_raises(build):
    with pytest.raises(ValueError):
        build()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lowest": (0.1, 0.0)},  # low above high
        {"lowest": (-0.6, 0.0)},  # below the floor, 0.5 m down
        {"lowest": (0.0, float("inf"))},
        {"lowest": (0.0,)},
        {"lowest": (0.0, 0.1), "clearance": -0.01},
        {"lowest": (0.0, 0.1), "clearance": float("nan")},
    ],
)
def test_a_malformed_release_raises(kwargs):
    with pytest.raises(ValueError):
        OpeningPlacer(size=(0.3, 0.3), depth=0.5).place(place_shape("can", 1.0), **kwargs)


# --------------------------------------------------------------------------- #
# Mutation gate: each corrupted generator must fail the checks above
# --------------------------------------------------------------------------- #

#: Cases that see every mutant: a tall bottle and a mug in the bin, cans in a circle.
CORPUS = [
    (place_shape("bottle", 1.0), BIN, BIN_DEPTH, (-0.1, 0.05), BIN_CLEARANCE),
    (place_shape("mug", 1.0), BIN, BIN_DEPTH, (0.0, 0.1), BIN_CLEARANCE),
    (place_shape("can", 1.0), ("circle", 0.09), 0.3, (-0.05, 0.02), 0.01),
    (place_shape("box", 1.0), ("rectangle", (0.5, 0.3)), 0.4, (0.0, 0.0), 0.02),
]


def _corpus_failures():
    rng = np.random.default_rng(0)
    return [f for case in CORPUS for f in release_failures(*case, rng=rng)]


_HALVES = opl.OpeningPlacer._halves


@contextlib.contextmanager
def _patched(name, value):
    original = getattr(opl.OpeningPlacer, name)
    setattr(opl.OpeningPlacer, name, value)
    try:
        yield
    finally:
        setattr(opl.OpeningPlacer, name, original)


def _circle_as_square(self, footprint, clearance):
    """The circle's radius taken as a square's half-extent, overhanging at the corners."""
    if self.size is not None:
        return (self.size[0] / 2 - clearance - footprint, self.size[1] / 2 - clearance - footprint)
    return (self.radius - clearance - footprint,) * 2


def _no_clearance(self, footprint, clearance):
    return _HALVES(self, footprint, 0.0)


def _no_footprint(self, footprint, clearance):
    return _HALVES(self, 0.0, clearance)


def _refuse_all(self, footprint, clearance):
    return (-1.0, -1.0)


def _heights_of_the_origin(place):
    """The heights applied to the object's origin rather than its lowest point."""

    def wrapped(self, shape, **kwargs):
        templates = place(self, shape, **kwargs)
        for t in templates:
            t.T_ref_tsr[2, 3] = 0.0
        return templates

    return wrapped


def test_the_corpus_passes_unmutated():
    assert not _corpus_failures()


@pytest.mark.parametrize("mutant", [_circle_as_square, _no_clearance, _no_footprint, _refuse_all])
def test_a_corrupted_inset_is_caught(mutant):
    with _patched("_halves", mutant):
        assert _corpus_failures()


def test_heights_of_the_origin_are_caught():
    with _patched("place", _heights_of_the_origin(opl.OpeningPlacer.place)):
        assert _corpus_failures()
