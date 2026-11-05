# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Oracle-backed soundness properties for every placement factory (issue #148).

The placement contract (``docs/ARCHITECTURE.md``) is a claim about **every** pose a
region admits, so these tests sample the region -- midpoint, per-dimension extrema,
corners, interior -- and certify each pose against the independent
:mod:`_placement_oracle`, which derives resting, support and containment from the posed
geometry's support function and never from the generator's formulas.

The defects this suite exists to prevent, each reproduced here deterministically:

* #149 -- roll/pitch freedom rotated the resting height, burying the sphere;
* #150 -- no footprint inset, so poses hung off the surface, and an object too large
  for the surface still returned templates;
* #154 -- non-finite dimensions passed ingress and produced finite templates.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tsr import placement as tp
from tsr.placement import StablePlacer

from .._hypothesis_strategies import transforms
from ._placement_matrix import (
    FACTORIES,
    bw_samples,
    cases,
    equivariance_failures,
    face_label_failures,
    matrix_settings,
    public_factories,
    soundness_failures,
)
from ._placement_oracle import Cylinder, Sphere, certify


def test_every_public_place_factory_is_in_the_matrix():
    assert public_factories() == sorted(FACTORIES)


# --------------------------------------------------------------------------- #
# The three clauses, over the whole region
# --------------------------------------------------------------------------- #


@matrix_settings(400)
@given(case=cases(), seed=st.integers(0, 2**32 - 1))
def test_admitted_poses_are_stable_placements(case, seed):
    templates = case.call()
    rng = np.random.default_rng(seed)
    assert not soundness_failures(case, templates, rng=rng)


@matrix_settings(150)
@given(case=cases(fit="fits"), T=transforms(), seed=st.integers(0, 2**32 - 1))
def test_soundness_is_independent_of_the_surface_pose(case, T, seed):
    """The claim is about the surface frame, so an arbitrarily placed surface is the
    same claim. This is the coverage gap that left every placement assertion at an
    identity table."""
    templates = case.call()
    rng = np.random.default_rng(seed)
    assert not soundness_failures(case, templates, table_pose=T, rng=rng)
    assert not equivariance_failures(case, templates, T)


@matrix_settings(150)
@given(case=cases(fit="fits"))
def test_face_labels_name_the_resting_face(case):
    assert not face_label_failures(case, case.call())


# --------------------------------------------------------------------------- #
# #149 -- the sphere's freed roll and pitch must rotate it in place
# --------------------------------------------------------------------------- #


def test_sphere_rests_at_every_admitted_orientation():
    r = 0.05
    placer = StablePlacer(table_x=0.30, table_y=0.20)
    (t,) = placer.place_sphere(r)
    # The freedom the factory advertises, which is the reason the defect mattered.
    np.testing.assert_allclose(t.Bw[3], [-np.pi, np.pi])
    np.testing.assert_allclose(t.Bw[4], [-np.pi, np.pi])
    np.testing.assert_allclose(t.Bw[5], [-np.pi, np.pi])

    tsr = t.instantiate(np.eye(4))
    prim = Sphere(r)
    rng = np.random.default_rng(0)
    for xi in bw_samples(t, rng=rng, n_interior=200):
        pose = tsr.to_transform(xi)
        # Before the fix the centre was r·cos(roll)·cos(pitch): at roll = pi the whole
        # sphere sat under the surface.
        assert np.isclose(pose[2, 3], r, atol=1e-12), f"centre at z={pose[2, 3]} for xi={xi}"
        assert certify(prim, pose, table=(0.30, 0.20)).ok


def test_freeing_roll_and_pitch_cannot_move_the_resting_height():
    """The invariant behind #149, stated structurally: ``Tw_e`` is a pure rotation, so
    no rotation the region admits can move the height that ``T_ref_tsr`` carries."""
    placer = StablePlacer(table_x=0.30, table_y=0.20)
    templates = (
        placer.place_cylinder(0.04, 0.12)
        + placer.place_box(0.10, 0.08, 0.06)
        + placer.place_sphere(0.05)
        + placer.place_torus(0.05, 0.012)
    )
    for t in templates:
        np.testing.assert_allclose(t.Tw_e[:3, 3], np.zeros(3), atol=0.0)
        assert t.T_ref_tsr[2, 3] > 0.0


# --------------------------------------------------------------------------- #
# #150 -- the footprint inset and the empty feasible set
# --------------------------------------------------------------------------- #


def test_box_at_the_worst_yaw_stays_on_the_surface():
    """The reported case: a 0.20 x 0.08 box whose centre could reach the surface edge.

    The required inset is the half-diagonal (0.1077 m), not ``lx/2`` -- at yaw 0.464 rad
    the overhang was 0.112 m with a per-axis inset.
    """
    placer = StablePlacer(table_x=0.30, table_y=0.20)
    lx, ly, lz = 0.20, 0.08, 0.28
    half_diag = np.hypot(lx, ly) / 2.0
    for t in placer.place_box(lx, ly, lz):
        if t.variant in ("-z", "+z"):
            np.testing.assert_allclose(t.Bw[0], [-(0.30 - half_diag), 0.30 - half_diag], atol=1e-12)
            np.testing.assert_allclose(t.Bw[1], [-(0.20 - half_diag), 0.20 - half_diag], atol=1e-12)
    # And behaviourally: the corners of the region keep the whole box on the surface.
    verts = np.array([[sx * lx / 2, sy * ly / 2, sz * lz / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    for t in placer.place_box(lx, ly, lz):
        tsr = t.instantiate(np.eye(4))
        for xi in bw_samples(t, rng=np.random.default_rng(1), n_interior=50):
            pose = tsr.to_transform(xi)
            world = verts @ pose[:3, :3].T + pose[:3, 3]
            assert world[:, 0].max() <= 0.30 + 1e-9 and world[:, 0].min() >= -0.30 - 1e-9
            assert world[:, 1].max() <= 0.20 + 1e-9 and world[:, 1].min() >= -0.20 - 1e-9


@pytest.mark.parametrize(
    "call,reason",
    [
        (lambda p: p.place_cylinder(2.0, 0.1), "exceeds_surface"),
        (lambda p: p.place_box(1.0, 1.0, 1.0), "exceeds_surface"),
        (lambda p: p.place_sphere(1.0), "exceeds_surface"),
        (lambda p: p.place_torus(1.0, 0.1), "exceeds_surface"),
    ],
)
def test_object_too_large_is_an_empty_feasible_set(call, reason, caplog):
    """A 4 m disc on a 0.6 x 0.4 m table is a valid request with no feasible pose: the
    factory contract says ``[]`` with the reason logged, not an unusable region."""
    placer = StablePlacer(table_x=0.30, table_y=0.20)
    with caplog.at_level(logging.DEBUG, logger="tsr.placement.stable_placer"):
        assert call(placer) == []
    messages = [r.getMessage() for r in caplog.records if "empty feasible set" in r.getMessage()]
    assert len(messages) == 1 and reason in messages[0]


def test_mesh_too_large_is_an_empty_feasible_set(caplog):
    verts = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * 0.5
    placer = StablePlacer(table_x=0.30, table_y=0.20)
    with caplog.at_level(logging.DEBUG, logger="tsr.placement.stable_placer"):
        assert placer.place_mesh(verts, verts.mean(axis=0)) == []
    messages = [r.getMessage() for r in caplog.records if "empty feasible set" in r.getMessage()]
    assert len(messages) == 1 and "exceeds_surface" in messages[0]


def test_footprint_exactly_equal_to_the_surface_is_the_boundary():
    """Equality is feasible and leaves an exact zero-width interval; one ulp past it is
    still feasible within the scale tolerance, and a clear excess is not."""
    r = 0.05
    placer = StablePlacer(table_x=r, table_y=r)
    (t,) = placer.place_sphere(r)
    np.testing.assert_allclose(t.Bw[0], [0.0, 0.0], atol=0.0)
    np.testing.assert_allclose(t.Bw[1], [0.0, 0.0], atol=0.0)
    assert certify(Sphere(r), t.instantiate(np.eye(4)).to_transform(np.zeros(6)), table=(r, r)).ok

    assert len(placer.place_sphere(np.nextafter(r, 0.0))) == 1
    assert len(placer.place_sphere(np.nextafter(r, 1.0))) == 1  # within length_atol(2r)
    assert placer.place_sphere(r * 1.01) == []


def test_partial_fit_keeps_only_the_faces_that_fit(caplog):
    """A tall thin box fits on its ends but not on its long sides."""
    placer = StablePlacer(table_x=0.06, table_y=0.06)
    with caplog.at_level(logging.DEBUG, logger="tsr.placement.stable_placer"):
        templates = placer.place_box(lx=0.05, ly=0.05, lz=0.40)
    assert sorted(t.variant for t in templates) == ["+z", "-z"]
    assert any("variants dropped" in r.getMessage() for r in caplog.records)
    verts = np.array([[sx * 0.025, sy * 0.025, sz * 0.20] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    for t in templates:
        tsr = t.instantiate(np.eye(4))
        for xi in bw_samples(t, rng=np.random.default_rng(2)):
            pose = tsr.to_transform(xi)
            world = verts @ pose[:3, :3].T + pose[:3, 3]
            assert np.abs(world[:, :2]).max() <= 0.06 + 1e-9


def test_cylinder_on_a_snug_surface_still_rests_and_fits():
    placer = StablePlacer(table_x=0.041, table_y=0.041)
    templates = placer.place_cylinder(0.04, 0.12)
    assert len(templates) == 2
    prim = Cylinder(0.04, 0.12)
    for t in templates:
        tsr = t.instantiate(np.eye(4))
        for xi in bw_samples(t, rng=np.random.default_rng(3)):
            assert certify(prim, tsr.to_transform(xi), table=(0.041, 0.041)).ok


# --------------------------------------------------------------------------- #
# #154 -- non-finite input is an invalid request, not an empty feasible set
# --------------------------------------------------------------------------- #

nonfinite = [np.nan, np.inf, -np.inf]


@pytest.mark.parametrize("bad", nonfinite)
def test_non_finite_surface_extents_raise(bad):
    with pytest.raises(ValueError, match="table_x"):
        StablePlacer(table_x=bad, table_y=0.2)
    with pytest.raises(ValueError, match="table_y"):
        StablePlacer(table_x=0.2, table_y=bad)


@pytest.mark.parametrize("bad", nonfinite)
@pytest.mark.parametrize(
    "call,argument",
    [
        (lambda p, v: p.place_cylinder(v, 0.1), "cylinder_radius"),
        (lambda p, v: p.place_cylinder(0.04, v), "cylinder_height"),
        (lambda p, v: p.place_box(v, 0.1, 0.1), "lx"),
        (lambda p, v: p.place_box(0.1, v, 0.1), "ly"),
        (lambda p, v: p.place_box(0.1, 0.1, v), "lz"),
        (lambda p, v: p.place_sphere(v), "radius"),
        (lambda p, v: p.place_torus(v, 0.01), "major_radius"),
        (lambda p, v: p.place_torus(0.05, v), "minor_radius"),
    ],
)
def test_non_finite_dimensions_raise_naming_the_argument(call, argument, bad):
    placer = StablePlacer(table_x=0.30, table_y=0.20)
    with pytest.raises(ValueError, match=argument):
        call(placer, bad)


@pytest.mark.parametrize("bad", nonfinite)
def test_non_finite_mesh_arguments_raise(bad):
    placer = StablePlacer(table_x=0.30, table_y=0.20)
    verts = np.array([[sx * 0.05, sy * 0.05, sz * 0.05] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    com = verts.mean(axis=0)
    with pytest.raises(ValueError, match="vertices"):
        bad_verts = verts.copy()
        bad_verts[0, 0] = bad
        placer.place_mesh(bad_verts, com)
    with pytest.raises(ValueError, match="com"):
        placer.place_mesh(verts, np.array([bad, 0.0, 0.0]))
    with pytest.raises(ValueError, match="min_margin_deg"):
        placer.place_mesh(verts, com, min_margin_deg=bad)


@pytest.mark.parametrize("bad", nonfinite)
@pytest.mark.parametrize(
    "build,argument",
    [
        (lambda v: tp.Box((v, 0.1, 0.1)), "size"),
        (lambda v: tp.Cylinder(v, 0.1), "radius"),
        (lambda v: tp.Cylinder(0.04, v), "height"),
        (lambda v: tp.Sphere(v), "radius"),
        (lambda v: tp.Capsule(0.02, v), "length"),
        (lambda v: tp.Hull(np.full((4, 3), v)), "vertices"),
        (lambda v: tp.Box((0.1, 0.1, 0.1), pose=np.diag([v, 1.0, 1.0, 1.0])), "pose"),
        (lambda v: tp.Shape([tp.Sphere(0.05)], com=np.array([v, 0.0, 0.0])), "com"),
    ],
)
def test_non_finite_shape_arguments_raise_naming_the_argument(build, argument, bad):
    with pytest.raises(ValueError, match=argument):
        build(bad)


@pytest.mark.parametrize(
    "build,argument",
    [
        (lambda: tp.Box((0.1, 0.1, 0.1), pose=np.diag([-1.0, 1.0, 1.0, 1.0])), "determinant"),  # a reflection
        (lambda: tp.Shape([]), "parts"),
        (lambda: tp.Shape(tp.Sphere(0.05)), "parts"),  # one part, not a sequence of them
        (lambda: tp.Shape([np.eye(4)]), "parts"),
        (lambda: tp.Hull([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]), "vertices"),  # flat
        (lambda: StablePlacer(0.3, 0.2).place([tp.Sphere(0.05)]), "Shape"),
    ],
)
def test_malformed_shapes_raise(build, argument):
    with pytest.raises(ValueError, match=argument):
        build()


@pytest.mark.parametrize(
    "part,factory",
    [
        (tp.Cylinder(0.04, 0.12), lambda p: p.place_cylinder(0.04, 0.12)),
        (tp.Box((0.2, 0.1, 0.3)), lambda p: p.place_box(0.2, 0.1, 0.3)),
        (tp.Sphere(0.05), lambda p: p.place_sphere(0.05)),
    ],
)
def test_place_on_a_centred_primitive_is_its_exact_factory(part, factory):
    """The exact path is exact: a lone centred primitive admits the same poses, region by
    region, as its own factory."""
    placer = StablePlacer(table_x=0.30, table_y=0.20)
    placed, direct = placer.place(tp.Shape([part])), factory(placer)
    assert [t.variant for t in placed] == [t.variant for t in direct]
    for a, b in zip(placed, direct):
        for xi in bw_samples(b, rng=np.random.default_rng(0)):
            pose, expected = (t.instantiate(np.eye(4)).to_transform(xi) for t in (a, b))
            np.testing.assert_allclose(pose, expected, atol=1e-12)


def test_place_frames_an_object_off_its_part():
    """A can framed at its base's centre, as an asset is, stands with that origin on the
    surface: nobody shifts it by half its height."""
    T = np.eye(4)
    T[2, 3] = 0.0615
    upright = StablePlacer(table_x=0.30, table_y=0.20).place(tp.Shape([tp.Cylinder(0.033, 0.123, pose=T)]))[0]
    assert upright.variant == "-z"
    np.testing.assert_allclose(upright.instantiate(np.eye(4)).to_transform(np.zeros(6)), np.eye(4), atol=1e-12)


def test_a_part_copies_its_pose():
    """A caller may reuse one buffer for every geom's pose."""
    T = np.eye(4)
    part = tp.Cylinder(0.03, 0.1, pose=T)
    T[2, 3] = 0.5
    assert T.flags.writeable and part.pose[2, 3] == 0.0


def test_a_variant_names_the_same_face_whatever_is_filtered():
    """A turned box's faces off the object's axes are named by the box's own faces, so a
    stricter ``min_margin_deg`` drops faces without renaming the ones it keeps."""
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    shape = tp.Shape([tp.Box((0.1, 0.2, 0.4), pose=T)])
    placer = StablePlacer(table_x=1.0, table_y=1.0)
    every = {t.variant: t.provenance.face_normal for t in placer.place(shape)}
    strict = {t.variant: t.provenance.face_normal for t in placer.place(shape, min_margin_deg=20.0)}
    assert strict and len(strict) < len(every)
    assert all(every[v] == n for v, n in strict.items())


def test_an_off_centre_sphere_leaves_the_object_as_framed():
    """A sphere's own rotation means nothing, so it cannot tilt the object it is part of."""
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # turned 90° about x
    T[:3, 3] = [0.1, 0.0, 0.05]
    (t,) = StablePlacer(table_x=1.0, table_y=1.0).place(tp.Shape([tp.Sphere(0.05, pose=T)]))
    np.testing.assert_allclose(t.Tw_e[:3, :3], np.eye(3), atol=1e-12)


@matrix_settings(250)
@given(case=cases())
def test_no_template_has_a_non_finite_entry(case):
    for t in case.call():
        assert np.all(np.isfinite(t.Tw_e)) and np.all(np.isfinite(t.T_ref_tsr)) and np.all(np.isfinite(t.Bw))
