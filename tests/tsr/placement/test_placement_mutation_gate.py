# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Mutation gate: the placement suite must FAIL on a corrupted generator (issue #159).

The placement counterpart of ``tests/tsr/hands/test_grasp_mutation_gate.py``. A suite
that has never failed proves nothing — and that is not hypothetical here: the 53 tests
this suite replaced passed while all eight defects of #148 sat underneath them.

Every defect that epic fixed is injected again, and the matrix's own check functions
must reject it:

* **black-box mutants** wrap the real factory's output (the resting height, the ``Bw``
  inset, the variant labels), so they survive a refactor of the internals;
* **code mutants** patch the real generator (``_plane_basis``, ``_inward_distance``,
  ``_group_coplanar_facets``, ``_tipping_angle``), so a corrupted *implementation*, not
  merely a corrupted result, is shown to be caught.

Two of them pin the facet-merge tolerance from opposite sides. Under-merging fragments
a float32 face until nothing supports the object; over-merging collapses a tessellated
curved surface into a face the caller never supplied. Only testing both directions pins
a tolerance.

Detection can legitimately be thin. ``resting_height_in_tw_e`` is caught by exactly one
corpus case, because moving the resting height into ``Tw_e`` is a true no-op wherever
roll and pitch are fixed: only the sphere, which frees them, can see it. One case is
enough, and widening the corpus would not make the mutation more detectable.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import Callable, Iterator, List

import numpy as np
import pytest

import tsr.placement._stable_poses as sp
import tsr.placement.shape as shp
import tsr.placement.stable_placer as spl
from tsr.placement import StablePlacer
from tsr.template import TSRTemplate

from ._placement_matrix import corpus, corpus_failures

# --------------------------------------------------------------------------- #
# Black-box mutants: wrap what the real factory returned
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _patch(owner, name: str, value) -> Iterator[None]:
    original = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, original)


def resting_height_in_tw_e() -> contextlib.AbstractContextManager:
    """#149: carry the resting height in ``Tw_e``'s translation, where the region's own
    roll and pitch rotate it."""

    def _template(
        self, name, description, variant, R, origin_height, Bw, subject, stability_margin=None, provenance=None
    ):
        Tw_e = np.eye(4)
        Tw_e[:3, :3] = R
        Tw_e[2, 3] = float(origin_height)
        return TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=Tw_e,
            Bw=Bw,
            task="place",
            subject=subject,
            reference=self.reference,
            name=name,
            description=description,
            variant=variant,
            stability_margin=stability_margin,
            provenance=provenance,
        )

    return _patch(StablePlacer, "_template", _template)


def no_footprint_inset() -> contextlib.AbstractContextManager:
    """#150: slide the object's origin over the whole surface, and never report an
    empty feasible set."""

    @contextlib.contextmanager
    def apply():
        def _bw(self, footprint_radius, roll_range=None, pitch_range=None):
            return original_bw(self, 0.0, roll_range, pitch_range)

        original_bw = StablePlacer._bw
        with _patch(StablePlacer, "_bw", _bw), _patch(StablePlacer, "_fits", lambda self, radius, scale: True):
            yield

    return apply()


def per_axis_inset() -> contextlib.AbstractContextManager:
    """#150, the *plausible* wrong fix: inset each axis by the resting face's own
    half-extent, which still overhangs by the half-diagonal at an oblique yaw."""

    @contextlib.contextmanager
    def apply():
        def place_box(self, lx, ly, lz, subject="object", min_margin_deg=0.0):
            templates = original(self, lx, ly, lz, subject=subject, min_margin_deg=min_margin_deg)
            half = {
                "-z": (lx / 2, ly / 2),
                "+z": (lx / 2, ly / 2),
                "-y": (lx / 2, lz / 2),
                "+y": (lx / 2, lz / 2),
                "-x": (ly / 2, lz / 2),
                "+x": (ly / 2, lz / 2),
            }
            for t in templates:
                hx, hy = half[t.variant]
                t.Bw[0] = [-(self.table_x - hx), self.table_x - hx]
                t.Bw[1] = [-(self.table_y - hy), self.table_y - hy]
            return templates

        original = StablePlacer.place_box
        with _patch(StablePlacer, "place_box", place_box):
            yield

    return apply()


def rotated_variant_labels() -> contextlib.AbstractContextManager:
    """The face a ``variant`` names is not the face that ends up on the surface."""

    @contextlib.contextmanager
    def apply():
        def place_box(self, lx, ly, lz, subject="object", min_margin_deg=0.0):
            templates = original(self, lx, ly, lz, subject=subject, min_margin_deg=min_margin_deg)
            labels = [t.variant for t in templates]
            # TSRTemplate is frozen, so the label has to move by rebuilding the record.
            # Assigning to it instead raises, which would make this mutant look detected
            # while proving only that the dataclass is frozen.
            return [replace(t, variant=label) for t, label in zip(templates, labels[1:] + labels[:1])]

        original = StablePlacer.place_box
        with _patch(StablePlacer, "place_box", place_box):
            yield

    return apply()


# --------------------------------------------------------------------------- #
# Code mutants: patch the real generator
# --------------------------------------------------------------------------- #


def squashed_projection() -> contextlib.AbstractContextManager:
    """#151: measure in-plane distances after dropping the normal's dominant axis."""

    def _plane_basis(n):
        i0 = int(np.argmax(np.abs(n)))
        axes = [i for i in range(3) if i != i0]
        basis = np.zeros((3, 2))
        basis[axes[0], 0] = 1.0
        basis[axes[1], 1] = 1.0
        return basis

    return _patch(sp, "_plane_basis", _plane_basis)


def absolute_containment_tolerance() -> contextlib.AbstractContextManager:
    """#153: decide containment on twice-area against an absolute constant, and treat an
    indeterminate result as "inside"."""

    @contextlib.contextmanager
    def apply():
        def _inward_distance(point, polygon):
            distance = original(point, polygon)
            shortest_edge = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1).min()
            return 1.0 if abs(distance * shortest_edge) < 1e-10 else distance

        original = sp._inward_distance
        with _patch(sp, "_inward_distance", _inward_distance):
            yield

    return apply()


def complementary_margin() -> contextlib.AbstractContextManager:
    """Report ``pi/2 - margin``: still an angle, still ordered, and still wrong."""

    @contextlib.contextmanager
    def apply():
        def stable_poses_mesh(vertices, com):
            for face in original(vertices, com):
                yield replace(face, stability_margin=np.pi / 2 - face.stability_margin)

        original = sp.stable_poses_mesh
        with _patch(spl, "stable_poses_mesh", stable_poses_mesh):
            yield

    return apply()


def shuffled_margins() -> contextlib.AbstractContextManager:
    """Each face reports another face's margin: the multiset is right, the mapping is
    not."""

    @contextlib.contextmanager
    def apply():
        def stable_poses_mesh(vertices, com):
            faces = list(original(vertices, com))
            margins = [f.stability_margin for f in faces]
            for face, margin in zip(faces, margins[1:] + margins[:1]):
                yield replace(face, stability_margin=margin)

        original = sp.stable_poses_mesh
        with _patch(spl, "stable_poses_mesh", stable_poses_mesh):
            yield

    return apply()


def unrelated_primitive_margin() -> contextlib.AbstractContextManager:
    """#156: a primitive margin built from the right inputs by the wrong function."""
    return _patch(spl, "_tipping_angle", lambda lever, height: float(np.arctan(lever)))


def rounded_normal_grouping() -> contextlib.AbstractContextManager:
    """#152: bucket an 8-decimal rounding of the hull normal, which splits a face whose
    facets straddle a rounding boundary."""

    def _group_coplanar_facets(hull, atol):
        from collections import defaultdict

        groups = defaultdict(list)
        for i in range(len(hull.simplices)):
            groups[tuple(np.round(hull.equations[i, :3], 8))].append(i)
        for key, indices in groups.items():
            n = np.array(key, dtype=float)
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            yield np.append(n / norm, hull.equations[indices[0], 3]), indices

    return _patch(sp, "_group_coplanar_facets", _group_coplanar_facets)


def over_merged_facets() -> contextlib.AbstractContextManager:
    """#152 from the other side: merge anything roughly parallel, collapsing a
    tessellated curved surface into a flat face that was never supplied."""
    return _patch(sp, "_NORMAL_ATOL", 0.5)


# --------------------------------------------------------------------------- #
# Provenance mutants: the record drifts from the geometry it describes (#160)
# --------------------------------------------------------------------------- #


def stale_footprint_in_record() -> contextlib.AbstractContextManager:
    """Record the *previous* variant's footprint while leaving ``Bw`` correct.

    The footprint radius is the #150 inset, unrecoverable from a template before the
    record existed. If the record may drift from the bound it claims to explain, the
    field is decorative."""

    @contextlib.contextmanager
    def apply():
        def _emit(self, method, primitive, subject, variants, min_margin_deg, scale, name, description):
            templates = original(self, method, primitive, subject, variants, min_margin_deg, scale, name, description)
            if len(templates) < 2:
                return templates
            shifted = [t.provenance.footprint_radius for t in templates]
            return [
                replace(t, provenance=replace(t.provenance, footprint_radius=r * 1.5))
                for t, r in zip(templates, shifted[1:] + shifted[:1])
            ]

        original = StablePlacer._emit
        with _patch(StablePlacer, "_emit", _emit):
            yield

    return apply()


def com_height_aliased_to_origin_height() -> contextlib.AbstractContextManager:
    """Record the mesh lever arm as the *origin* height.

    They coincide whenever the frame origin is the centre of mass, so this is invisible
    on a centred shape and only the ``corner-origin cube`` corpus case exposes it. That
    is precisely why ``com_height`` is a real field rather than an alias."""

    @contextlib.contextmanager
    def apply():
        def stable_poses_mesh(vertices, com):
            for face in original(vertices, com):
                yield replace(face, com_height=face.origin_height)

        original = sp.stable_poses_mesh
        with _patch(spl, "stable_poses_mesh", stable_poses_mesh):
            yield

    return apply()


def sphere_claims_stable() -> contextlib.AbstractContextManager:
    """Let the sphere call itself stably resting rather than neutrally balanced.

    The entire reason ``equilibrium`` is recorded: a rolling sphere and a knife-edge face
    both look like a zero tipping angle."""

    @contextlib.contextmanager
    def apply():
        def place_sphere(self, radius, subject="object", min_margin_deg=0.0):
            return [
                replace(t, provenance=replace(t.provenance, equilibrium="stable"))
                for t in original(self, radius, subject=subject, min_margin_deg=min_margin_deg)
            ]

        original = StablePlacer.place_sphere
        with _patch(StablePlacer, "place_sphere", place_sphere):
            yield

    return apply()


def wrong_face_normal_in_record() -> contextlib.AbstractContextManager:
    """Give each mesh face the *next* face's normal, leaving ``R`` untouched.

    ``R`` is built from the normal, so any check of the form ``R @ face_normal == -z``
    would still pass -- this proves the mesh check reads the record as an input and the
    pose as the truth."""

    @contextlib.contextmanager
    def apply():
        def stable_poses_mesh(vertices, com):
            faces = list(original(vertices, com))
            normals = [f.face_normal for f in faces]
            for face, normal in zip(faces, normals[1:] + normals[:1]):
                yield replace(face, face_normal=normal)

        original = sp.stable_poses_mesh
        with _patch(spl, "stable_poses_mesh", stable_poses_mesh):
            yield

    return apply()


def sampled_contact_trusted() -> contextlib.AbstractContextManager:
    """#201: ``place`` takes a candidate face's height and contact from the curved parts'
    sampled surface points instead of the parts themselves, so a cylinder lying between
    two rim samples rests on a sliver face, and the bottle gains rests where it rolls."""

    @contextlib.contextmanager
    def apply():
        def support(self, n):
            return float((self.surface() @ n).max())

        def contact(self, n, h, atol):
            points = self.surface()
            return points[points @ n >= h - atol]

        with contextlib.ExitStack() as stack:
            stack.enter_context(_patch(shp.Shape, "support", support))
            for part in (shp.Cylinder, shp.Sphere, shp.Capsule):
                stack.enter_context(_patch(part, "support", support))
                stack.enter_context(_patch(part, "contact", contact))
            yield

    return apply()


def footprint_about_part_centre() -> contextlib.AbstractContextManager:
    """#201: a part's reach is taken about its own centre rather than the object origin
    the region slides, so an object framed off its part hangs off the surface's edge."""

    @contextlib.contextmanager
    def apply():
        def box_reach(self, R):
            return shp._horizontal(self.corners - self.centre, R)

        def cylinder_reach(self, R):
            return self.radius

        with _patch(shp.Box, "reach", box_reach), _patch(shp.Cylinder, "reach", cylinder_reach):
            yield

    return apply()


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Mutant:
    name: str
    apply: Callable[[], contextlib.AbstractContextManager]
    recreates: str


MUTANTS: List[Mutant] = [
    Mutant("resting_height_in_tw_e", resting_height_in_tw_e, "#149 the sphere sinks through the surface"),
    Mutant("no_footprint_inset", no_footprint_inset, "#150 poses hang off the surface"),
    Mutant("per_axis_inset", per_axis_inset, "#150 the plausible wrong fix, which ignores yaw"),
    Mutant("rotated_variant_labels", rotated_variant_labels, "a variant names the wrong face"),
    Mutant("squashed_projection", squashed_projection, "#151 the margin is not the tipping angle"),
    Mutant("absolute_containment_tolerance", absolute_containment_tolerance, "#153 containment fails open"),
    Mutant("complementary_margin", complementary_margin, "the margin is an unverified number"),
    Mutant("shuffled_margins", shuffled_margins, "the margin is attached to the wrong face"),
    Mutant("unrelated_primitive_margin", unrelated_primitive_margin, "#156 a primitive margin nobody checks"),
    Mutant("rounded_normal_grouping", rounded_normal_grouping, "#152 a float32 face fragments"),
    Mutant("over_merged_facets", over_merged_facets, "#152 a tessellated curved surface collapses"),
    Mutant("stale_footprint_in_record", stale_footprint_in_record, "#160 the record drifts from the Bw inset"),
    Mutant(
        "com_height_aliased_to_origin_height",
        com_height_aliased_to_origin_height,
        "#160 the lever arm is confused with the positioning height",
    ),
    Mutant("sphere_claims_stable", sphere_claims_stable, "#160 neutral and knife-edge equilibria conflated"),
    Mutant("wrong_face_normal_in_record", wrong_face_normal_in_record, "#160 the recorded face normal is wrong"),
    Mutant("sampled_contact_trusted", sampled_contact_trusted, "#201 a sampled curve invents rests where it rolls"),
    Mutant("footprint_about_part_centre", footprint_about_part_centre, "#201 an off-centre frame hangs off the edge"),
]


def test_the_unmutated_generator_passes_the_corpus():
    """The negative control: every detection below is attributable to its mutation."""
    assert corpus_failures() == []


def test_the_corpus_covers_every_factory():
    from ._placement_matrix import FACTORIES

    assert {case.factory for case in corpus()} == set(FACTORIES)


@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda m: m.name)
def test_the_suite_detects(mutant: Mutant):
    with mutant.apply():
        failures = corpus_failures()
    assert failures, f"{mutant.name} went undetected: the suite no longer catches {mutant.recreates}"


def test_every_mutant_is_reverted():
    """A leaked patch would silently weaken every later test in the session."""
    assert sp._NORMAL_ATOL == 1e-6
    assert corpus_failures() == []
