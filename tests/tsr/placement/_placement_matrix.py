# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Shared Hypothesis matrix for every placement factory (issue #148).

The placement counterpart of ``tests/tsr/hands/_grasp_matrix.py``: it drives all public
``place_*`` factories through the independent :mod:`_placement_oracle` and provides

* a factory table (which oracle primitive a factory describes, and how to build it
  from the call's arguments);
* Hypothesis strategies over surface extents and object dimensions, spanning the
  regimes where the object fits comfortably, fits exactly, and cannot fit at all;
* **pure** check functions that return failure strings instead of asserting, so the
  property matrix and the mutation gate share one definition of "the suite detects
  this".

**A check may read a template's provenance as the subject under test; it may never use
it as the expected value.** The record is the generator's claim, so comparing a claim
against itself proves only internal consistency. Where a record is checked, the truth
comes from the oracle or from the pose -- see ``provenance_failures``.

Pose sampling comes from the grasp matrix (``bw_samples``): the midpoint, every free
dimension's extrema, all corners and interior points. A placement claim is about the
whole region, so checking the midpoint alone is what let #149 and #150 through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from hypothesis import strategies as st

from tsr import placement as shapes
from tsr.placement import StablePlacer
from tsr.placement._conformance import validate_builtin_placement_template
from tsr.placement_provenance import PlacementProvenance

from ..hands._grasp_matrix import budget, bw_samples, matrix_settings  # noqa: F401  (re-exported)
from ._placement_oracle import (
    Box,
    Capsule,
    Compound,
    Cylinder,
    Mesh,
    Sphere,
    Torus,
    centre_of_mass,
    certify,
    length_atol,
    tipping_angle,
)

# --------------------------------------------------------------------------- #
# Factory table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Factory:
    name: str
    #: Build the oracle primitive from the factory's keyword arguments.
    oracle: Callable[[Dict[str, Any]], Any]
    #: The ``primitive`` label this factory's provenance records must claim (#160), or a
    #: function of the call's keyword arguments for a factory that chooses one.
    primitive: Any = ""
    #: ``variant`` -> the outward object-frame normal of the face that must face down.
    #: Empty for factories whose variants are not face labels.
    #:
    #: Kept even though the template now *records* its face normal (#160), and
    #: deliberately so: this table is the independent statement of what a label promises.
    #: If ``face_label_failures`` read the record instead, nothing would connect the
    #: label to the geometry and the ``rotated_variant_labels`` mutant would survive.
    face_normals: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)


_AXES = {
    "-z": (0.0, 0.0, -1.0),
    "+z": (0.0, 0.0, 1.0),
    "-y": (0.0, -1.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+x": (1.0, 0.0, 0.0),
}

FACTORIES: Dict[str, Factory] = {
    "place_cylinder": Factory(
        "place_cylinder",
        lambda kw: Cylinder(kw["cylinder_radius"], kw["cylinder_height"]),
        "cylinder",
        {k: _AXES[k] for k in ("-z", "+z")},
    ),
    "place_box": Factory(
        "place_box",
        lambda kw: Box(kw["lx"], kw["ly"], kw["lz"]),
        "box",
        dict(_AXES),
    ),
    "place_sphere": Factory("place_sphere", lambda kw: Sphere(kw["radius"]), "sphere"),
    "place_torus": Factory(
        "place_torus",
        lambda kw: Torus(kw["major_radius"], kw["minor_radius"]),
        "torus",
        {k: _AXES[k] for k in ("-z", "+z")},
    ),
    "place_mesh": Factory(
        "place_mesh", lambda kw: Mesh(np.asarray(kw["vertices"], float), np.asarray(kw["com"], float)), "mesh"
    ),
    "place": Factory("place", lambda kw: oracle_shape(kw["shape"]), lambda kw: shape_primitive(kw["shape"])),
}


def oracle_shape(shape) -> Compound:
    """The oracle's body for a :class:`tsr.placement.Shape`, read from its public fields."""
    parts = []
    for p in shape.parts:
        if isinstance(p, shapes.Box):
            parts.append((Box(*p.size), p.pose))
        elif isinstance(p, shapes.Cylinder):
            parts.append((Cylinder(p.radius, p.height), p.pose))
        elif isinstance(p, shapes.Sphere):
            parts.append((Sphere(p.radius), p.pose))
        elif isinstance(p, shapes.Capsule):
            parts.append((Capsule(p.radius, p.length), p.pose))
        else:
            parts.append((Mesh(np.asarray(p.vertices), np.zeros(3)), np.eye(4)))
    return Compound(tuple(parts), np.asarray(shape.com))


def shape_primitive(shape) -> str:
    """The primitive ``place`` documents for a shape: a lone box, cylinder or sphere whose
    centre of mass is its centre is placed exactly, and anything else as a compound."""
    (part, *rest) = shape.parts
    single = not rest and isinstance(part, (shapes.Box, shapes.Cylinder, shapes.Sphere))
    if single and np.linalg.norm(shape.com - part.centre) <= length_atol(oracle_shape(shape).scale):
        return {shapes.Box: "box", shapes.Cylinder: "cylinder", shapes.Sphere: "sphere"}[type(part)]
    return "compound"


def public_factories() -> List[str]:
    """Every public placement factory, ``place`` and each ``place_*`` method, so the
    matrix cannot silently miss one."""
    return sorted(
        n for n in dir(StablePlacer) if (n == "place" or n.startswith("place_")) and callable(getattr(StablePlacer, n))
    )


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlacementCase:
    factory: str
    kwargs: Dict[str, Any]
    table: Tuple[float, float]
    #: How many templates this call must return. Set for the deterministic corpus and
    #: left ``None`` for generated cases. Without it a mutant that *empties* a result
    #: passes every per-pose check vacuously -- which is exactly what the #152 facet
    #: fragmentation does.
    expected: Optional[int] = None
    label: str = ""

    @property
    def prim(self):
        return FACTORIES[self.factory].oracle(self.kwargs)

    def placer(self) -> StablePlacer:
        return StablePlacer(table_x=self.table[0], table_y=self.table[1])

    def call(self) -> List[Any]:
        return getattr(self.placer(), self.factory)(**self.kwargs)

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        return self.label or f"{self.factory}({self.kwargs}) on table {self.table}"


def _at(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = x, y, z
    return T


def _turned(T: np.ndarray, axis: str, angle: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    R = np.eye(4)
    R[:3, :3] = Rotation.from_euler(axis, angle).as_matrix()
    return T @ R


#: Objects for ``place``, built in a frame with z up and the base at z = 0, as a model
#: file holds them; ``place_shape`` then moves the object frame anywhere.
SHAPE_KINDS = ("box", "cylinder", "sphere", "can", "bottle", "mug", "table", "hull", "capsule")


def place_shape(kind: str, scale: float, T: Optional[np.ndarray] = None):
    """A :class:`tsr.placement.Shape` of ``kind`` at ``scale``, in an object frame moved by
    ``T`` (the parts at ``inv(T)`` of where they were). ``box``, ``cylinder`` and
    ``sphere`` are one centred primitive; the rest are what collision models are made of.
    """
    T_new_old = np.linalg.inv(np.eye(4) if T is None else T)
    s = scale

    def at(pose):
        return T_new_old @ pose

    if kind == "box":
        parts = [shapes.Box((0.2 * s, 0.1 * s, 0.3 * s), pose=at(np.eye(4)))]
    elif kind == "cylinder":
        parts = [shapes.Cylinder(0.04 * s, 0.12 * s, pose=at(np.eye(4)))]
    elif kind == "sphere":
        parts = [shapes.Sphere(0.05 * s, pose=at(np.eye(4)))]
    elif kind == "can":  # its frame at the base's centre, as prl_assets' objects are
        parts = [shapes.Cylinder(0.033 * s, 0.123 * s, pose=at(_at(0, 0, 0.0615 * s)))]
    elif kind == "bottle":  # stacked cylinders: lying, it rolls
        parts = [
            shapes.Cylinder(0.04 * s, 0.15 * s, pose=at(_at(0, 0, 0.075 * s))),
            shapes.Cylinder(0.03 * s, 0.03 * s, pose=at(_at(0, 0, 0.165 * s))),
            shapes.Cylinder(0.015 * s, 0.04 * s, pose=at(_at(0, 0, 0.2 * s))),
        ]
    elif kind == "mug":  # a box body with a capsule handle out of its +x face
        parts = [
            shapes.Box((0.08 * s, 0.08 * s, 0.1 * s), pose=at(_at(0, 0, 0.05 * s))),
            shapes.Capsule(0.008 * s, 0.05 * s, pose=at(_turned(_at(0.06 * s, 0, 0.05 * s), "x", np.pi / 2))),
        ]
    elif kind == "table":  # a top on four inset legs
        parts = [shapes.Box((0.6 * s, 0.4 * s, 0.03 * s), pose=at(_at(0, 0, 0.735 * s)))]
        parts += [
            shapes.Box((0.04 * s, 0.04 * s, 0.72 * s), pose=at(_at(x * 0.25 * s, y * 0.15 * s, 0.36 * s)))
            for x in (-1, 1)
            for y in (-1, 1)
        ]
    elif kind == "hull":  # a convex hull part, as a decomposed scan is
        cube = _mesh_vertices("cube", 0.1 * s)[0] + [0, 0, 0.05 * s]
        parts = [shapes.Hull(cube, pose=at(np.eye(4)))]
    elif kind == "capsule":  # rolls lying and topples standing: no rest
        parts = [shapes.Capsule(0.02 * s, 0.1 * s, pose=at(np.eye(4)))]
    else:
        raise AssertionError(kind)
    return shapes.Shape(parts)


def _mesh_vertices(kind: str, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """A few small vertex clouds with an honestly computed centre of mass.

    ``corner-cube`` has its origin at a corner rather than the centroid, which is the
    case that distinguishes "the origin's resting height" from "half an extent".
    """
    if kind == "cube":
        v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * (scale / 2.0)
        return v, v.mean(axis=0)
    if kind == "corner-cube":
        v = np.array([[x, y, z] for x in (0.0, scale) for y in (0.0, scale) for z in (0.0, scale)])
        return v, v.mean(axis=0)
    if kind == "slab":
        v = np.array(
            [[sx * scale / 2, sy * scale / 3, sz * scale / 10] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )
        return v, v.mean(axis=0)
    if kind == "tetra":
        v = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]) * (scale / 2.0)
        return v, v.mean(axis=0)
    raise AssertionError(kind)


@st.composite
def cases(draw, factories: Optional[Tuple[str, ...]] = None, fit: str = "any") -> PlacementCase:
    """One placement call across scale regimes.

    ``fit`` selects the regime: ``"fits"`` keeps the object comfortably on the surface,
    ``"any"`` also draws objects that cannot fit, which is where the empty-feasible-set
    half of the contract lives.
    """
    factory = draw(st.sampled_from(sorted(factories or FACTORIES)))
    scale = draw(st.sampled_from([1e-3, 1e-2, 0.1, 1.0]))
    table = (draw(st.floats(0.5, 5.0)) * scale, draw(st.floats(0.5, 5.0)) * scale)
    if fit == "fits":
        # Keep the largest possible footprint (the half-diagonal of a cube of side
        # ``scale``, or R + r) inside the surface with room to spare.
        table = (max(table[0], 2.0 * scale), max(table[1], 2.0 * scale))

    def dim(lo: float, hi: float) -> float:
        return draw(st.floats(lo, hi)) * scale

    if factory == "place_cylinder":
        kwargs: Dict[str, Any] = {
            "cylinder_radius": dim(0.05, 0.5),
            "cylinder_height": dim(0.05, 1.0),
        }
    elif factory == "place_box":
        kwargs = {"lx": dim(0.05, 1.0), "ly": dim(0.05, 1.0), "lz": dim(0.05, 1.0)}
    elif factory == "place_sphere":
        kwargs = {"radius": dim(0.05, 0.5)}
    elif factory == "place_torus":
        major = dim(0.1, 0.5)
        kwargs = {"major_radius": major, "minor_radius": major * draw(st.floats(0.05, 0.9))}
    elif factory == "place":
        from .._hypothesis_strategies import transforms

        frame = draw(transforms()) if draw(st.booleans()) else None
        if frame is not None:
            frame[:3, 3] *= 0.1 * scale  # the object frame moved within the object's scale
        kwargs = {"shape": place_shape(draw(st.sampled_from(SHAPE_KINDS)), scale, frame)}
        if fit == "fits":  # the table's top spans 0.36 of its scale in footprint
            table = (max(table[0], 4.0 * scale), max(table[1], 4.0 * scale))
    else:
        vertices, com = _mesh_vertices(draw(st.sampled_from(["cube", "corner-cube", "slab", "tetra"])), scale)
        kwargs = {"vertices": vertices, "com": com}
    return PlacementCase(factory, kwargs, table)


# --------------------------------------------------------------------------- #
# Checks (pure: they return failures, they do not assert)
# --------------------------------------------------------------------------- #


def soundness_failures(case: PlacementCase, templates, *, table_pose=None, rng=None) -> List[str]:
    """Every pose the region admits must satisfy all three placement clauses.

    Poses are taken at the ``Bw`` midpoint, every free dimension's extrema, all corners
    and interior points -- the region, not a representative.
    """
    failures: List[str] = []
    prim = case.prim
    T = np.eye(4) if table_pose is None else table_pose
    T_inv = np.linalg.inv(T)
    for t in templates:
        tsr = t.instantiate(T)
        for xi in bw_samples(t, rng=rng):
            # Certify in the surface frame: the oracle's z = 0 is the surface plane.
            pose = T_inv @ tsr.to_transform(xi)
            w = certify(prim, pose, table=case.table)
            if not w.ok:
                failures.append(f"{case} [{t.variant}] xi={np.round(xi, 6).tolist()}: {w.failed}")
    return failures


def provenance_failures(case: PlacementCase, templates) -> List[str]:
    """Clause 5: the record describes the template, and the geometry bears it out (#160).

    Three kinds of check, in increasing strength:

    1. the record exists, is a placement record, and conforms to the built-in vocabulary
       (``tsr.placement._conformance``) -- a self-consistency check;
    2. ``primitive`` matches the factory that produced it;
    3. **oracle agreement**: ``com_height``, ``support_margin`` and ``footprint_radius``
       against the independently posed geometry. This is what makes the record
       falsifiable rather than decorative.

    The lengths are compared at ``rtol=2e-3`` for the same reason ``margin_failures``
    documents: the oracle samples a circular contact as a 72-gon and the footprint over a
    180-direction fan, biases of 9.5e-4 and 1.5e-4 -- an order below the tolerance and
    orders below the drift this exists to catch.
    """
    failures: List[str] = []
    prim = case.prim
    expected_primitive = FACTORIES[case.factory].primitive
    if callable(expected_primitive):
        expected_primitive = expected_primitive(case.kwargs)
    for t in templates:
        record = t.provenance
        if record is None:
            failures.append(f"{case} [{t.variant}]: no placement provenance (#160)")
            continue
        if not isinstance(record, PlacementProvenance):
            failures.append(f"{case} [{t.variant}]: provenance is {type(record).__name__}")
            continue
        try:
            validate_builtin_placement_template(t)
        except ValueError as e:
            failures.append(f"{case} [{t.variant}]: provenance {e}")
            continue
        if record.primitive != expected_primitive:
            failures.append(f"{case} [{t.variant}]: primitive {record.primitive!r} != {expected_primitive!r}")

        pose = t.instantiate(np.eye(4)).to_transform(np.zeros(6))
        witness = certify(prim, pose, table=case.table)
        com_z = centre_of_mass(prim, pose[:3, :3], pose[:3, 3])[2]
        if not np.isclose(record.com_height, com_z, rtol=0.0, atol=length_atol(prim.scale)):
            failures.append(f"{case} [{t.variant}]: com_height {record.com_height} but the COM rests at {com_z}")
        if not np.isclose(record.support_margin, witness.support_margin, rtol=2e-3, atol=1e-9):
            failures.append(
                f"{case} [{t.variant}]: support_margin {record.support_margin} "
                f"but the posed contact patch gives {witness.support_margin}"
            )
        if not np.isclose(record.footprint_radius, witness.footprint_radius, rtol=2e-3, atol=1e-9):
            failures.append(
                f"{case} [{t.variant}]: footprint_radius {record.footprint_radius} "
                f"but the posed object reaches {witness.footprint_radius}"
            )
        if case.factory == "place_mesh":
            failures += _mesh_face_normal_failures(case, t, record)
    return failures


def _mesh_face_normal_failures(case: PlacementCase, t, record) -> List[str]:
    """The recorded normal must select the vertices the pose puts on the surface.

    For a mesh there is no label table to compare against, and ``R`` is *built* from the
    normal, so ``R @ face_normal == -z`` holds however wrong the normal is -- it proves
    only that the generator was internally consistent. This instead feeds the record in
    as an **input** and takes the pose as the truth: the vertices extremal along the
    recorded normal in the object frame are exactly the vertices that end up on the
    surface. A wrong normal selects the wrong set.
    """
    vertices = np.asarray(case.kwargs["vertices"], float)
    atol = length_atol(float(np.ptp(vertices, axis=0).max()))
    along = vertices @ np.asarray(record.face_normal, float)
    claimed = set(np.flatnonzero(along >= along.max() - atol).tolist())

    pose = t.instantiate(np.eye(4)).to_transform(np.zeros(6))
    world_z = (vertices @ pose[:3, :3].T + pose[:3, 3])[:, 2]
    resting = set(np.flatnonzero(world_z <= world_z.min() + atol).tolist())

    if claimed != resting:
        return [
            f"{case} [{t.variant}]: face_normal {record.face_normal} selects vertices "
            f"{sorted(claimed)} but the pose rests on {sorted(resting)}"
        ]
    return []


def equilibrium_failures(cases_and_templates) -> List[str]:
    """Corpus-wide: exactly the zero-margin templates are the neutral ones (#160).

    Per-template this would be near-tautological -- the generator derives ``equilibrium``
    from ``support_margin``, and knife-edge mesh faces are filtered out before they are
    returned, so the sphere is the only neutral emitter. The biconditional over the whole
    corpus is what actually catches the confusion the field exists to resolve: a sphere's
    neutral rest and a vanishing tipping angle looking identical.
    """
    failures: List[str] = []
    for case, templates in cases_and_templates:
        for t in templates:
            if t.provenance is None or not isinstance(t.provenance, PlacementProvenance):
                continue
            neutral = t.provenance.equilibrium == "neutral"
            if neutral != (t.stability_margin == 0.0):
                failures.append(
                    f"{case} [{t.variant}]: equilibrium {t.provenance.equilibrium!r} "
                    f"with stability_margin {t.stability_margin}"
                )
    return failures


def margin_failures(case: PlacementCase, templates) -> List[str]:
    """Clause 4: a reported ``stability_margin`` is the pose's physical tipping angle.

    Compared on ``tan`` of the angle, which is the ratio the quantity is built from, at
    ``rtol = 2e-3``. The oracle samples a circular contact as a 72-gon, whose apothem is
    ``cos(pi/72) = 0.99905`` of the true radius, so a curved primitive's tipping angle
    carries a known conservative bias of 9.5e-4 -- an order of magnitude below the
    tolerance, and four orders below the 40% error this check exists to catch (#151).
    """
    failures: List[str] = []
    prim = case.prim
    for t in templates:
        if t.stability_margin is None:
            failures.append(f"{case} [{t.variant}]: no stability_margin (#156)")
            continue
        for xi in bw_samples(t):
            pose = t.instantiate(np.eye(4)).to_transform(xi)
            expected = tipping_angle(prim, pose)
            if not np.isclose(np.tan(t.stability_margin), np.tan(expected), rtol=2e-3, atol=1e-9):
                failures.append(
                    f"{case} [{t.variant}]: margin {np.degrees(t.stability_margin):.4f}° "
                    f"but the posed object tips at {np.degrees(expected):.4f}°"
                )
                break
    return failures


#: Factories that promise an order. ``place_mesh`` documents most-stable-first; the
#: primitives return their faces in a fixed labelled order instead, which is the more
#: useful contract when the caller cares which face is down.
ORDERED_FACTORIES = ("place_mesh",)


def _ordered(case: PlacementCase) -> bool:
    """``place`` orders its compound rests as ``place_mesh`` does."""
    if case.factory == "place":
        return FACTORIES["place"].primitive(case.kwargs) == "compound"
    return case.factory in ORDERED_FACTORIES


def ordering_failures(case: PlacementCase, templates) -> List[str]:
    """``place_mesh`` promises most-stable-first, judged by the reported angles."""
    if not _ordered(case):
        return []
    reported = [t.stability_margin for t in templates]
    if any(m is None for m in reported):
        return []
    if reported != sorted(reported, reverse=True):
        return [f"{case}: margins {np.degrees(reported).round(3).tolist()} are not descending"]
    return []


def face_label_failures(case: PlacementCase, templates) -> List[str]:
    """The face a ``variant`` names must be the face that ends up on the surface.

    Without this a relabelling of the variants is invisible, and "cereal box front-face
    down" is an unbacked promise.
    """
    normals = FACTORIES[case.factory].face_normals
    if not normals:
        return []
    failures: List[str] = []
    down = np.array([0.0, 0.0, -1.0])
    for t in templates:
        n = np.array(normals[t.variant])
        for xi in bw_samples(t):
            pose = t.instantiate(np.eye(4)).to_transform(xi)
            world_n = pose[:3, :3] @ n
            if not np.allclose(world_n, down, atol=1e-9):
                failures.append(f"{case} [{t.variant}]: face normal maps to {np.round(world_n, 6).tolist()}, not -z")
                break
    return failures


def equivariance_failures(case: PlacementCase, templates, T: np.ndarray) -> List[str]:
    """Instantiating on a moved surface must move the poses rigidly, and nothing else."""
    failures: List[str] = []
    for t in templates:
        at_identity = t.instantiate(np.eye(4))
        moved = t.instantiate(T)
        for xi in bw_samples(t):
            expected = T @ at_identity.to_transform(xi)
            got = moved.to_transform(xi)
            if not np.allclose(got, expected, atol=1e-9):
                failures.append(f"{case} [{t.variant}]: instantiate({T.round(3).tolist()}) is not rigid")
                break
    return failures


# --------------------------------------------------------------------------- #
# The deterministic corpus, and the one definition of "the suite detects this"
# --------------------------------------------------------------------------- #


def octahedron(scale: float = 0.10) -> np.ndarray:
    """Every face tilted with respect to the object frame, which is what makes a
    foreshortened in-plane projection visible (#151)."""
    return np.array([[s * scale if i == j else 0.0 for j in range(3)] for i in range(3) for s in (-1, 1)])


def tessellated_cylinder(radius: float, height: float, n: int) -> np.ndarray:
    """A curved surface cut into ``n`` flat sides, each a real resting face at
    ``180/n`` degrees. Merging them would invent a face the caller never supplied."""
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rim = np.stack([radius * np.cos(a), radius * np.sin(a)], axis=-1)
    return np.vstack([np.c_[rim, np.full(n, z)] for z in (-height / 2, height / 2)])


def float32_box(lx: float, ly: float, lz: float, seed: int = 0) -> np.ndarray:
    """A rotated box whose vertices round-tripped through float32 -- the ordinary
    result of loading a mesh, since STL, OBJ, glTF and MuJoCo all store float32."""
    from scipy.spatial.transform import Rotation

    Q = Rotation.random(random_state=seed).as_matrix()
    box = np.array([[sx * lx, sy * ly, sz * lz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) / 2.0
    return (box.astype(np.float32) @ Q.T.astype(np.float32)).astype(np.float64)


#: Its centroid projects outside the ``z = 0`` facet, so that facet is not a rest.
_OBTUSE_TETRA = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.15, 0.0], [0.9, 0.1, 1.0]])


#: An object frame off every part and turned off every axis, for ``place``'s corpus.
_MOVED = _turned(_at(0.03, -0.02, 0.05), "xyz", (0.4, -0.7, 1.1))


def corpus() -> List[PlacementCase]:
    """A fixed case per factory plus the shapes that make a specific defect visible.

    Deterministic on purpose: the gate has to attribute a detection to its mutation, so
    it cannot depend on what Hypothesis happened to draw.
    """
    cube = _mesh_vertices("cube", 0.10)
    corner = _mesh_vertices("corner-cube", 0.10)
    tetra = _mesh_vertices("tetra", 0.10)
    f32 = float32_box(0.20, 0.10, 0.30)
    return [
        PlacementCase(
            "place_cylinder", {"cylinder_radius": 0.04, "cylinder_height": 0.12}, (0.30, 0.20), 2, "cylinder"
        ),
        # A disc far larger than the surface: a valid request with no feasible pose.
        PlacementCase("place_cylinder", {"cylinder_radius": 2.0, "cylinder_height": 0.10}, (0.30, 0.20), 0, "4 m disc"),
        PlacementCase("place_box", {"lx": 0.20, "ly": 0.10, "lz": 0.30}, (0.60, 0.60), 6, "box"),
        PlacementCase("place_box", {"lx": 0.01, "ly": 0.01, "lz": 0.30}, (0.60, 0.60), 6, "needle"),
        PlacementCase("place_sphere", {"radius": 0.05}, (0.30, 0.20), 1, "sphere"),
        PlacementCase("place_torus", {"major_radius": 0.05, "minor_radius": 0.012}, (0.30, 0.20), 2, "torus"),
        PlacementCase("place_mesh", {"vertices": cube[0], "com": cube[1]}, (0.60, 0.60), 6, "mesh cube"),
        PlacementCase("place_mesh", {"vertices": corner[0], "com": corner[1]}, (0.60, 0.60), 6, "corner-origin cube"),
        PlacementCase("place_mesh", {"vertices": tetra[0], "com": tetra[1]}, (0.60, 0.60), 4, "tetrahedron"),
        PlacementCase("place_mesh", {"vertices": octahedron(), "com": np.zeros(3)}, (0.60, 0.60), 8, "octahedron"),
        PlacementCase("place_mesh", {"vertices": f32, "com": f32.mean(axis=0)}, (1.0, 1.0), 6, "float32 box"),
        PlacementCase(
            "place_mesh",
            {"vertices": tessellated_cylinder(0.04, 0.12, 24), "com": np.zeros(3)},
            (0.60, 0.60),
            26,  # two caps and twenty-four sides: the prism the caller supplied
            "24-gon cylinder",
        ),
        # An obtuse tetrahedron whose centroid projects outside its bottom facet, at a
        # scale where an absolute containment tolerance stops discriminating (#153).
        PlacementCase(
            "place_mesh",
            {"vertices": _OBTUSE_TETRA * 1e-5, "com": (_OBTUSE_TETRA * 1e-5).mean(axis=0)},
            (2e-5, 2e-5),
            3,  # four facets, and the one the centroid overhangs is not a rest
            "obtuse tetrahedron at 1e-5 m",
        ),
        # The centre of mass exactly above a support edge: a critical equilibrium, which
        # leaves only the face it is squarely above (#153).
        PlacementCase(
            "place_mesh",
            {"vertices": cube[0], "com": np.array([0.0, 0.05, 0.0])},
            (0.60, 0.60),
            1,
            "cube with its COM on a support edge",
        ),
        # ``place`` (#201). The counts are what each object physically does: the can and
        # the bottle stand on either end and roll lying down; the mug cannot rest on the
        # face its handle stands out from; the table stands, lies on its top, or rests on
        # one side's top edge and two feet; a capsule rolls or topples.
        *[
            PlacementCase("place", {"shape": place_shape(kind, 1.0, frame)}, (1.0, 1.0), count, f"place {kind}{label}")
            for kind, count in (
                ("box", 6),
                ("cylinder", 2),
                ("sphere", 1),
                ("can", 2),
                ("bottle", 2),
                ("mug", 5),
                ("table", 6),
                ("hull", 6),
                ("capsule", 0),
            )
            for frame, label in ((None, ""), (_MOVED, ", frame moved"))
        ],
    ]


def check_failures(case: PlacementCase, templates) -> List[str]:
    """Every check the placement suite makes about one call, as failure strings.

    The property matrix asserts this is empty; the mutation gate asserts a corrupted
    generator makes it non-empty. Sharing the definition is what makes "the suite
    detects this" mean the same thing in both places.
    """
    failures: List[str] = []
    if case.expected is not None and len(templates) != case.expected:
        failures.append(f"{case}: {len(templates)} templates, expected {case.expected}")
    rng = np.random.default_rng(0)
    failures += soundness_failures(case, templates, rng=rng)
    failures += margin_failures(case, templates)
    failures += face_label_failures(case, templates)
    failures += ordering_failures(case, templates)
    failures += provenance_failures(case, templates)
    return failures


def corpus_failures() -> List[str]:
    """Run every check over the whole corpus, plus the corpus-wide ones."""
    generated = [(case, case.call()) for case in corpus()]
    failures = [f for case, templates in generated for f in check_failures(case, templates)]
    return failures + equilibrium_failures(generated)
