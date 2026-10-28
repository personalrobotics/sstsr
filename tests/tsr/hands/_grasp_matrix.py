# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Shared Hypothesis matrix for every primitive grasp factory (issue #73).

This test-side library drives all public ``grasp_*`` factories (including the
combined entry points and the named grippers) through the independent #67 analytic
oracle. It provides:

* a factory table (primitive, argument vocabulary, oracle-primitive builder,
  constituent factories for combined entry points);
* :class:`GraspCase` and Hypothesis strategies over grippers, primitive dimensions
  across scale/aspect regimes, ``k``/``n_minor``/angle restrictions, preshape and
  clearance -- plus ``boundary_cases`` that sit exactly on, and one ulp either side
  of, every documented feasibility boundary;
* Bw pose sampling (midpoint, per-dimension extrema, corners, interior points);
* **pure** check functions that return a list of failure strings instead of
  asserting, so the property matrix and the mutation gate share one definition of
  "the suite detects this".

Clearance policy: the oracle checks the *explicit* requested clearance. When a case
leaves clearance at its default (``None``) the oracle checks ``0`` -- weaker but
sound, and it never re-derives the generator's per-factory default formula.
"""

from __future__ import annotations

import inspect
import itertools
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

from tsr.hands import FrankaHand, GripperBase, ParallelJawGripper, Robotiq2F85, Robotiq2F140
from tsr.hands._conformance import validate_builtin_provenance

from ._grasp_oracle import Box, Cylinder, Sphere, Torus, certify, pose_from

TWO_PI = 2.0 * math.pi
HALF_PI = math.pi / 2.0


def budget(n: int) -> int:
    """Example budget ``n`` scaled by ``TSR_MATRIX_SCALE`` (default 1; release gate 10)."""
    return max(1, int(round(n * float(os.environ.get("TSR_MATRIX_SCALE", "1")))))


def matrix_settings(n: int) -> settings:
    """Matrix settings: scaled budget, no deadline (stable across 3.10-3.14), and the
    reproduction blob printed on failure."""
    return settings(
        max_examples=budget(n),
        deadline=None,
        print_blob=True,
        suppress_health_check=[HealthCheck.too_slow],
    )


# --------------------------------------------------------------------------- #
# Factory table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Factory:
    name: str
    primitive: str
    parts: Tuple[str, ...] = ()  # constituent factories of a combined entry point
    angle_range: bool = False
    n_minor: bool = False
    minor_angle_range: bool = False
    # Opposite-side approach labels this factory emits together; the two sides must
    # mirror each other exactly. A factory that emits one side only (grasp_box_top)
    # declares none -- declared coverage is checked separately.
    mirror_pairs: Tuple[Tuple[str, str], ...] = ()
    # (mode, approach, finger_orientation) families the docs promise in a
    # comfortably feasible regime. A combined entry point inherits its parts'.
    families: Tuple[Tuple[str, str, str], ...] = ()

    def declared_families(self) -> set:
        own = set(self.families)
        for part in self.parts:
            own |= FACTORIES[part].declared_families()
        return own


FACTORIES: Dict[str, Factory] = {
    f.name: f
    for f in (
        Factory("grasp_sphere", "sphere", angle_range=True, families=(("surface", "radial", "diameter"),)),
        Factory("grasp_cylinder_side", "cylinder", angle_range=True, families=(("side", "radial", "tangential"),)),
        Factory("grasp_cylinder_top", "cylinder", angle_range=True, families=(("top", "+z", "diameter"),)),
        Factory("grasp_cylinder_bottom", "cylinder", angle_range=True, families=(("bottom", "-z", "diameter"),)),
        Factory(
            "grasp_cylinder",
            "cylinder",
            parts=("grasp_cylinder_side", "grasp_cylinder_top", "grasp_cylinder_bottom"),
            angle_range=True,
            mirror_pairs=(("+z", "-z"),),
        ),
        Factory("grasp_box_top", "box", families=(("top", "+z", "x"), ("top", "+z", "y"))),
        Factory("grasp_box_bottom", "box", families=(("bottom", "-z", "x"), ("bottom", "-z", "y"))),
        Factory(
            "grasp_box_face_x",
            "box",
            mirror_pairs=(("+x", "-x"),),
            families=tuple(("face", a, o) for a in ("+x", "-x") for o in ("y", "z")),
        ),
        Factory(
            "grasp_box_face_y",
            "box",
            mirror_pairs=(("+y", "-y"),),
            families=tuple(("face", a, o) for a in ("+y", "-y") for o in ("x", "z")),
        ),
        Factory(
            "grasp_box",
            "box",
            parts=("grasp_box_top", "grasp_box_bottom", "grasp_box_face_x", "grasp_box_face_y"),
            mirror_pairs=(("+z", "-z"), ("+x", "-x"), ("+y", "-y")),
        ),
        Factory(
            "grasp_torus_side",
            "torus",
            angle_range=True,
            n_minor=True,
            minor_angle_range=True,
            families=(("side", "tube", "tangential"),),
        ),
        Factory(
            "grasp_torus_span",
            "torus",
            mirror_pairs=(("+z", "-z"),),
            families=(("span", "+z", "diameter"), ("span", "-z", "diameter")),
        ),
        Factory(
            "grasp_torus",
            "torus",
            parts=("grasp_torus_side", "grasp_torus_span"),
            angle_range=True,
            mirror_pairs=(("+z", "-z"),),
            n_minor=True,
            minor_angle_range=True,
        ),
    )
}


_ACCEPTED: Dict[str, set] = {
    name: set(inspect.signature(getattr(ParallelJawGripper, name)).parameters) - {"self"} for name in FACTORIES
}


def public_grasp_factories() -> set:
    return {n for n in dir(GripperBase) if n.startswith("grasp_")}


def oracle_primitive(primitive: str, dims: Dict[str, float]):
    if primitive == "sphere":
        return Sphere(dims["object_radius"])
    if primitive == "cylinder":
        return Cylinder(dims["cylinder_radius"], dims["cylinder_height"])
    if primitive == "box":
        return Box(dims["box_x"], dims["box_y"], dims["box_z"])
    return Torus(dims["torus_radius"], dims["tube_radius"])


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #

NAMED_GRIPPERS = {"Robotiq2F85": Robotiq2F85, "Robotiq2F140": Robotiq2F140, "FrankaHand": FrankaHand}


@dataclass(frozen=True)
class GripperSpec:
    """A readable, rebuildable gripper description (shrinks to a legible repr)."""

    name: str  # "ParallelJawGripper" or a NAMED_GRIPPERS key
    finger_length: float
    max_aperture: float

    def build(self) -> ParallelJawGripper:
        if self.name in NAMED_GRIPPERS:
            return NAMED_GRIPPERS[self.name]()
        return ParallelJawGripper(finger_length=self.finger_length, max_aperture=self.max_aperture)

    def with_dims(self, finger_length: float, max_aperture: float) -> "GripperSpec":
        return GripperSpec("ParallelJawGripper", finger_length, max_aperture)


@dataclass(frozen=True)
class GraspCase:
    gripper: GripperSpec
    factory: str
    dims: Dict[str, float]
    options: Dict[str, Any] = field(default_factory=dict)  # k, clearance, preshape, ranges

    @property
    def spec(self) -> Factory:
        return FACTORIES[self.factory]

    @property
    def prim(self):
        return oracle_primitive(self.spec.primitive, self.dims)

    @property
    def oracle_clearance(self) -> float:
        c = self.options.get("clearance")
        return 0.0 if c is None else float(c)

    def kwargs(self) -> Dict[str, Any]:
        return {**self.dims, **self.options}

    def run(self, gripper: Optional[ParallelJawGripper] = None, **overrides) -> list:
        g = gripper if gripper is not None else self.gripper.build()
        return getattr(g, self.factory)(**{**self.kwargs(), **overrides})

    def replace(self, **changes) -> "GraspCase":
        d = dict(gripper=self.gripper, factory=self.factory, dims=self.dims, options=self.options)
        d.update(changes)
        return GraspCase(**d)


def _ratio(lo: float, hi: float) -> st.SearchStrategy:
    return st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False, width=64)


@st.composite
def gripper_specs(draw, named: bool = True) -> GripperSpec:
    """Generic grippers across scale regimes, or (optionally) the named hardware."""
    choices = ["ParallelJawGripper"] + (list(NAMED_GRIPPERS) if named else [])
    name = draw(st.sampled_from(choices))
    if name in NAMED_GRIPPERS:
        g = NAMED_GRIPPERS[name]()
        return GripperSpec(name, g.finger_length, g.max_aperture)
    scale = draw(st.sampled_from([0.01, 0.1, 1.0]))  # mm-, cm-, and m-scale hands
    return GripperSpec(name, scale * draw(_ratio(0.2, 2.0)), scale * draw(_ratio(0.2, 3.0)))


@st.composite
def primitive_dims(draw, primitive: str, L: float, A: float) -> Dict[str, float]:
    """Dimensions relative to reach ``L`` and aperture ``A`` (feasible and infeasible)."""
    if primitive == "sphere":
        return {"object_radius": A * draw(_ratio(0.02, 0.65))}
    if primitive == "cylinder":
        return {"cylinder_radius": A * draw(_ratio(0.02, 0.65)), "cylinder_height": L * draw(_ratio(0.1, 4.0))}
    if primitive == "box":
        dim = st.one_of(_ratio(0.02, 1.2).map(lambda u: A * u), _ratio(0.1, 4.0).map(lambda u: L * u))
        return {"box_x": draw(dim), "box_y": draw(dim), "box_z": draw(dim)}
    r = A * draw(_ratio(0.01, 0.6))
    return {"torus_radius": r * draw(_ratio(1.05, 3.0)) + A * draw(_ratio(0.0, 0.5)), "tube_radius": r}


@st.composite
def ordered_range(draw, lo: float, hi: float, full: Tuple[float, float]) -> Tuple[float, float]:
    """The default ``full`` range, or an ordered sub-interval (possibly a point)."""
    if draw(st.booleans()):
        return full
    a = draw(_ratio(lo, hi))
    b = draw(st.one_of(st.just(a), _ratio(lo, hi)))
    return (min(a, b), max(a, b))


@st.composite
def options_for(draw, spec: Factory, L: float, A: float, explicit_clearance: bool = False) -> Dict[str, Any]:
    opts: Dict[str, Any] = {"k": draw(st.integers(1, 5))}
    clearance = st.one_of(st.just(0.0), _ratio(0.0, 0.6).map(lambda u: L * u))
    opts["clearance"] = draw(clearance if explicit_clearance else st.one_of(st.none(), clearance))
    opts["preshape"] = draw(st.one_of(st.none(), _ratio(0.05, 1.1).map(lambda u: A * u)))
    if spec.angle_range:
        yaw = draw(ordered_range(-TWO_PI, TWO_PI, (0.0, TWO_PI)))
        if yaw[1] - yaw[0] <= TWO_PI:
            opts["angle_range"] = yaw
    if spec.n_minor:
        opts["n_minor"] = draw(st.integers(1, 5))
    if spec.minor_angle_range:
        opts["minor_angle_range"] = draw(ordered_range(-HALF_PI, HALF_PI, (-HALF_PI, HALF_PI)))
    return opts


@st.composite
def grasp_cases(draw, factories: Sequence[str] = tuple(FACTORIES), named: bool = True, explicit_clearance=False):
    g = draw(gripper_specs(named=named))
    spec = FACTORIES[draw(st.sampled_from(sorted(factories)))]
    dims = draw(primitive_dims(spec.primitive, g.finger_length, g.max_aperture))
    opts = draw(options_for(spec, g.finger_length, g.max_aperture, explicit_clearance))
    return GraspCase(g, spec.name, dims, opts)


# The straddle floor: each pad clears the object by 2*atol, so the realized margin
# floor is 4*atol (#107, #129). A default preshape realizes AT LEAST the requested
# clearance (#131), quantized by ulp(span), so it is not exactly equal to it.
_STRADDLE_FLOOR = 4.0


def boundary_clearances(spec: Factory, dims: Dict[str, float], L: float, A: float) -> List[float]:
    """Clearances that place a boundary **active for this factory** exactly (#126).

    Factory-aware, not merely primitive-aware: a cap band is irrelevant to
    ``grasp_cylinder_side`` and a height band is irrelevant to ``grasp_cylinder_top``,
    and selecting an irrelevant formula wastes the case. Every boundary in the
    contract is solved for ``c`` (never for the gripper), so the named hardware
    participates. See ``BOUNDARIES`` for the same boundaries pinned with slack
    dimensions and asserted as feasibility transitions.
    """
    name, primitive = spec.name, spec.primitive
    emits = {f for f in (spec.name, *spec.parts)}
    cands: List[float] = []

    if primitive in ("sphere", "cylinder", "torus"):
        r = _radius(dims)
        if primitive == "sphere":
            scale = r
        elif primitive == "cylinder":
            scale = max(r, dims["cylinder_height"])
        else:
            scale = dims["torus_radius"] + r
        radial = {"grasp_sphere", "grasp_cylinder_side", "grasp_torus_side"} & emits
        if radial:
            cands += [L - r, r]  # reach (L < 2r) and far surface (2r <= L)
        if {"grasp_cylinder_top", "grasp_cylinder_bottom"} & emits:
            cands.append(min(L, dims["cylinder_height"]) / 2)  # cap band (#122)
        if "grasp_cylinder_side" in emits:
            cands.append(dims["cylinder_height"] / 2)  # side height band (#124)
        if "grasp_torus_span" in emits:
            cands += [L - r, A - 2 * (dims["torus_radius"] + r)]
        cands += [A - 2 * r, _STRADDLE_FLOOR * (1e-9 + 1e-6 * scale)]  # aperture, straddle floor
    else:
        ds = {"box_x": dims["box_x"], "box_y": dims["box_y"], "box_z": dims["box_z"]}
        scale = max(ds.values())
        extents = (
            [ds[BOX_APPROACH_AXIS[f]] for f in emits if f in BOX_APPROACH_AXIS]
            if emits & set(BOX_APPROACH_AXIS)
            else list(ds.values())
        )
        cands += [min(L, e) / 2 for e in extents]  # approach bands
        cands += [d / 2 for d in ds.values()]  # slide bands
        cands += [A - d for d in ds.values()]  # aperture, per span dimension
        cands.append(_STRADDLE_FLOOR * (1e-9 + 1e-6 * scale))  # straddle floor

    del name
    return [c for c in cands if c >= 0.0 and math.isfinite(c)]


def _nudge(x: float, ulps: int) -> float:
    for _ in range(abs(ulps)):
        x = math.nextafter(x, math.inf if ulps > 0 else -math.inf)
    return x


@st.composite
def boundary_cases(draw, factories: Sequence[str] = tuple(FACTORIES), named: bool = True):
    """A case whose clearance sits on a feasibility boundary, or one ulp either side."""
    case = draw(grasp_cases(factories, named=named))
    L, A = case.gripper.finger_length, case.gripper.max_aperture
    cands = boundary_clearances(case.spec, case.dims, L, A)
    if not cands:
        return case.replace(options={**case.options, "preshape": None, "clearance": 0.0})
    c = _nudge(draw(st.sampled_from(cands)), draw(st.sampled_from([-1, 0, 1])))
    return case.replace(options={**case.options, "preshape": None, "clearance": max(c, 0.0)})


def any_cases(factories: Sequence[str] = tuple(FACTORIES), named: bool = True) -> st.SearchStrategy:
    return st.one_of(grasp_cases(factories, named=named), boundary_cases(factories, named=named))


# --------------------------------------------------------------------------- #
# Pose sampling and checks
# --------------------------------------------------------------------------- #


def bw_samples(t, rng: Optional[np.random.Generator] = None, n_interior: int = 2) -> List[np.ndarray]:
    """Midpoint, every free-dimension extremum, all corners, and interior points."""
    lo, hi = t.Bw[:, 0], t.Bw[:, 1]
    mid = (lo + hi) / 2.0
    free = [i for i in range(6) if hi[i] > lo[i]]
    samples = [mid]
    for i in free:
        for v in (lo[i], hi[i]):
            s = mid.copy()
            s[i] = v
            samples.append(s)
    for corner in itertools.product(*[(lo[i], hi[i]) for i in free]):
        s = mid.copy()
        s[free] = corner
        samples.append(s)
    if rng is not None:
        for _ in range(n_interior):
            samples.append(lo + (hi - lo) * rng.random(6))
    return samples


def _label(t) -> str:
    p = t.provenance
    return f"{p.mode}/{p.approach}/{p.finger_orientation}/{p.symmetry}#{p.depth_index}"


def certify_pose(case: GraspCase, t, pose, *, prim=None, approach: Optional[str] = None):
    p = t.provenance
    return certify(
        prim if prim is not None else case.prim,
        pose,
        finger_length=case.gripper.finger_length,
        max_aperture=case.gripper.max_aperture,
        preshape=float(t.preshape[0]),
        clearance=case.oracle_clearance,
        mode=p.mode,
        approach=approach if approach is not None else p.approach,
        finger_orientation=p.finger_orientation,
    )


def soundness_failures(case: GraspCase, templates, rng: Optional[np.random.Generator] = None) -> List[str]:
    """Every sampled pose of every template has an oracle witness (clauses 1-8)."""
    out: List[str] = []
    for t in templates:
        try:
            validate_builtin_provenance(t.provenance)
        except ValueError as e:
            out.append(f"{_label(t)}: provenance {e}")
            continue
        if t.provenance.primitive != case.spec.primitive:
            out.append(f"{_label(t)}: primitive {t.provenance.primitive!r} != {case.spec.primitive!r}")
            continue
        tsr = t.instantiate(np.eye(4))
        for xi in bw_samples(t, rng):
            w = certify_pose(case, t, tsr.to_transform(xi))
            if not w.ok:
                out.append(f"{_label(t)} at {np.round(xi, 6).tolist()}: {w.failed}")
                break
    return out


def run_checks(case: GraspCase, checks: Sequence[Callable[[GraspCase, list], List[str]]], **run_kw) -> List[str]:
    """Run ``case`` (optionally with overrides) and collect all check failures."""
    templates = case.run(**run_kw)
    out: List[str] = []
    for check in checks:
        out.extend(check(case, templates))
    return out


# --------------------------------------------------------------------------- #
# Equivariance, symmetry, serialization
# --------------------------------------------------------------------------- #


def equivariance_failures(case: GraspCase, templates, T: np.ndarray) -> List[str]:
    """Instantiating at ``T`` maps every pose by ``T`` and preserves the witness."""
    out: List[str] = []
    Tinv = np.linalg.inv(T)
    for t in templates:
        at_identity, at_T = t.instantiate(np.eye(4)), t.instantiate(T)
        for xi in bw_samples(t):
            moved, base = at_T.to_transform(xi), at_identity.to_transform(xi)
            if not np.allclose(moved, T @ base, atol=1e-9, rtol=0.0):
                out.append(f"{_label(t)}: instantiate(T) != T @ instantiate(I)")
                break
            w = certify_pose(case, t, Tinv @ moved)  # back to the object frame
            if not w.ok:
                out.append(f"{_label(t)}: witness lost under reference transform: {w.failed}")
                break
    return out


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    R = np.eye(4)
    R[:2, :2] = [[c, -s], [s, c]]
    return R


# One label map per reflection: a reflection renames only the sides it swaps, so a
# z-mirror must leave a ±x face label alone.
_MIRROR_LABELS = {
    2: {"+z": "-z", "-z": "+z", "top": "bottom", "bottom": "top"},
    0: {"+x": "-x", "-x": "+x"},
    1: {"+y": "-y", "-y": "+y"},
}


def symmetry_maps(case: GraspCase, theta: float) -> List[Tuple[str, Callable, Dict[str, str]]]:
    """``(name, pose_map, label_map)`` for each symmetry of the case's primitive.

    Rotations about the object axis are proper maps applied on the left. The
    equatorial mirror is improper, so the mapped pose is rebuilt from mapped axes
    with :func:`pose_from` (mirroring a grasp keeps it right-handed). Labels that
    name a side (``mode``/``approach``) are mapped with it.
    """
    prim, dims = case.spec.primitive, case.dims
    maps: List[Tuple[str, Callable, Dict[str, str]]] = []

    # Rotation about the object z-axis: a symmetry of sphere/cylinder/torus; for a
    # (generally non-square) box only the half-turn maps the solid onto itself.
    angle = math.pi if prim == "box" else theta
    R = _rot_z(angle)
    # A half-turn swaps the ±x and ±y faces; a free rotation of a body of revolution
    # leaves the radial / ±z labels alone.
    rot_labels = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y"} if prim == "box" else {}
    maps.append((f"rot_z({angle:.3f})", lambda pose, R=R: R @ pose, rot_labels))

    # Equatorial mirror: z -> mirror_z - z (sphere/torus are centred at 0).
    mirror_z = {"cylinder": dims.get("cylinder_height", 0.0), "box": dims.get("box_z", 0.0)}.get(prim, 0.0)

    def mirror(pose, mirror_z=mirror_z):
        p, a, y = pose[:3, 3].copy(), pose[:3, 2].copy(), pose[:3, 1].copy()
        p[2] = mirror_z - p[2]
        a[2], y[2] = -a[2], -y[2]
        return pose_from(p, a, y)

    maps.append(("mirror_z", mirror, _MIRROR_LABELS[2]))

    # Every primitive is centred in x and y, so reflecting either is a symmetry too.
    for axis, name in ((0, "mirror_x"), (1, "mirror_y")):

        def reflect(pose, axis=axis):
            p, a, y = pose[:3, 3].copy(), pose[:3, 2].copy(), pose[:3, 1].copy()
            p[axis], a[axis], y[axis] = -p[axis], -a[axis], -y[axis]
            return pose_from(p, a, y)

        maps.append((name, reflect, _MIRROR_LABELS[axis]))
    return maps


def symmetry_failures(case: GraspCase, templates, theta: float = 0.7) -> List[str]:
    """Mapping a pose by a symmetry of the primitive preserves the witness.

    Side-naming labels travel with the map (a mirrored top grasp is a bottom grasp),
    so the mapped pose is certified against the mapped ``mode``/``approach``.
    """
    out: List[str] = []
    for name, pose_map, label_map in symmetry_maps(case, theta):
        for t in templates:
            p = t.provenance
            tsr = t.instantiate(np.eye(4))
            for xi in bw_samples(t):
                w = certify(
                    case.prim,
                    pose_map(tsr.to_transform(xi)),
                    finger_length=case.gripper.finger_length,
                    max_aperture=case.gripper.max_aperture,
                    preshape=float(t.preshape[0]),
                    clearance=case.oracle_clearance,
                    mode=label_map.get(p.mode, p.mode),
                    approach=label_map.get(p.approach, p.approach),
                    finger_orientation=p.finger_orientation,
                )
                if not w.ok:
                    out.append(f"{_label(t)} under {name}: {w.failed}")
                    break
    return out


def mirror_family_failures(case: GraspCase, templates) -> List[str]:
    """Each declared opposite-side pair is present on both sides with equal depths.

    Only the pairs a factory declares are checked: ``grasp_box_top`` emits the +z
    family alone by design, while ``grasp_box`` must emit both sides of all three
    axes. Whether a factory emits every family it declares is checked separately.
    """
    depths: Dict[Tuple[str, str, str], List[float]] = {}
    for t in templates:
        p = t.provenance
        depths.setdefault((p.mode, p.approach, p.finger_orientation), []).append(round(p.depth, 12))
    out: List[str] = []
    for lo, hi in case.spec.mirror_pairs:
        labels = _MIRROR_LABELS["xyz".index(lo[1])]
        sides = {side: {(m, o): d for (m, a, o), d in depths.items() if a == side} for side in (lo, hi)}
        for (mode, orientation), ds in sides[lo].items():
            mirrored = (labels.get(mode, mode), orientation)
            other = sides[hi].get(mirrored)
            if other is None:
                out.append(f"{mode}/{lo}/{orientation} has no {hi} mirror family {mirrored}")
            elif sorted(other) != sorted(ds):
                out.append(f"{mode}/{lo}/{orientation} depths {sorted(ds)} != {hi} {sorted(other)}")
        for mode, orientation in sides[hi]:
            mirrored = (labels.get(mode, mode), orientation)
            if mirrored not in sides[lo]:
                out.append(f"{mode}/{hi}/{orientation} has no {lo} mirror family {mirrored}")
    return out


def serialization_failures(case: GraspCase, templates) -> List[str]:
    """dict/JSON/YAML round-trips preserve the poses and the witness."""
    out: List[str] = []
    for t in templates:
        for fmt, restore in (
            ("dict", lambda t: type(t).from_dict(t.to_dict())),
            ("json", lambda t: type(t).from_json(t.to_json())),
            ("yaml", lambda t: type(t).from_yaml(t.to_yaml())),
        ):
            back = restore(t)
            if back.provenance != t.provenance:
                out.append(f"{_label(t)}: {fmt} provenance changed")
                continue
            tsr_a, tsr_b = t.instantiate(np.eye(4)), back.instantiate(np.eye(4))
            for xi in bw_samples(t):
                pose_a, pose_b = tsr_a.to_transform(xi), tsr_b.to_transform(xi)
                if not np.array_equal(pose_a, pose_b):
                    out.append(f"{_label(t)}: {fmt} round-trip changed the pose")
                    break
                if not certify_pose(case, back, pose_b).ok:
                    out.append(f"{_label(t)}: {fmt} round-trip lost the witness")
                    break
    return out


# --------------------------------------------------------------------------- #
# Arguments, coverage, uniqueness, monotonicity
# --------------------------------------------------------------------------- #


def family_key(p) -> Tuple:
    """The depth family key of #120: every provenance field except the depth fields."""
    return (p.primitive, p.mode, p.approach, p.finger_orientation, p.symmetry, tuple(sorted(p.metadata.items())))


def structured_key(p) -> Tuple:
    return family_key(p) + (p.depth_index,)


INVALID_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "nonpositive_dimension": {"__dim__": 0.0},
    "negative_dimension": {"__dim__": -0.05},
    "nonfinite_dimension": {"__dim__": float("inf")},
    "nan_dimension": {"__dim__": float("nan")},
    "zero_k": {"k": 0},
    "negative_k": {"k": -2},
    "float_k": {"k": 2.5},
    "bool_k": {"k": True},
    "negative_clearance": {"clearance": -0.01},
    "nan_clearance": {"clearance": float("nan")},
    "nonfinite_preshape": {"preshape": float("inf")},
    "nonpositive_preshape": {"preshape": 0.0},
}
"""One corrupted argument per case; ``__dim__`` corrupts the first primitive dimension."""

RANGE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "unordered_angle_range": {"angle_range": (1.0, 0.0)},
    "nonfinite_angle_range": {"angle_range": (0.0, float("inf"))},
    "zero_n_minor": {"n_minor": 0},
    "outside_minor_range": {"minor_angle_range": (-2.0, 2.0)},
}


def invalid_overrides_for(spec: Factory) -> Dict[str, Dict[str, Any]]:
    out = dict(INVALID_OVERRIDES)
    for name, override in RANGE_OVERRIDES.items():
        arg = next(iter(override))
        if getattr(spec, arg, False):
            out[name] = override
    if spec.primitive == "torus":
        out["tube_not_smaller_than_major"] = {"tube_radius": 1.0, "torus_radius": 1.0}
    return out


def apply_override(case: GraspCase, override: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve ``__dim__`` against the case's first dimension and return call kwargs."""
    kwargs = dict(case.kwargs())
    for arg, value in override.items():
        kwargs[sorted(case.dims)[0] if arg == "__dim__" else arg] = value
    return kwargs


def oversize_dims(primitive: str, A: float) -> Dict[str, float]:
    """Dimensions no jaw opening of aperture ``A`` can straddle in any mode."""
    if primitive == "sphere":
        return {"object_radius": A}
    if primitive == "cylinder":
        return {"cylinder_radius": A, "cylinder_height": 4 * A}
    if primitive == "box":
        return {"box_x": 2 * A, "box_y": 2 * A, "box_z": 2 * A}
    return {"torus_radius": 4 * A, "tube_radius": A}


def uniqueness_failures(case: GraspCase, templates) -> List[str]:
    """No two templates in one result share the full structured key."""
    seen: Dict[Tuple, int] = {}
    out: List[str] = []
    for t in templates:
        key = structured_key(t.provenance)
        seen[key] = seen.get(key, 0) + 1
    for key, n in seen.items():
        if n > 1:
            out.append(f"{n} templates share the structured key {key}")
    return out


def combined_failures(case: GraspCase, templates) -> List[str]:
    """A combined entry point equals the concatenation of its parts."""
    if not case.spec.parts:
        return []
    gripper = case.gripper.build()
    parts: List[Any] = []
    for part in case.spec.parts:
        kwargs = {k: v for k, v in case.kwargs().items() if k in _ACCEPTED[part]}
        parts.extend(getattr(gripper, part)(**kwargs))
    if len(parts) != len(templates):
        return [f"{case.factory}: {len(templates)} templates, parts give {len(parts)}"]
    # Matched by structured key, not position: the emission ORDER of a combined entry
    # point is not part of the contract (grasp_box emits the faces first). Once matched,
    # the WHOLE template must agree -- geometry, provenance, preshape and the public
    # semantic/display fields (task, subject, reference, name, description, variant,
    # stability_margin) -- so ``to_dict`` is compared rather than a partial field list
    # that silently omits what it does not name (#127).
    by_key = {structured_key(t.provenance): t for t in parts}
    out: List[str] = []
    for a in templates:
        b = by_key.get(structured_key(a.provenance))
        if b is None:
            out.append(f"{_label(a)}: no matching template from the part factories")
            continue
        da, db = a.to_dict(), b.to_dict()
        if da != db:
            differing = sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))
            out.append(f"{_label(a)}: differs from the part factory in {differing}")
    return out


def coverage_failures(case: GraspCase, templates) -> List[str]:
    """Every declared family is present (comfortably feasible regime only)."""
    emitted = {(p.provenance.mode, p.provenance.approach, p.provenance.finger_orientation) for p in templates}
    missing = case.spec.declared_families() - emitted
    extra = emitted - case.spec.declared_families()
    out = []
    if missing:
        out.append(f"{case.factory}: declared families missing {sorted(missing)}")
    if extra:
        out.append(f"{case.factory}: undeclared families emitted {sorted(extra)}")
    return out


def monotonicity_failures(case: GraspCase, attr: str, factor: float = 2.0) -> List[str]:
    """Growing reach or aperture never removes a family."""
    spec = case.gripper
    dims = {"finger_length": spec.finger_length, "max_aperture": spec.max_aperture}
    before = {family_key(t.provenance) for t in case.run(gripper=spec.build())}
    dims[attr] *= factor
    bigger = ParallelJawGripper(**dims)
    after = {family_key(t.provenance) for t in case.run(gripper=bigger)}
    lost = before - after
    return [f"{attr} x{factor} lost families {sorted(lost)}"] if lost else []


# --------------------------------------------------------------------------- #
# Mutation gate: corpus and mutants (#73)
# --------------------------------------------------------------------------- #

CORPUS_GRIPPERS = (
    GripperSpec("ParallelJawGripper", 0.08, 0.20),
    # Short fingers relative to the corpus objects (L < 2r for the sphere and the
    # cylinder), so the radial deep limit min(L, 2r) - c is the BINDING one: pushing
    # past it puts the palm inside the solid. With long fingers that limit is merely
    # conservative and exceeding it is still a sound grasp, which no check may reject.
    GripperSpec("ParallelJawGripper", 0.03, 0.20),
    GripperSpec("Robotiq2F85", 0.059, 0.085),
)

_CORPUS_DIMS = {
    "sphere": {"object_radius": 0.02},
    "cylinder": {"cylinder_radius": 0.02, "cylinder_height": 0.12},
    "box": {"box_x": 0.05, "box_y": 0.04, "box_z": 0.06},
    # A tube thick enough that the 2F-85's fingertips, closed across it, stay inside its far
    # side (#193), on a ring small enough that its outer diameter still fits the 2F-85.
    "torus": {"torus_radius": 0.022, "tube_radius": 0.018},
}


def corpus() -> List[GraspCase]:
    """A deterministic, comfortably feasible case per factory per corpus gripper."""
    cases = []
    for spec in CORPUS_GRIPPERS:
        for factory in sorted(FACTORIES):
            options: Dict[str, Any] = {"k": 3, "clearance": 0.004}
            cases.append(GraspCase(spec, factory, dict(_CORPUS_DIMS[FACTORIES[factory].primitive]), options))
    return cases


def _rot(axis: int, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    R = np.eye(4)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


def mutate_approach_sign(case: GraspCase, templates):
    """Reverse the approach axis: the hand faces away from the object."""
    out = []
    for t in templates:
        # Half-turn about y_EE: z_EE (approach) and x_EE flip, handedness preserved.
        out.append(_respawn(t, Tw_e=t.Tw_e @ _rot(1, math.pi)))
    return out


def mutate_face_origin(case: GraspCase, templates):
    """Shift the TSR origin by half the object extent along the object z-axis."""
    extent = {
        "sphere": case.dims.get("object_radius", 0.0),
        "cylinder": case.dims.get("cylinder_height", 0.0) / 2,
        "box": case.dims.get("box_z", 0.0) / 2,
        "torus": case.dims.get("tube_radius", 0.0),
    }[case.spec.primitive]
    out = []
    for t in templates:
        T = t.T_ref_tsr.copy()
        T[2, 3] += extent
        out.append(_respawn(t, T_ref_tsr=T))
    return out


def mutate_axis_mapping(case: GraspCase, templates):
    """Permute the TSR frame's axes (a quarter-turn about the object x-axis)."""
    return [_respawn(t, T_ref_tsr=_rot(0, math.pi / 2) @ t.T_ref_tsr) for t in templates]


def mutate_object_extent(case: GraspCase, templates):
    """Generate for a HALF-SIZE object, certify against the real one.

    The shrink direction is the corrupting one: a grasp planned for a *larger*
    object can still be a soundly executed grasp of the real, smaller one (the hand
    simply stands off further), so inflating the extent is not by itself unsound.
    """
    smaller = case.replace(dims={k: v * 0.5 for k, v in case.dims.items()})
    return smaller.run()


def mutate_clearance_term(case: GraspCase, templates):
    """Generate with clearance 0, certify against the requested clearance."""
    return case.run(clearance=0.0)


def mutate_reach(case: GraspCase, templates):
    """Generate with 1.5x finger length, certify with the real gripper."""
    spec = case.gripper
    longer = ParallelJawGripper(finger_length=spec.finger_length * 1.5, max_aperture=spec.max_aperture)
    return case.run(gripper=longer)


BLACK_BOX_MUTANTS = {
    "approach_sign": mutate_approach_sign,
    "face_origin": mutate_face_origin,
    "axis_mapping": mutate_axis_mapping,
    "object_extent": mutate_object_extent,
    "clearance_term": mutate_clearance_term,
    "reach": mutate_reach,
}


def _respawn(t, **changes):
    """A copy of template ``t`` with geometry replaced (provenance is kept)."""
    fields = dict(
        T_ref_tsr=t.T_ref_tsr,
        Tw_e=t.Tw_e,
        Bw=t.Bw,
        task=t.task,
        subject=t.subject,
        reference=t.reference,
        name=t.name,
        description=t.description,
        variant=t.variant,
        preshape=t.preshape,
        stability_margin=t.stability_margin,
        provenance=t.provenance,
    )
    fields.update(changes)
    return type(t)(**fields)


def detected(case: GraspCase, templates) -> bool:
    """Would the matrix reject this (possibly mutated) result?"""
    if not templates:
        return True  # a mutant that empties a feasible family is itself a detection
    return bool(
        soundness_failures(case, templates)
        or uniqueness_failures(case, templates)
        or mirror_family_failures(case, templates)
        or symmetry_failures(case, templates)
    )


# --------------------------------------------------------------------------- #
# Factory-aware feasibility boundaries (#126)
# --------------------------------------------------------------------------- #
#
# ``boundary_cases`` above biases a random case toward a boundary; that is a
# soundness aid only, and an EMPTY result satisfies it vacuously. The descriptors
# below instead pin one boundary per factory family and assert the documented
# feasibility TRANSITION across it: the infeasible side is empty, while the feasible
# side -- and the boundary value itself, when the interval is closed -- is non-empty
# AND oracle-sound. That is what catches a closed interval silently becoming open
# (the #124 bug class), which every other check passes vacuously.
#
# All dimensions are dyadic multiples of a scale, so the boundary identities
# (``h - h/2 == h/2``, ``L - (L - r) == r``) hold EXACTLY in binary and the
# ``nextafter`` neighbours really do straddle the comparison the generator makes.


@dataclass(frozen=True)
class Boundary:
    """One documented feasibility boundary, for the factories where it is active."""

    name: str
    factories: Tuple[str, ...]
    setup: Callable[[float, str], Tuple[float, float, Dict[str, float]]]  # (scale, factory) -> L, A, dims
    clearance: Callable[[float, float, Dict[str, float], str], float]  # (L, A, dims, factory) -> exact c
    family: Callable[[Any], bool]  # the provenance records this boundary governs
    # Which argument carries the boundary. "preshape" is used where the rule is only
    # ulp-exact through an explicit preshape: the straddle margin is
    # ``preshape - span``, exact by Sterbenz, whereas the DEFAULT preshape recovers it
    # from ``span + c``, whose resolution is ulp(span) -- so a single ulp of the
    # clearance is not observable there (#129).
    param: str = "clearance"
    fixed_clearance: Optional[Callable[[float, float, Dict[str, float], str], float]] = None
    equality_feasible: bool = True  # is the boundary value itself feasible?
    feasible_below: bool = True  # which side of the boundary is feasible
    relative_step: bool = False  # step by a relative epsilon instead of one ulp
    # Certify the boundary value itself? False only where the ORACLE's own tolerance
    # sits on the same value, so certification there is ambiguous by one ulp; presence
    # is still asserted.
    certify_exact: bool = True

    def neighbours(self, c: float) -> Tuple[float, float]:
        """``(below, above)`` the boundary value."""
        if self.relative_step:
            # The generator compares ``span + c`` against ``span + 2*atol``; one ulp of
            # ``c`` vanishes inside that sum, so step by a fraction of the value.
            return c * (1.0 - 1e-3), c * (1.0 + 1e-3)
        return _nudge(c, -1), _nudge(c, 1)


CAP_FACTORIES = ("grasp_cylinder_top", "grasp_cylinder_bottom", "grasp_cylinder")
RADIAL_FACTORIES = ("grasp_sphere", "grasp_cylinder_side", "grasp_cylinder", "grasp_torus_side", "grasp_torus")
BOX_FACE_FACTORIES = ("grasp_box_top", "grasp_box_bottom", "grasp_box_face_x", "grasp_box_face_y", "grasp_box")

# Which object axis a box factory approaches along (the combined entry point is
# pinned on its ±z faces).
BOX_APPROACH_AXIS = {
    "grasp_box_top": "box_z",
    "grasp_box_bottom": "box_z",
    "grasp_box_face_x": "box_x",
    "grasp_box_face_y": "box_y",
    "grasp_box": "box_z",
}


def _radius(dims: Dict[str, float]) -> float:
    for key in ("object_radius", "cylinder_radius", "tube_radius"):
        if key in dims:
            return dims[key]
    raise KeyError(dims)


def _radial_setup(scale, factory, l_mult, r_mult):
    """A radial-family case: sphere, tall cylinder (height slack), or torus tube."""
    L, r = l_mult * scale, r_mult * scale
    if "sphere" in factory:
        return L, 0.5 * scale, {"object_radius": r}
    if "torus" in factory:
        return L, 0.75 * scale, {"torus_radius": 8 * r, "tube_radius": r}
    return L, 0.5 * scale, {"cylinder_radius": r, "cylinder_height": 0.5 * scale}


def _box_setup(scale, factory):
    """Box dims whose APPROACH extent is the binding one; slides stay slack."""
    dims = {
        "box_x": (0.125, 0.25, 0.375),
        "box_y": (0.25, 0.125, 0.375),
        "box_z": (0.25, 0.375, 0.125),
    }[BOX_APPROACH_AXIS[factory]]
    return 0.25 * scale, 0.75 * scale, dict(zip(("box_x", "box_y", "box_z"), (d * scale for d in dims)))


def _box_approach_clearance(L, A, dims, factory):
    return min(L, dims[BOX_APPROACH_AXIS[factory]]) / 2


def _is_cap(p):
    return p.mode in ("top", "bottom")


def _is_radial(p):
    return p.mode in ("surface", "side")


def _is_box_face(factory):
    axis = BOX_APPROACH_AXIS[factory]
    if axis == "box_z":
        return _is_cap
    sign = {"box_x": ("+x", "-x"), "box_y": ("+y", "-y")}[axis]
    return lambda p: p.approach in sign


BOUNDARIES: Tuple[Boundary, ...] = (
    # Short-cylinder cap band: the fingertip must stop a margin short of the far cap,
    # so the band [m, min(L, h) - m] closes at c == h/2 when h < L (#122).
    Boundary(
        name="cap_band_height_half",
        factories=CAP_FACTORIES,
        setup=lambda s, f: (0.125 * s, 0.5 * s, {"cylinder_radius": 0.03125 * s, "cylinder_height": 0.0625 * s}),
        clearance=lambda L, A, d, f: d["cylinder_height"] / 2,
        family=_is_cap,
    ),
    # Cylinder-side HEIGHT band: c == h/2 is one centred height, not empty (#124).
    Boundary(
        name="cylinder_side_height_half",
        factories=("grasp_cylinder_side", "grasp_cylinder"),
        setup=lambda s, f: (0.125 * s, 0.5 * s, {"cylinder_radius": 0.0625 * s, "cylinder_height": 0.0625 * s}),
        clearance=lambda L, A, d, f: d["cylinder_height"] / 2,
        family=lambda p: p.mode == "side",
    ),
    # Radial reach with short fingers (L < 2r): the band [r, L - c] closes at c == L - r.
    Boundary(
        name="radial_reach",
        factories=RADIAL_FACTORIES,
        setup=lambda s, f: _radial_setup(s, f, 0.03125, 0.015625),
        clearance=lambda L, A, d, f: L - _radius(d),
        family=_is_radial,
    ),
    # Radial far-surface limit with long fingers (2r <= L): closes at c == r.
    Boundary(
        name="radial_far_surface",
        factories=RADIAL_FACTORIES,
        setup=lambda s, f: _radial_setup(s, f, 0.25, 0.0625),
        clearance=lambda L, A, d, f: _radius(d),
        family=_is_radial,
    ),
    # Box approach band: closes at c == min(L, approach extent) / 2.
    Boundary(
        name="box_approach_band",
        factories=BOX_FACE_FACTORIES,
        setup=_box_setup,
        clearance=_box_approach_clearance,
        family=lambda p: True,  # replaced per factory in boundary_transition_failures
    ),
    # Box slide band: a slide dimension of exactly 2c is one centred pose (#110).
    Boundary(
        name="box_slide_band",
        factories=("grasp_box_top", "grasp_box"),
        setup=lambda s, f: (0.25 * s, 0.75 * s, {"box_x": 0.125 * s, "box_y": 0.25 * s, "box_z": 0.25 * s}),
        clearance=lambda L, A, d, f: d["box_x"] / 2,
        family=lambda p: p.mode == "top" and p.finger_orientation == "y",  # spans y, slides along x
    ),
    # Torus span reach: the band [r, L - c] closes at c == L - r.
    Boundary(
        name="torus_span_reach",
        factories=("grasp_torus_span", "grasp_torus"),
        setup=lambda s, f: (0.125 * s, 0.75 * s, {"torus_radius": 0.125 * s, "tube_radius": 0.03125 * s}),
        clearance=lambda L, A, d, f: L - d["tube_radius"],
        family=lambda p: p.mode == "span",
    ),
    # Default-preshape aperture limit: span + c <= A closes at c == A - span.
    Boundary(
        name="aperture_max",
        factories=("grasp_box_top", "grasp_box"),
        setup=lambda s, f: (0.25 * s, 0.5 * s, {"box_x": 0.4375 * s, "box_y": 0.25 * s, "box_z": 0.25 * s}),
        clearance=lambda L, A, d, f: A - d["box_x"],
        family=lambda p: p.mode == "top" and p.finger_orientation == "x",  # spans box_x
        # The generator compares ``box_x + c`` against ``A``; one ulp of the smaller
        # ``c`` is lost in that sum, so step relatively.
        relative_step=True,
    ),
    # Scale-aware straddle floor: the default preshape span + c clears each pad by
    # 2*atol only for c >= 4*length_atol(scale) (#107, #129). Feasible ABOVE it.
    Boundary(
        name="straddle_floor",
        factories=("grasp_sphere",),
        setup=lambda s, f: (0.5 * s, 0.5 * s, {"object_radius": 0.0625 * s}),
        clearance=lambda L, A, d, f: straddle_floor_preshape(2 * d["object_radius"], d["object_radius"]),
        family=lambda p: p.mode == "surface",
        feasible_below=False,
        param="preshape",
        fixed_clearance=lambda L, A, d, f: 0.1 * d["object_radius"],
    ),
)


def boundary_triplet(b: Boundary, factory: str, scale: float) -> List[Tuple[str, GraspCase, bool]]:
    """``(position, case, expect_family)`` for below / exact / above the boundary."""
    L, A, dims = b.setup(scale, factory)
    spec = GripperSpec("ParallelJawGripper", L, A)
    value = b.clearance(L, A, dims, factory)
    below, above = b.neighbours(value)
    out = []
    for position, v in (("below", below), ("exact", value), ("above", above)):
        expect = b.equality_feasible if position == "exact" else (position == "below") == b.feasible_below
        options: Dict[str, Any] = {"k": 3}
        if b.param == "clearance":
            options["clearance"] = v
        else:
            options["preshape"] = v
            options["clearance"] = b.fixed_clearance(L, A, dims, factory)
        out.append((position, GraspCase(spec, factory, dims, options), expect))
    return out


def straddle_floor_preshape(span: float, scale: float) -> float:
    """The smallest preshape whose realized margin clears the straddle floor (#129).

    ``span + 4*atol`` may round to a float whose margin is a hair under the floor, so
    step up until the margin -- the quantity the rule actually tests -- clears it.
    """
    floor = 4.0 * (1e-9 + 1e-6 * scale)
    p = span + floor
    while p - span < floor:
        p = math.nextafter(p, math.inf)
    return p


def boundary_transition_failures(b: Boundary, factory: str, scale: float) -> List[str]:
    """The family appears on the feasible side, vanishes on the other, and is sound."""
    family = _is_box_face(factory) if b.name == "box_approach_band" else b.family
    out: List[str] = []
    for position, case, expect in boundary_triplet(b, factory, scale):
        templates = [t for t in case.run() if family(t.provenance)]
        where = f"{b.name}/{factory}/scale={scale:g}/{position}"
        if expect and not templates:
            out.append(f"{where}: expected the family to be feasible, got none")
        elif not expect and templates:
            out.append(f"{where}: expected no templates, got {len(templates)}")
        elif expect and (position != "exact" or b.certify_exact):
            out.extend(f"{where}: {f}" for f in soundness_failures(case, templates))
    return out
