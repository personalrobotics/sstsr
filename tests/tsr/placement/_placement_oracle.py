# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Independent analytic oracle for placement TSRs (issue #148).

Given only the object's own geometry, a concrete pose, and the surface extents, this
certifies what a placement template claims. It is **independent**: every quantity comes
from a support function of the posed body, never from the generator's construction
formulas, so a template cannot certify itself.

For a convex body the support function ``h(d) = max_{p in body} p·d`` gives exact
extremes with no sampling, which is what makes the resting clause testable at the
precision the claim needs:

* the lowest point along the surface normal is ``t_z - h(-z)``;
* the reach in a horizontal direction ``d`` is ``t·d + h(d)``, to be compared with the
  surface rectangle's own support ``table_x |d_x| + table_y |d_y|``.

Clauses, in the numbering the placement contract uses:

* ``1`` **resting**  — the object touches the surface and does not penetrate it;
* ``2`` **supported** — the centre of mass projects inside the contact patch;
* ``3`` **on the surface** — the whole object lies within the surface footprint.

A pose that fails any clause is not a stable placement, whatever the template says.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

_Z = np.array([0.0, 0.0, 1.0])

#: Horizontal directions for the containment clause. The bodies here are convex with
#: at most circular horizontal sections, so a fan of directions bounds the footprint to
#: O(1/n²) -- far below the length tolerance at these scales.
_FAN = np.stack([np.array([math.cos(a), math.sin(a), 0.0]) for a in np.linspace(0.0, 2 * math.pi, 180, endpoint=False)])


#: Horizontal directions for a circular opening's containment clause: 1 − cos(π/4096) is
#: 3e-7, below the length tolerance's 1e-6 of the scale for any object within it.
_FINE_FAN = np.stack(
    [np.array([math.cos(a), math.sin(a), 0.0]) for a in np.linspace(0.0, 2 * math.pi, 4096, endpoint=False)]
)


def length_atol(scale: float) -> float:
    """Scale-aware length tolerance ``1e-9 + 1e-6 · scale`` (the geometric contract)."""
    return 1e-9 + 1e-6 * float(scale)


# --------------------------------------------------------------------------- #
# Primitives: support function, contact patch, centre of mass
# --------------------------------------------------------------------------- #


class _Primitive:
    """A convex body in its own frame, described by its support function."""

    def _support(self, local: np.ndarray) -> np.ndarray:
        """``max p·d`` for each row of ``local``, expressed in the body frame."""
        raise NotImplementedError

    def support(self, d: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Support of the body rotated by ``R``, for one direction or a stack of them."""
        d = np.asarray(d, dtype=float)
        local = np.atleast_2d(d) @ R  # R.T @ d, row-wise
        h = self._support(local)
        return h[0] if d.ndim == 1 else h

    def contact(self, R: np.ndarray, t: np.ndarray, atol: float) -> np.ndarray:
        """Surface points at the lowest world z: the contact patch, as xy points."""
        world = self._boundary() @ R.T + t
        return world[world[:, 2] <= world[:, 2].min() + atol]

    def _boundary(self) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class Box(_Primitive):
    lx: float
    ly: float
    lz: float

    @property
    def scale(self) -> float:
        return max(self.lx, self.ly, self.lz)

    @property
    def _half(self) -> np.ndarray:
        return np.array([self.lx, self.ly, self.lz]) / 2.0

    def _support(self, local: np.ndarray) -> np.ndarray:
        return np.abs(local) @ self._half

    def _boundary(self) -> np.ndarray:
        return np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * self._half


@dataclass(frozen=True)
class Cylinder(_Primitive):
    radius: float
    height: float

    @property
    def scale(self) -> float:
        return max(2 * self.radius, self.height)

    def _support(self, local: np.ndarray) -> np.ndarray:
        axial = np.abs(local[:, 2]) * (self.height / 2.0)
        radial = np.hypot(local[:, 0], local[:, 1]) * self.radius
        return axial + radial

    def _boundary(self) -> np.ndarray:
        a = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)
        rim = np.stack([self.radius * np.cos(a), self.radius * np.sin(a)], axis=-1)
        caps = [np.c_[rim, np.full(len(rim), z)] for z in (-self.height / 2, self.height / 2)]
        return np.vstack(caps)


@dataclass(frozen=True)
class Sphere(_Primitive):
    radius: float

    @property
    def scale(self) -> float:
        return 2 * self.radius

    def _support(self, local: np.ndarray) -> np.ndarray:
        return self.radius * np.linalg.norm(local, axis=1)

    def contact(self, R: np.ndarray, t: np.ndarray, atol: float) -> np.ndarray:
        # A sphere touches at the single point directly below its centre.
        return np.array([[t[0], t[1], t[2] - self.radius]])


@dataclass(frozen=True)
class Torus(_Primitive):
    major: float
    minor: float

    @property
    def scale(self) -> float:
        return 2 * (self.major + self.minor)

    def _support(self, local: np.ndarray) -> np.ndarray:
        ring = np.hypot(local[:, 0], local[:, 1]) * self.major  # the tube-centre circle
        return ring + self.minor * np.linalg.norm(local, axis=1)

    def _boundary(self) -> np.ndarray:
        u = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)
        v = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        ring = self.major + self.minor * np.cos(vv)
        return np.stack([ring * np.cos(uu), ring * np.sin(uu), self.minor * np.sin(vv)], axis=-1).reshape(-1, 3)


@dataclass(frozen=True)
class Mesh(_Primitive):
    vertices: np.ndarray
    com: np.ndarray

    @property
    def scale(self) -> float:
        return float(np.ptp(self.vertices, axis=0).max())

    def _support(self, local: np.ndarray) -> np.ndarray:
        return (self.vertices @ local.T).max(axis=0)

    def _boundary(self) -> np.ndarray:
        return self.vertices


@dataclass(frozen=True)
class Capsule(_Primitive):
    """The points within ``radius`` of a segment of ``length`` along z, centred."""

    radius: float
    length: float

    @property
    def scale(self) -> float:
        return self.length + 2 * self.radius

    def _support(self, local: np.ndarray) -> np.ndarray:
        return np.abs(local[:, 2]) * (self.length / 2.0) + self.radius * np.linalg.norm(local, axis=1)

    def _boundary(self) -> np.ndarray:
        u = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)
        v = np.linspace(-np.pi / 2, np.pi / 2, 37)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        ball = np.stack([np.cos(vv) * np.cos(uu), np.cos(vv) * np.sin(uu), np.sin(vv)], axis=-1).reshape(-1, 3)
        return np.vstack([ball * self.radius + [0.0, 0.0, s * self.length / 2.0] for s in (-1, 1)])


@dataclass(frozen=True)
class Compound(_Primitive):
    """The union of posed primitives ``(prim, T_object_prim)``, with its centre of mass.

    The support of a union is the largest of its parts' supports, and its contact patch
    is the union of the patches of the parts that reach the lowest point, so it is
    certified from the parts' own support functions like every other body here.
    """

    parts: Tuple[Tuple[_Primitive, np.ndarray], ...]
    com: np.ndarray

    @property
    def scale(self) -> float:
        return max(float(np.linalg.norm(T[:3, 3])) + p.scale for p, T in self.parts)

    def support(self, d: np.ndarray, R: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        rows = np.atleast_2d(d)
        h = np.max([rows @ (R @ T[:3, 3]) + np.atleast_1d(p.support(d, R @ T[:3, :3])) for p, T in self.parts], axis=0)
        return h[0] if d.ndim == 1 else h

    def contact(self, R: np.ndarray, t: np.ndarray, atol: float) -> np.ndarray:
        posed = [(p, R @ T[:3, :3], R @ T[:3, 3] + t) for p, T in self.parts]
        lowest = [float(tp[2] - p.support(-_Z, Rp)) for p, Rp, tp in posed]
        floor = min(lowest)
        patches = [p.contact(Rp, tp, atol) for (p, Rp, tp), z in zip(posed, lowest) if z <= floor + atol]
        return np.vstack(patches)


def centre_of_mass(prim, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The posed centre of mass; the frame origin for the centred primitives."""
    local = prim.com if isinstance(prim, (Mesh, Compound)) else np.zeros(3)
    return R @ local + t


# --------------------------------------------------------------------------- #
# Certification
# --------------------------------------------------------------------------- #


@dataclass
class PlacementWitness:
    """Structured verdict for one posed object, with the geometry behind it."""

    ok: bool
    lowest_z: float
    footprint_radius: float
    support_margin: float  # distance from the COM projection to the patch boundary [m]
    failed: List[Tuple[int, str]]


def _hull_2d(points: np.ndarray) -> Optional[np.ndarray]:
    from scipy.spatial import ConvexHull, QhullError

    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    return points[hull.vertices]


def _distance_inside(polygon: np.ndarray, point: np.ndarray) -> float:
    """Signed distance from ``point`` to the polygon's boundary; positive inside."""
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > 0.0
    normals = np.stack([edges[keep, 1], -edges[keep, 0]], axis=-1) / lengths[keep, None]  # outward for CCW
    signed = np.einsum("ij,ij->i", normals, point - polygon[keep])
    # Inside: the clearance to the nearest edge. Outside: minus the deepest violation.
    return float(-signed.max())


def tipping_angle(prim, pose: np.ndarray) -> float:
    """Clause 4: the angle the posed object must rotate before it falls, in radians.

    Derived from the pose alone — the contact patch at the lowest ``z``, its convex hull,
    and where the centre of mass projects into it — so it is independent of whatever
    arithmetic produced the template's ``stability_margin``. Rotating about the nearest
    support edge, the centre of mass sits ``d`` from that edge horizontally and ``h``
    above the surface, and passes over it after ``arctan(d / h)``.

    A sphere returns 0: point contact directly beneath the centre is neutral, not
    critical. A contact patch with no area returns 0 for the same reason it fails
    clause 2 — there is no edge to tip about.
    """
    R, t = pose[:3, :3], pose[:3, 3]
    com = centre_of_mass(prim, R, t)
    if isinstance(prim, Sphere):
        return 0.0
    patch = prim.contact(R, t, max(length_atol(prim.scale), 1e-9))[:, :2]
    polygon = _hull_2d(patch)
    if polygon is None:
        return 0.0
    return float(math.atan2(_distance_inside(polygon, com[:2]), com[2]))


def certify(prim, pose: np.ndarray, *, table: Tuple[float, float], atol: Optional[float] = None) -> PlacementWitness:
    """Certify that ``pose`` places ``prim`` as a stable placement on the surface.

    Args:
        prim: One of the primitives above, in its own frame.
        pose: 4x4 object pose in the surface frame (``z = 0`` is the surface).
        table: ``(table_x, table_y)`` half-extents of the surface.
        atol: Length tolerance; the scale-aware contract value by default.
    """
    R, t = pose[:3, :3], pose[:3, 3]
    atol = length_atol(prim.scale) if atol is None else atol
    failed: List[Tuple[int, str]] = []

    # Clause 1: resting. The lowest point of the posed body is the surface plane.
    lowest = float(t[2] - prim.support(-_Z, R))
    if lowest < -atol:
        failed.append((1, f"penetrates the surface by {-lowest:.6g} m"))
    elif lowest > atol:
        failed.append((1, f"floats {lowest:.6g} m above the surface"))

    # Clause 2: supported. The COM projects inside the contact patch, which is what
    # separates a resting pose from one that tips.
    com = centre_of_mass(prim, R, t)
    margin = 0.0
    if not isinstance(prim, Sphere):  # point contact under the centre: neutral, never tips
        patch = prim.contact(R, t, max(atol, 1e-9))[:, :2]
        polygon = _hull_2d(patch)
        if polygon is None:
            if len(patch) == 0:
                failed.append((2, "no contact patch"))
        else:
            margin = _distance_inside(polygon, com[:2])
            if margin < -atol:
                failed.append((2, f"centre of mass projects {-margin:.6g} m outside the contact patch"))

    # Clause 3: on the surface. The footprint comes from the support function, so this
    # bounds the whole body and not merely its vertices.
    reach = _FAN[:, :2] @ t[:2] + prim.support(_FAN, R)
    limit = table[0] * np.abs(_FAN[:, 0]) + table[1] * np.abs(_FAN[:, 1])
    over = reach - limit
    worst = int(np.argmax(over))
    if over[worst] > atol:
        failed.append((3, f"extends {over[worst]:.6g} m beyond the surface"))

    return PlacementWitness(not failed, lowest, float(prim.support(_FAN, R).max()), float(margin), failed)


def certify_release(
    prim,
    pose: np.ndarray,
    *,
    opening: Tuple[str, Any],
    lowest: Tuple[float, float],
    clearance: float,
    atol: Optional[float] = None,
) -> PlacementWitness:
    """Certify that ``pose`` releases ``prim`` into a container through its opening (#200).

    Args:
        prim: One of the primitives above, in its own frame.
        pose: 4x4 object pose in the opening frame (``z = 0`` is the rim plane, z out).
        opening: ``("rectangle", (x, y))``, full extents, or ``("circle", radius)``.
        lowest: The heights the object's lowest point may take above the rim plane.
        clearance: How far inside the opening the whole object must stay.
        atol: Length tolerance; the scale-aware contract value by default.

    Clauses of the release contract:

    * ``1`` **resting** -- moved down onto the rim plane, the object is a stable
      placement there (the surface clauses 1 and 2), so it is released as it rests;
    * ``2`` **inside** -- the whole object lies inside the opening by ``clearance``;
    * ``3`` **at its height** -- its lowest point is within ``lowest`` of the rim plane.
    """
    R, t = pose[:3, :3], pose[:3, 3]
    atol = length_atol(prim.scale) if atol is None else atol
    failed: List[Tuple[int, str]] = []

    lowest_z = float(t[2] - prim.support(-_Z, R))
    rested = np.array(pose, dtype=float)
    rested[2, 3] -= lowest_z
    on_rim = certify(prim, rested, table=(1.0e6, 1.0e6), atol=atol)
    failed += [(1, message) for clause, message in on_rim.failed if clause in (1, 2)]

    # The opening's own support, less the clearance, bounds the body's in every direction.
    reach = _FAN[:, :2] @ t[:2] + prim.support(_FAN, R)
    # A rectangle's bound is decided along its axes, which the fan holds exactly; a
    # circle's, in every direction, so it takes a fan fine enough that what lies between
    # its directions, |t|·(1 − cos(π/n)), is below the length tolerance.
    kind, extent = opening
    if kind == "rectangle":
        limit = (extent[0] / 2 - clearance) * np.abs(_FAN[:, 0]) + (extent[1] / 2 - clearance) * np.abs(_FAN[:, 1])
    else:
        reach = _FINE_FAN[:, :2] @ t[:2] + prim.support(_FINE_FAN, R)
        limit = np.full(len(_FINE_FAN), float(extent) - clearance)
    over = reach - limit
    worst = int(np.argmax(over))
    if over[worst] > atol:
        failed.append((2, f"extends {over[worst]:.6g} m past the opening's clearance"))

    if not lowest[0] - atol <= lowest_z <= lowest[1] + atol:
        failed.append((3, f"lowest point {lowest_z:.6g} m from the rim, outside {lowest}"))

    return PlacementWitness(not failed, lowest_z, float(prim.support(_FAN, R).max()), on_rim.support_margin, failed)
