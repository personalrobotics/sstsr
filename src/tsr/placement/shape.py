# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""An object's geometry for :meth:`StablePlacer.place`: parts in the object frame (#201).

A :class:`Shape` is what a collision model already is: one or more convex parts, each a
box, cylinder, sphere, capsule or convex hull at its own pose in the object frame, and the
object's centre of mass. The caller describes the object once; ``place`` decides how to
place it, so nothing upstream has to know which parts it is made of.

Each part answers four questions in the object frame, which is all placement needs:

* its **support** ``h(n) = max p·n`` over the part, exactly;
* its **contact** with the plane ``p·n = h``: the part's points on that plane, exactly for
  flat contacts and as the vertices of an inscribed polygon for a cylinder's cap, so a
  contact patch is never larger than the true one;
* **surface points**, every one on the part, from which candidate resting faces are found;
* its **reach**: how far it extends horizontally from the object origin once the object is
  rotated by ``R``, exactly, or for a tilted cylinder bounded from above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..core.tsr import _check_frame

#: Points per circle where a circle is sampled: a cylinder's rims, a sphere's surface.
_CIRCLE = 64
#: Samples of a tilted cylinder's rim for its reach, which is then bounded above by the
#: largest gap to a sample, ``radius · pi / _REACH``: within 0.16% of the true reach.
_REACH = 2048


def _positive(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return float(value)


def _pose(value) -> np.ndarray:
    T = _check_frame("pose", np.eye(4) if value is None else value)
    T.setflags(write=False)
    return T


def _fibonacci(n: int) -> np.ndarray:
    """``n`` unit vectors spread over the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    a = math.pi * (1.0 + math.sqrt(5.0)) * i
    return np.stack([r * np.cos(a), r * np.sin(a), z], axis=-1)


def _circle(centre: np.ndarray, axis: np.ndarray, radius: float, n: int) -> np.ndarray:
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed - np.dot(seed, axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return centre + radius * (np.outer(np.cos(a), u) + np.outer(np.sin(a), v))


def _horizontal(points: np.ndarray, R: np.ndarray) -> float:
    return float(np.linalg.norm((points @ R.T)[:, :2], axis=1).max())


class _Part:
    """A convex part at ``pose`` in the object frame."""

    pose: np.ndarray

    @property
    def centre(self) -> np.ndarray:
        return self.pose[:3, 3]

    @property
    def _axis(self) -> np.ndarray:
        return self.pose[:3, 2]

    def support(self, n: np.ndarray) -> float:
        raise NotImplementedError

    def contact(self, n: np.ndarray, h: float, atol: float) -> np.ndarray:
        """The part's points on the plane ``p·n = h``, or none if it does not reach it."""
        raise NotImplementedError

    def surface(self) -> np.ndarray:
        raise NotImplementedError

    def reach(self, R: np.ndarray) -> float:
        raise NotImplementedError

    @property
    def volume(self) -> float:
        raise NotImplementedError

    @property
    def centroid(self) -> np.ndarray:
        return self.centre


@dataclass(frozen=True, eq=False)
class Box(_Part):
    """A box of full extents ``size`` (m), centred on ``pose``'s origin, along its axes."""

    size: Tuple[float, float, float]
    pose: np.ndarray = field(default=None)

    def __post_init__(self):
        size = tuple(_positive(f"size[{i}]", s) for i, s in enumerate(np.asarray(self.size).ravel()))
        if len(size) != 3:
            raise ValueError(f"size must have 3 entries, got {len(size)}")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "pose", _pose(self.pose))

    @property
    def corners(self) -> np.ndarray:
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        return (signs * np.array(self.size) / 2.0) @ self.pose[:3, :3].T + self.centre

    def support(self, n):
        return float((self.corners @ n).max())

    def contact(self, n, h, atol):
        c = self.corners
        return c[c @ n >= h - atol]

    def surface(self):
        return self.corners

    def reach(self, R):
        return _horizontal(self.corners, R)

    @property
    def volume(self):
        return float(np.prod(self.size))


@dataclass(frozen=True, eq=False)
class Cylinder(_Part):
    """A solid cylinder of ``radius`` and full ``height`` (m) along ``pose``'s z axis,
    centred on its origin."""

    radius: float
    height: float
    pose: np.ndarray = field(default=None)

    def __post_init__(self):
        object.__setattr__(self, "radius", _positive("radius", self.radius))
        object.__setattr__(self, "height", _positive("height", self.height))
        object.__setattr__(self, "pose", _pose(self.pose))

    @property
    def ends(self) -> np.ndarray:
        half = self._axis * self.height / 2.0
        return np.stack([self.centre - half, self.centre + half])

    def support(self, n):
        along = float(np.dot(self._axis, n))
        return (
            float(np.dot(self.centre, n))
            + abs(along) * self.height / 2.0
            + self.radius * math.sqrt(max(0.0, 1.0 - along * along))
        )

    def contact(self, n, h, atol):
        # The cap's inscribed polygon covers a cap-down contact, and the exact tangent
        # points on each rim cover a side or rim contact.
        candidates = [self.surface()]
        across = n - np.dot(n, self._axis) * self._axis
        if np.linalg.norm(across) > 1e-12:
            candidates.append(self.ends + self.radius * across / np.linalg.norm(across))
        points = np.vstack(candidates)
        return points[points @ n >= h - atol]

    def surface(self):
        return np.vstack([_circle(e, self._axis, self.radius, _CIRCLE) for e in self.ends])

    def reach(self, R):
        if abs(abs((R @ self._axis)[2]) - 1.0) <= 1e-12:  # upright: its rims are level circles
            return float(np.linalg.norm((R @ self.centre)[:2])) + self.radius
        rims = np.vstack([_circle(e, self._axis, self.radius, _REACH) for e in self.ends])
        return _horizontal(rims, R) + self.radius * math.pi / _REACH

    @property
    def volume(self):
        return math.pi * self.radius**2 * self.height


@dataclass(frozen=True, eq=False)
class Sphere(_Part):
    """A solid sphere of ``radius`` (m) centred on ``pose``'s origin."""

    radius: float
    pose: np.ndarray = field(default=None)

    def __post_init__(self):
        object.__setattr__(self, "radius", _positive("radius", self.radius))
        object.__setattr__(self, "pose", _pose(self.pose))

    def support(self, n):
        return float(np.dot(self.centre, n)) + self.radius

    def contact(self, n, h, atol):
        point = self.centre + self.radius * n
        return point[None, :] if point @ n >= h - atol else np.empty((0, 3))

    def surface(self):
        return self.centre + self.radius * _fibonacci(_CIRCLE)

    def reach(self, R):
        return float(np.linalg.norm((R @ self.centre)[:2])) + self.radius

    @property
    def volume(self):
        return 4.0 / 3.0 * math.pi * self.radius**3


@dataclass(frozen=True, eq=False)
class Capsule(_Part):
    """A capsule: the points within ``radius`` (m) of a segment of ``length`` (m) along
    ``pose``'s z axis, centred on its origin. ``length`` excludes the end caps."""

    radius: float
    length: float
    pose: np.ndarray = field(default=None)

    def __post_init__(self):
        object.__setattr__(self, "radius", _positive("radius", self.radius))
        object.__setattr__(self, "length", _positive("length", self.length))
        object.__setattr__(self, "pose", _pose(self.pose))

    @property
    def ends(self) -> np.ndarray:
        half = self._axis * self.length / 2.0
        return np.stack([self.centre - half, self.centre + half])

    def support(self, n):
        return float((self.ends @ n).max()) + self.radius

    def contact(self, n, h, atol):
        points = self.ends + self.radius * n
        return points[points @ n >= h - atol]

    def surface(self):
        return np.vstack([e + self.radius * _fibonacci(_CIRCLE) for e in self.ends])

    def reach(self, R):
        return _horizontal(self.ends, R) + self.radius

    @property
    def volume(self):
        return math.pi * self.radius**2 * self.length + 4.0 / 3.0 * math.pi * self.radius**3


@dataclass(frozen=True, eq=False)
class Hull(_Part):
    """The convex hull of ``vertices`` (m), given in ``pose``'s frame."""

    vertices: np.ndarray
    pose: np.ndarray = field(default=None)

    def __post_init__(self):
        v = np.asarray(self.vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
            raise ValueError(f"vertices must have shape (N, 3) with N >= 1, got {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError("vertices must be finite")
        pose = _pose(self.pose)
        world = v @ pose[:3, :3].T + pose[:3, 3]
        world.setflags(write=False)
        object.__setattr__(self, "vertices", world)  # held in the object frame
        object.__setattr__(self, "pose", np.eye(4))

    def support(self, n):
        return float((self.vertices @ n).max())

    def contact(self, n, h, atol):
        return self.vertices[self.vertices @ n >= h - atol]

    def surface(self):
        return self.vertices

    def reach(self, R):
        return _horizontal(self.vertices, R)

    def _tetrahedra(self) -> Tuple[np.ndarray, np.ndarray]:
        """Each hull facet with an interior point: the volumes and centroids of a
        decomposition of the hull."""
        try:
            hull = ConvexHull(self.vertices)
        except QhullError:
            return np.zeros(1), self.vertices.mean(axis=0)[None, :]  # flat: no volume
        apex = self.vertices[hull.vertices].mean(axis=0)
        tri = self.vertices[hull.simplices]
        volumes = np.abs(np.einsum("ij,ij->i", tri[:, 0] - apex, np.cross(tri[:, 1] - apex, tri[:, 2] - apex))) / 6.0
        return volumes, (tri.sum(axis=1) + apex) / 4.0

    @property
    def volume(self):
        return float(self._tetrahedra()[0].sum())

    @property
    def centroid(self):
        volumes, centroids = self._tetrahedra()
        if volumes.sum() <= 0.0:
            return centroids.mean(axis=0)
        return (volumes[:, None] * centroids).sum(axis=0) / volumes.sum()


Part = Union[Box, Cylinder, Sphere, Capsule, Hull]
_PARTS = (Box, Cylinder, Sphere, Capsule, Hull)


@dataclass(frozen=True, eq=False)
class Shape:
    """An object for :meth:`StablePlacer.place`: its convex ``parts`` in the object frame,
    and its centre of mass ``com`` in the same frame.

    Without ``com``, the parts' volume-weighted centroid is used, as for a uniform solid
    whose parts do not overlap. Pass the real centre of mass when it is known: a
    simulator's compiled model has it.
    """

    parts: Tuple[Part, ...]
    com: Optional[np.ndarray] = None

    def __post_init__(self):
        if isinstance(self.parts, _PARTS) or not isinstance(self.parts, Sequence) or not self.parts:
            raise ValueError("parts must be a non-empty sequence of Box, Cylinder, Sphere, Capsule or Hull")
        if not all(isinstance(p, _PARTS) for p in self.parts):
            raise ValueError("parts must be a non-empty sequence of Box, Cylinder, Sphere, Capsule or Hull")
        object.__setattr__(self, "parts", tuple(self.parts))
        if self.com is None:
            volumes = np.array([p.volume for p in self.parts])
            centroids = np.stack([p.centroid for p in self.parts])
            com = centroids.mean(axis=0) if volumes.sum() <= 0.0 else volumes @ centroids / volumes.sum()
        else:
            com = np.asarray(self.com, dtype=float)
            if com.shape != (3,) or not np.all(np.isfinite(com)):
                raise ValueError(f"com must be a finite length-3 point, got {self.com!r}")
        com = np.array(com, dtype=float)
        com.setflags(write=False)
        object.__setattr__(self, "com", com)

    def support(self, n: np.ndarray) -> float:
        return max(p.support(n) for p in self.parts)

    def surface(self) -> np.ndarray:
        return np.vstack([p.surface() for p in self.parts])

    def reach(self, R: np.ndarray) -> float:
        return max(p.reach(R) for p in self.parts)
