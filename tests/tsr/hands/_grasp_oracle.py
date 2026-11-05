# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Independent analytic oracle for primitive parallel-jaw grasps (issue #67).

This is a **test-side** oracle. Given only

* the primitive type and dimensions,
* the idealized gripper dimensions (finger length ``L``, max aperture ``A``),
* the requested preshape ``a`` and clearance ``c``,
* a concrete end-effector pose ``E`` (in the primitive's reference frame), and
* the structural grasp selectors ``mode`` (and optional ``approach`` /
  ``finger_orientation``), used **only** to select and check the analytic model,

it certifies the geometric-soundness clauses of ``docs/ARCHITECTURE.md``.

**Exact, not sampled (#93).** Contacts, spans, normals, realized insertion depth,
and torus minor angle are derived in **closed form** per ``(primitive, mode)`` from
the concrete pose; there are no approach/closing rasters, so results do not depend
on any grid resolution and each certification is O(1) arithmetic. The independent
signed-distance representation is used only as a *verifier* -- it confirms the
analytic contacts lie on the surface and the analytic normals match the SDF
gradient -- never as the search itself.

**Provenance independence (#84).** ``mode``/``approach``/``finger_orientation`` are
structural claim labels that select which analytic model to apply and are *checked*
against the realized pose geometry (a mislabeled mode fails). The oracle never
consumes recorded numeric provenance (``depth``, minor angle, span, metadata) as
evidence; every certified quantity is derived from the pose and primitive geometry.

Frame convention (library invariant): ``z_EE`` = approach, ``y_EE`` = finger opening
(jaws always close along ``±y_EE``), ``x_EE = y_EE × z_EE``. The palm is the
translation of ``E``; the fingertips are ``palm + L·z_EE``.

Clause codes on :class:`GraspWitness`:

* ``1`` model/pose validity (finite, proper SE(3); see also :func:`certify` which
  raises ``ValueError`` for a malformed *model*);
* ``2`` aperture validity and the object fits strictly between the pads;
* ``3`` palm outside the solid with the requested clearance;
* ``4`` the clearance-aware usable band (finger depth, cylinder ends, box edges);
* ``5`` reach: two opposing contacts exist within finger reach;
* ``6`` contact normals oppose the closing directions;
* ``8`` the declared ``mode``/``approach``/``finger_orientation``/``grip`` match the pose;
* ``9`` the contacts lie where the declared grip holds: on the pads, a clearance inside
  both ends, or behind them for an encompassing grasp of a wide enough object (#189).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

ANGLE_ATOL = 1e-6  # rad, contact-normal opposition (docs/ARCHITECTURE.md)

_EX = np.array([1.0, 0.0, 0.0])
_EY = np.array([0.0, 1.0, 0.0])
_EZ = np.array([0.0, 0.0, 1.0])

SUPPORTED_MODES = {
    "sphere": ("surface",),
    "cylinder": ("side", "top", "bottom"),
    "box": ("top", "bottom", "face"),
    "torus": ("span", "side"),
}

# Built-in selector vocabulary, aligned with tsr.hands._conformance._NATIVE (#98).
# ``approach`` is the object side the HAND OCCUPIES (not the sign of z_EE); a signed
# label ``+x`` means the palm is on the +x side, i.e. z_EE = -x. Family labels
# (radial / tube / diameter / tangential) map to geometric predicates below.
_BUILTIN_SELECTORS = {
    ("sphere", "surface"): {"approach": {"radial"}, "finger_orientation": {"diameter"}},
    ("cylinder", "side"): {"approach": {"radial"}, "finger_orientation": {"tangential"}},
    ("cylinder", "top"): {"approach": {"+z"}, "finger_orientation": {"diameter"}},
    ("cylinder", "bottom"): {"approach": {"-z"}, "finger_orientation": {"diameter"}},
    ("box", "top"): {"approach": {"+z"}, "finger_orientation": {"x", "y"}},
    ("box", "bottom"): {"approach": {"-z"}, "finger_orientation": {"x", "y"}},
    ("box", "face"): {"approach": {"+x", "-x", "+y", "-y"}, "finger_orientation": {"x", "y", "z"}},
    ("torus", "span"): {"approach": {"+z", "-z"}, "finger_orientation": {"diameter"}},
    ("torus", "side"): {"approach": {"tube"}, "finger_orientation": {"tangential"}},
}
_SIGNED_AXES = {
    "+x": (_EX, 1.0),
    "-x": (_EX, -1.0),
    "+y": (_EY, 1.0),
    "-y": (_EY, -1.0),
    "+z": (_EZ, 1.0),
    "-z": (_EZ, -1.0),
}


def length_atol(scale: float) -> float:
    """Scale-aware length tolerance ``1e-9 + 1e-6 · scale``."""
    return 1e-9 + 1e-6 * float(scale)


# --------------------------------------------------------------------------- #
# Independent primitive signed-distance representations (used only as a VERIFIER
# for the analytic contacts below -- never as the search). Standard closed-form
# SDFs from the object-frame conventions in tsr.hands.base.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Sphere:
    radius: float

    @property
    def scale(self) -> float:
        return self.radius

    def sdf(self, p: np.ndarray) -> float:
        return float(np.linalg.norm(p) - self.radius)


@dataclass(frozen=True)
class Cylinder:
    radius: float
    height: float  # axis +z, z in [0, height]

    @property
    def scale(self) -> float:
        return max(self.radius, self.height)

    def sdf(self, p: np.ndarray) -> float:
        radial = float(np.hypot(p[0], p[1])) - self.radius
        axial = abs(p[2] - self.height / 2.0) - self.height / 2.0
        outside = np.array([max(radial, 0.0), max(axial, 0.0)])
        return float(min(max(radial, axial), 0.0) + np.linalg.norm(outside))


@dataclass(frozen=True)
class Box:
    dx: float
    dy: float
    dz: float  # x,y centered; z in [0, dz]

    @property
    def scale(self) -> float:
        return max(self.dx, self.dy, self.dz)

    def sdf(self, p: np.ndarray) -> float:
        center = np.array([0.0, 0.0, self.dz / 2.0])
        half = np.array([self.dx / 2.0, self.dy / 2.0, self.dz / 2.0])
        q = np.abs(p - center) - half
        return float(np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1], q[2]), 0.0))


@dataclass(frozen=True)
class Torus:
    major: float  # tube-center ring radius R
    minor: float  # tube radius r

    @property
    def scale(self) -> float:
        return self.major + self.minor

    def sdf(self, p: np.ndarray) -> float:
        q = np.array([float(np.hypot(p[0], p[1])) - self.major, p[2]])
        return float(np.linalg.norm(q) - self.minor)


def _primitive_name(prim) -> str:
    return type(prim).__name__.lower()


def sdf_normal(prim, point: np.ndarray) -> np.ndarray:
    """Independent outward unit normal from the SDF gradient (central difference).

    The step is scaled to the primitive only (no additive one-metre constant), so
    the verifier is scale-equivariant (#93).
    """
    eps = 1e-6 * prim.scale
    grad = np.empty(3)
    for i in range(3):
        step = np.zeros(3)
        step[i] = eps
        grad[i] = prim.sdf(point + step) - prim.sdf(point - step)
    n = np.linalg.norm(grad)
    return grad / n if n > 0 else grad


# --------------------------------------------------------------------------- #
# Model / pose validation (clause 1, issue #95).
# --------------------------------------------------------------------------- #


def _validate_model(prim, finger_length, max_aperture, preshape, clearance, mode, approach, finger_orientation) -> None:
    """Validate the oracle *model*; raise ``ValueError`` if malformed (#95).

    A malformed model means the test itself is wrong, so this raises rather than
    returning a clause-1 failure (which is reserved for a malformed candidate pose).
    """
    name = _primitive_name(prim)
    if name == "sphere":
        dims = {"radius": prim.radius}
    elif name == "cylinder":
        dims = {"radius": prim.radius, "height": prim.height}
    elif name == "box":
        dims = {"dx": prim.dx, "dy": prim.dy, "dz": prim.dz}
    elif name == "torus":
        dims = {"major": prim.major, "minor": prim.minor}
    else:
        raise ValueError(f"unsupported primitive type {type(prim).__name__!r}")
    for label, value in dims.items():
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{name}.{label} must be finite and > 0, got {value!r}")
    if name == "torus" and not (prim.major > prim.minor):
        raise ValueError(f"torus requires major > minor, got major={prim.major!r}, minor={prim.minor!r}")

    for label, value in (("finger_length", finger_length), ("max_aperture", max_aperture), ("preshape", preshape)):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    if not (np.isfinite(clearance) and clearance >= 0.0):
        raise ValueError(f"clearance must be finite and >= 0, got {clearance!r}")

    if not isinstance(mode, str) or not mode:
        raise ValueError(f"mode must be a non-empty string, got {mode!r}")
    if mode not in SUPPORTED_MODES[name]:
        raise ValueError(f"mode {mode!r} is not supported for a {name} (expected one of {SUPPORTED_MODES[name]})")

    # Selector vocabulary: a non-string or unrecognized label for a built-in
    # (primitive, mode) is a malformed oracle model and raises, like an unsupported
    # mode (#95, #98). The type check keeps unhashable inputs from leaking a
    # TypeError out of the set membership test below (#102).
    for label, value in (("approach", approach), ("finger_orientation", finger_orientation)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{label} must be a string, got {type(value).__name__}")
    allowed = _BUILTIN_SELECTORS[(name, mode)]
    if approach is not None and approach not in allowed["approach"]:
        raise ValueError(
            f"approach {approach!r} is not valid for ({name}, {mode}); expected {sorted(allowed['approach'])}"
        )
    if finger_orientation is not None and finger_orientation not in allowed["finger_orientation"]:
        raise ValueError(
            f"finger_orientation {finger_orientation!r} is not valid for ({name}, {mode}); "
            f"expected {sorted(allowed['finger_orientation'])}"
        )


def _validate_pose(pose) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return ``(pose_float, None)`` for a proper SE(3) pose, else ``(None, reason)``."""
    try:
        p = np.asarray(pose, dtype=float)
    except (ValueError, TypeError):
        return None, "pose is not convertible to a float array"
    if p.shape != (4, 4):
        return None, f"pose must be shape (4, 4), got {p.shape}"
    if not np.all(np.isfinite(p)):
        return None, "pose has non-finite entries"
    if not np.allclose(p[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        return None, f"pose homogeneous row is {p[3]!r}, expected [0, 0, 0, 1]"
    R = p[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        return None, "pose rotation is not orthonormal"
    det = float(np.linalg.det(R))
    if not np.isclose(det, 1.0, atol=1e-6):
        return None, f"pose rotation has det(R)={det:.6g}, not a proper rotation (+1)"
    return p, None


# --------------------------------------------------------------------------- #
# Closed-form line/surface intersections (exact contact geometry).
# --------------------------------------------------------------------------- #


def _line_quadric_roots(o2: np.ndarray, d2: np.ndarray, radius: float) -> Optional[Tuple[float, float]]:
    """Roots of ``|o2 + s·d2|² = radius²`` in 2-D (unit ``d2``), or ``None`` if it misses."""
    b = float(o2 @ d2)
    c = float(o2 @ o2 - radius * radius)
    disc = b * b - c
    if disc < 0.0:
        return None
    root = float(np.sqrt(disc))
    return -b - root, -b + root


@dataclass
class _Contacts:
    base: np.ndarray  # closing-plane point on the approach axis
    contact_lo: np.ndarray
    contact_hi: np.ndarray
    normal_lo: np.ndarray  # analytic outward normals
    normal_hi: np.ndarray
    span: float
    realized_depth: float
    boundary_clearance: float  # min margin to a free edge along the object extent
    minor_angle: Optional[float]
    failed: List[Tuple[int, str]]  # mode/reach failures from the handler (clauses 4/5/6/8)


def _axis_component(vec: np.ndarray, axis: np.ndarray) -> float:
    return float(vec @ axis)


def _aligned(vec: np.ndarray, axis: np.ndarray) -> bool:
    """True iff unit ``vec`` is within ANGLE_ATOL of unit ``axis`` (a true angular
    tolerance, not ``np.isclose``'s default relative tolerance, #100)."""
    return float(vec @ axis) >= np.cos(ANGLE_ATOL)


def _perpendicular(vec: np.ndarray, axis: np.ndarray) -> bool:
    """True iff unit ``vec`` is within ANGLE_ATOL of perpendicular to unit ``axis``."""
    return abs(float(vec @ axis)) <= np.sin(ANGLE_ATOL)


# --------------------------------------------------------------------------- #
# Per-mode analytic handlers. Each derives contacts/normals/depth/boundary from
# the concrete pose and returns a _Contacts (with any mode-consistency, reach, or
# boundary-clearance failures already attached).
# --------------------------------------------------------------------------- #


def _handle_sphere_surface(prim: Sphere, p, a, y, x, L, c, atol) -> _Contacts:
    r = prim.radius
    failed: List[Tuple[int, str]] = []
    # Approach must point toward the centre (origin).
    t_center = float(-(p @ a))
    if t_center <= 0.0:
        failed.append((8, "sphere approach does not point toward the centre"))
    # The clean diameter closing plane is at the centre's projection, capped by reach.
    t_base = min(max(t_center, 0.0), L)
    base = p + t_base * a
    roots = _line_quadric_roots(base, y, r)
    if roots is None:
        failed.append((5, "closing line misses the sphere"))
        return _Contacts(base, base, base, y, -y, np.nan, np.nan, np.inf, None, failed)
    s_lo, s_hi = roots
    contact_lo = base + s_lo * y
    contact_hi = base + s_hi * y
    span = s_hi - s_lo
    # Near-surface entry along the approach ray, for realized insertion depth.
    entry = _line_quadric_roots(p, a, r)
    realized_depth = np.nan if entry is None else L - max(entry[0], 0.0)
    if entry is not None and entry[0] > L + atol:
        failed.append((5, "sphere surface is beyond finger reach"))
    return _Contacts(
        base, contact_lo, contact_hi, contact_lo / r, contact_hi / r, span, realized_depth, np.inf, None, failed
    )


def _handle_cylinder_side(prim: Cylinder, p, a, y, x, L, c, atol) -> _Contacts:
    r, h = prim.radius, prim.height
    failed: List[Tuple[int, str]] = []
    if not _perpendicular(a, _EZ):
        failed.append((8, "cylinder side approach is not radial (perpendicular to the axis)"))
    if not _perpendicular(y, _EZ):
        failed.append((8, "cylinder side closing axis is not perpendicular to the cylinder axis"))
    # Closest approach-ray point to the cylinder (z) axis.
    a_xy = a[:2]
    denom = float(a_xy @ a_xy)
    if denom < 1e-12:
        failed.append((8, "cylinder side approach is parallel to the axis"))
        return _Contacts(p, p, p, y, -y, np.nan, np.nan, np.inf, None, failed)
    # The approach must point radially INWARD (toward the axis), not merely be
    # horizontal, and its ray must actually reach the tube (#101).
    if float((-p[:2]) @ a_xy) <= 0.0:
        failed.append((8, "cylinder side approach does not point toward the axis"))
    t_base = float(-(p[:2] @ a_xy) / denom)
    if t_base > L + atol:
        failed.append((5, "cylinder axis crossing is beyond finger reach"))
    t_base = min(max(t_base, 0.0), L)
    base = p + t_base * a
    roots = _line_quadric_roots(base[:2], y[:2], r)
    if roots is None:
        failed.append((5, "closing line misses the cylinder"))
        return _Contacts(base, base, base, y, -y, np.nan, np.nan, np.inf, None, failed)
    s_lo, s_hi = roots
    contact_lo = base + s_lo * y
    contact_hi = base + s_hi * y
    span = s_hi - s_lo
    z0 = base[2]
    boundary = min(z0 - 0.0, h - z0)  # distance to the nearer end
    if boundary < c - atol:
        failed.append((4, f"side grasp height {z0:.6g} within {c:.6g} of a cylinder end (margin {boundary:.6g})"))
    entry = _line_quadric_roots(p[:2], a[:2], r)
    realized_depth = np.nan if entry is None else L - max(entry[0], 0.0)
    n_lo = np.array([contact_lo[0], contact_lo[1], 0.0]) / r
    n_hi = np.array([contact_hi[0], contact_hi[1], 0.0]) / r
    return _Contacts(base, contact_lo, contact_hi, n_lo, n_hi, span, realized_depth, boundary, None, failed)


def _handle_cylinder_cap(prim: Cylinder, p, a, y, x, L, c, atol, *, top: bool) -> _Contacts:
    r, h = prim.radius, prim.height
    failed: List[Tuple[int, str]] = []
    want = -1.0 if top else 1.0  # top approaches -z, bottom approaches +z
    if not _aligned(a, want * _EZ):
        failed.append((8, f"cylinder {'top' if top else 'bottom'} approach is not along {'-z' if top else '+z'}"))
    if not _perpendicular(y, _EZ):
        failed.append((8, "cylinder cap closing axis is not perpendicular to the cylinder axis"))
    face_z = h if top else 0.0
    # Insertion below the approached face; the closing plane sits at the fingertip.
    standoff = (p[2] - face_z) if top else (face_z - p[2])
    realized_depth = L - standoff
    z0 = min(max(p[2] + want * L, 0.0), h)  # fingertip height, capped to the solid
    base = np.array([p[0], p[1], z0])
    roots = _line_quadric_roots(base[:2], y[:2], r)
    if roots is None:
        failed.append((5, "closing line misses the cylinder cross-section"))
        return _Contacts(base, base, base, y, -y, np.nan, realized_depth, np.inf, None, failed)
    if realized_depth < atol:
        failed.append((5, "fingertips do not reach below the approached cap"))
    s_lo, s_hi = roots
    contact_lo = base + s_lo * y
    contact_hi = base + s_hi * y
    span = s_hi - s_lo
    n_lo = np.array([contact_lo[0], contact_lo[1], 0.0]) / r
    n_hi = np.array([contact_hi[0], contact_hi[1], 0.0]) / r
    # Clearance-aware insertion band (#96): the fingertip must be at least c past
    # the approached cap AND at least c short of the opposite cap.
    boundary = min(realized_depth, h - realized_depth)
    if boundary < c - atol:
        failed.append(
            (4, f"cap insertion depth {realized_depth:.6g} within {c:.6g} of a cylinder end (margin {boundary:.6g})")
        )
    return _Contacts(base, contact_lo, contact_hi, n_lo, n_hi, span, realized_depth, boundary, None, failed)


def _box_axis_bounds(prim: Box, axis_index: int) -> Tuple[float, float]:
    if axis_index == 0:
        return -prim.dx / 2.0, prim.dx / 2.0
    if axis_index == 1:
        return -prim.dy / 2.0, prim.dy / 2.0
    return 0.0, prim.dz


def _match_axis(vec: np.ndarray) -> Optional[Tuple[int, float]]:
    """If ``vec`` is (near) ``±`` an object axis, return ``(index, sign)``, else ``None``."""
    for i, axis in enumerate((_EX, _EY, _EZ)):
        d = float(vec @ axis)
        if abs(d) >= np.cos(ANGLE_ATOL):  # true angular tolerance, not sqrt(2e-6) (#100)
            return i, float(np.sign(d))
    return None


def _handle_box(prim: Box, p, a, y, x, L, c, atol, *, mode: str) -> _Contacts:
    failed: List[Tuple[int, str]] = []
    a_match = _match_axis(a)
    y_match = _match_axis(y)
    if a_match is None or y_match is None:
        failed.append((8, "box grasp approach/closing axes are not aligned with object axes"))
        return _Contacts(p, p, p, y, -y, np.nan, np.nan, np.inf, None, failed)
    a_ax, a_sign = a_match
    y_ax, y_sign = y_match
    if a_ax == y_ax:
        failed.append((8, "box approach and closing axes coincide"))
        return _Contacts(p, p, p, y, -y, np.nan, np.nan, np.inf, None, failed)
    # Mode must match the approached face.
    if mode == "top" and not (a_ax == 2 and a_sign < 0):
        failed.append((8, "box top approach is not -z"))
    if mode == "bottom" and not (a_ax == 2 and a_sign > 0):
        failed.append((8, "box bottom approach is not +z"))
    if mode == "face" and a_ax == 2:
        failed.append((8, "box face approach must be along ±x or ±y, not the z faces"))

    slide_ax = ({0, 1, 2} - {a_ax, y_ax}).pop()
    lo_s, hi_s = _box_axis_bounds(prim, y_ax)  # span-axis (closing) bounds
    lo_a, hi_a = _box_axis_bounds(prim, a_ax)  # approach-axis bounds
    lo_g, hi_g = _box_axis_bounds(prim, slide_ax)  # slide-axis bounds
    span = hi_s - lo_s

    # Approached face plane along the approach axis, and realized insertion depth.
    face_val = hi_a if a_sign < 0 else lo_a  # -z approach hits the +z (upper) face, etc.
    standoff = (p[a_ax] - face_val) * (-a_sign)
    realized_depth = L - standoff
    if realized_depth < atol:
        failed.append((5, "fingertips do not reach past the approached box face"))

    # The closing plane sits at the fingertip depth; the slide coordinate is the
    # palm's. Fingertip = palm + L·approach, so along the approach axis it moves by
    # ``a_sign · L`` (approach column's component on that object axis).
    depth_coord = min(max(p[a_ax] + a_sign * L, lo_a), hi_a)
    base = np.array(p, dtype=float)
    base[a_ax] = depth_coord
    base[slide_ax] = p[slide_ax]
    base[y_ax] = (lo_s + hi_s) / 2.0  # contacts are symmetric about the span-axis centre

    contact_hi = base.copy()
    contact_hi[y_ax] = hi_s
    contact_lo = base.copy()
    contact_lo[y_ax] = lo_s
    n_hi = np.zeros(3)
    n_hi[y_ax] = 1.0
    n_lo = -n_hi
    # Order lo/hi along +y_EE so normals oppose ±y_EE consistently.
    if y_sign < 0:
        contact_lo, contact_hi = contact_hi, contact_lo
        n_lo, n_hi = n_hi, n_lo

    # Clearance-aware usable band (#96): the fingertip must clear the approached
    # face, the opposite face, and both lateral slide edges, each by at least c.
    approach_margin = realized_depth
    far_margin = (hi_a - lo_a) - realized_depth
    slide_margin = min(p[slide_ax] - lo_g, hi_g - p[slide_ax])
    boundary = min(approach_margin, far_margin, slide_margin)
    if boundary < c - atol:
        failed.append(
            (
                4,
                f"grasp within {c:.6g} of a box boundary (approach {approach_margin:.4g}, "
                f"far {far_margin:.4g}, slide {slide_margin:.4g})",
            )
        )
    if not (lo_a - atol <= depth_coord <= hi_a + atol):
        failed.append((5, "closing plane falls outside the box along the approach axis"))
    return _Contacts(base, contact_lo, contact_hi, n_lo, n_hi, span, realized_depth, boundary, None, failed)


def _handle_torus_span(prim: Torus, p, a, y, x, L, c, atol) -> _Contacts:
    R, r = prim.major, prim.minor
    failed: List[Tuple[int, str]] = []
    if not (_aligned(a, _EZ) or _aligned(a, -_EZ)):
        failed.append((8, "torus span approach is not along ±z"))
    if not _perpendicular(y, _EZ):
        failed.append((8, "torus span closing axis is not horizontal"))
    # The hand must occupy the side its approach descends from: approaching -z means
    # the palm is above the equator, and vice versa (#101).
    a_sign = float(np.sign(_axis_component(a, _EZ)))
    if a_sign * p[2] >= 0.0:
        failed.append((8, "torus span palm is not on the side its approach descends from"))
    # Reach the equator plane (z = 0), where the outer diameter is widest; the plane
    # must be strictly forward of the palm and within reach (no clamping, #101).
    t_equator = float((0.0 - p[2]) / a[2]) if abs(a[2]) > 1e-12 else -1.0
    if not (atol < t_equator <= L + atol):
        failed.append((5, f"torus equator plane depth {t_equator:.6g} not on the forward finger segment (0, {L:.6g}]"))
        return _Contacts(p, p, p, y, -y, np.nan, np.nan, np.inf, None, failed)
    base = p + t_equator * a
    roots = _line_quadric_roots(base[:2], y[:2], R + r)  # outer equator radius
    if roots is None:
        failed.append((5, "closing line misses the torus outer equator"))
        return _Contacts(base, base, base, y, -y, np.nan, np.nan, np.inf, None, failed)
    s_lo, s_hi = roots
    contact_lo = base + s_lo * y
    contact_hi = base + s_hi * y
    span = s_hi - s_lo
    # Insertion below the approached extremal tube plane (z = ±r, nearest to palm).
    standoff = abs(p[2]) - r
    realized_depth = L - standoff
    n_lo = _torus_normal(prim, contact_lo)
    n_hi = _torus_normal(prim, contact_hi)
    return _Contacts(base, contact_lo, contact_hi, n_lo, n_hi, span, realized_depth, np.inf, 0.0, failed)


def _torus_normal(prim: Torus, point: np.ndarray) -> np.ndarray:
    R = prim.major
    rho = float(np.hypot(point[0], point[1]))
    if rho < 1e-12:
        return np.array([0.0, 0.0, np.sign(point[2]) or 1.0])
    ring = np.array([point[0] / rho * R, point[1] / rho * R, 0.0])  # nearest tube-centre point
    out = point - ring
    n = np.linalg.norm(out)
    return out / n if n > 0 else out


def _handle_torus_side(prim: Torus, p, a, y, x, L, c, atol) -> _Contacts:
    """Torus side grasp at any approach minor angle ``alpha in [-pi/2, pi/2]`` (#99).

    The approach is the inward tube-surface normal at the grasp point, so
    ``a = -(cos(alpha) radial(phi) + sin(alpha) z)``. We recover ``(phi, alpha)`` from
    the concrete pose, require the approach ray to actually pass through the tube
    centre on the forward finger segment, and close across the minor circle.
    """
    R, r = prim.major, prim.minor
    failed: List[Tuple[int, str]] = []
    fail = _Contacts(p, p, p, y, -y, np.nan, np.nan, np.inf, None, failed)

    # alpha from the axial component; phi from the horizontal (or the palm when the
    # approach is vertical, alpha = ±pi/2).
    sin_alpha = float(np.clip(-a[2], -1.0, 1.0))
    alpha = float(np.arcsin(sin_alpha))
    a_xy = a[:2]
    if float(a_xy @ a_xy) > 1e-12:
        phi = float(np.arctan2(-a[1], -a[0]))
    elif float(p[:2] @ p[:2]) > 1e-12:
        phi = float(np.arctan2(p[1], p[0]))
    else:
        failed.append((8, "torus side azimuth is undefined (vertical approach over the axis)"))
        return fail
    radial = np.array([np.cos(phi), np.sin(phi), 0.0])
    normal = np.cos(alpha) * radial + sin_alpha * _EZ  # outward tube-surface normal
    # The pose approach must be the inward tube normal at (phi, alpha).
    if float(a @ (-normal)) < np.cos(1e-6):
        failed.append((8, "torus side approach is not the inward tube-surface normal at a minor angle"))
        return fail

    tube_center = R * radial
    # Require the approach ray p + t·a to pass through the tube centre, forward.
    t_center = float((tube_center - p) @ a)
    residual = float(np.linalg.norm((tube_center - p) - t_center * a))
    if residual > 10.0 * atol:
        failed.append((5, "approach ray does not pass through the torus tube centre"))
        return fail
    if not (atol < t_center <= L + atol):
        failed.append((5, f"torus tube centre depth {t_center:.6g} not on the forward finger segment (0, {L:.6g}]"))
        return fail

    # Close across the minor-circle diameter: tube_center ± r·y_EE (verified on the
    # surface by the SDF cross-check in certify).
    contact_hi = tube_center + r * y
    contact_lo = tube_center - r * y
    span = 2.0 * r
    # Realized insertion from the approached tube surface (r before the centre).
    realized_depth = L - (t_center - r)
    n_lo = _torus_normal(prim, contact_lo)
    n_hi = _torus_normal(prim, contact_hi)
    return _Contacts(tube_center, contact_lo, contact_hi, n_lo, n_hi, span, realized_depth, np.inf, alpha, failed)


# --------------------------------------------------------------------------- #
# Public witness + certify.
# --------------------------------------------------------------------------- #


@dataclass
class GraspWitness:
    """Structured geometric witness for one concrete pose (issue #67).

    ``ok`` is True iff every soundness clause holds. ``failed`` lists
    ``(clause, message)`` for each violation so a failure reports geometry, not a
    bare Boolean. All quantities are pose-derived: ``realized_depth`` is the
    insertion from the approached surface [m] (the ``GraspProvenance.depth`` frame),
    ``boundary_clearance`` is the min margin to a free edge along the object extent,
    ``minor_angle`` is the torus tube cross-section angle (torus only, else None),
    and ``contact_plane_distance`` is the palm-to-closing-plane distance.
    """

    ok: bool
    palm_clearance: float
    realized_depth: float
    boundary_clearance: float
    span: float
    contact_neg: Optional[np.ndarray]
    contact_pos: Optional[np.ndarray]
    normal_neg: Optional[np.ndarray]
    normal_pos: Optional[np.ndarray]
    minor_angle: Optional[float] = None
    contact_plane_distance: float = np.nan
    failed: List[Tuple[int, str]] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.failed is None:
            self.failed = []


def certify(
    prim,
    pose,
    *,
    finger_length: float,
    max_aperture: float,
    preshape: float,
    clearance: float,
    mode: str,
    approach: Optional[str] = None,
    finger_orientation: Optional[str] = None,
    pad_length: Optional[float] = None,
    min_encompassing_diameter: Optional[float] = None,
    grip: Optional[str] = None,
    closed_reach: Optional[Sequence[Tuple[float, float]]] = None,
) -> GraspWitness:
    """Certify the soundness clauses for ``prim`` at end-effector ``pose``.

    ``pose`` is the 4×4 end-effector transform in the primitive's reference frame
    (``E(ξ) = T_ref_tsr · xyzrpy_to_trans(ξ) · Tw_e``). Raises ``ValueError`` for a
    malformed *model* (bad dimensions/gripper/clearance or an unsupported mode,
    #95); a malformed *pose* returns a clause-1 :class:`GraspWitness` failure.

    ``mode`` selects the analytic model; ``approach``/``finger_orientation`` are
    optional structural labels checked against the realized geometry. None of these
    supply numeric evidence -- contacts are derived from the pose.

    ``pad_length`` and ``min_encompassing_diameter`` describe the hand's gripping
    surfaces (#189); ``grip`` is the declared grip, a selector like ``mode``. A hand
    without pads must declare no grip; a hand with pads must declare ``"pad"`` or
    ``"encompassing"``, and clause 9 checks where on the finger the contacts fall.
    ``closed_reach`` is the hand's declared ``((width, reach), ...)`` curve (#193): the
    pads ride with the fingertips, so clause 9 measures from the fingertips closed on the
    realized span, ``max(L, reach(span))`` past the palm; ``None`` for a prismatic jaw.
    """
    _validate_model(prim, finger_length, max_aperture, preshape, clearance, mode, approach, finger_orientation)
    pose_f, reason = _validate_pose(pose)
    if pose_f is None:
        return GraspWitness(False, np.nan, np.nan, np.nan, np.nan, None, None, None, None, failed=[(1, reason)])

    atol = length_atol(prim.scale)
    p = pose_f[:3, 3]
    a = pose_f[:3, 2]
    y = pose_f[:3, 1]
    x = pose_f[:3, 0]
    name = _primitive_name(prim)

    if name == "sphere":
        cg = _handle_sphere_surface(prim, p, a, y, x, finger_length, clearance, atol)
    elif name == "cylinder":
        if mode == "side":
            cg = _handle_cylinder_side(prim, p, a, y, x, finger_length, clearance, atol)
        else:
            cg = _handle_cylinder_cap(prim, p, a, y, x, finger_length, clearance, atol, top=(mode == "top"))
    elif name == "box":
        cg = _handle_box(prim, p, a, y, x, finger_length, clearance, atol, mode=mode)
    else:  # torus
        cg = (
            _handle_torus_span(prim, p, a, y, x, finger_length, clearance, atol)
            if mode == "span"
            else _handle_torus_side(prim, p, a, y, x, finger_length, clearance, atol)
        )

    failed: List[Tuple[int, str]] = list(cg.failed)

    # Clause 3: palm outside the solid with the requested clearance (SDF is the
    # authoritative signed distance here).
    palm_clearance = prim.sdf(p)
    if palm_clearance < clearance - atol:
        failed.append((3, f"palm clearance {palm_clearance:.6g} < requested c {clearance:.6g}"))

    span = cg.span
    contact_distance = float((cg.base - p) @ a) if np.all(np.isfinite(cg.base)) else np.nan

    if np.isfinite(span):
        # Shared accepted-witness invariants (#100, #101): the contact plane must lie
        # strictly on the forward finger segment, both contacts must lie in the
        # concrete swept finger plane p + t·z_EE + s·y_EE (no off-plane x_EE
        # component from a snapped axis), and required scalar geometry must be finite.
        if not (atol < contact_distance <= finger_length + atol):
            failed.append(
                (
                    5,
                    f"contact plane distance {contact_distance:.6g} not in forward interval (0, {finger_length:.4g}]",
                )
            )
        for label, contact in (("+", cg.contact_hi), ("-", cg.contact_lo)):
            off = float((contact - p) @ x)
            if abs(off) > 10.0 * atol:
                failed.append((5, f"analytic {label} contact leaves the posed finger plane (x_EE offset {off:.2e})"))
        if not np.isfinite(cg.realized_depth):
            failed.append((5, "realized insertion depth is not finite"))

        # Independent SDF verification of the analytic contacts and normals (#93).
        verify_atol = 10.0 * atol
        for label, contact, normal in (("+", cg.contact_hi, cg.normal_hi), ("-", cg.contact_lo, cg.normal_lo)):
            if abs(prim.sdf(contact)) > verify_atol:
                failed.append((5, f"analytic {label} contact is not on the surface (sdf={prim.sdf(contact):.2e})"))
            elif float(sdf_normal(prim, contact) @ normal) < np.cos(1e-3):
                failed.append((6, f"analytic {label} normal disagrees with the SDF gradient"))

        # Clause 6: inward contact normals oppose the closing directions.
        ang_hi = float(np.arccos(np.clip(cg.normal_hi @ y, -1.0, 1.0)))
        ang_lo = float(np.arccos(np.clip(cg.normal_lo @ (-y), -1.0, 1.0)))
        if ang_hi > ANGLE_ATOL or ang_lo > ANGLE_ATOL:
            failed.append((6, f"contact normals do not oppose closing (angles {ang_hi:.2e}, {ang_lo:.2e} rad)"))

        # Clause 2: the object must lie strictly between the OPEN JAWS, not merely
        # have a small span (#97). Span is translation-invariant; the jaws are the
        # pose-relative interval [-a/2, +a/2] along y_EE about the palm. Require both
        # contacts' signed jaw coordinates inside it.
        jaw = preshape / 2.0
        s_hi = float((cg.contact_hi - p) @ y)
        s_lo = float((cg.contact_lo - p) @ y)
        if max(s_hi, s_lo) > jaw - atol or min(s_hi, s_lo) < -jaw + atol:
            failed.append(
                (
                    2,
                    f"contacts at jaw coords [{s_lo:.6g}, {s_hi:.6g}] are not strictly inside open jaws [±{jaw:.6g}]",
                )
            )

    # Clause 2: aperture bound (independent of whether contacts were found).
    if preshape > max_aperture + atol:
        failed.append((2, f"preshape {preshape:.6g} exceeds max aperture {max_aperture:.6g}"))

    # Optional declared-selector consistency (#94, #98): recognized labels are
    # checked against the realized pose geometry (clause 8); never trusted as input.
    failed.extend(_selector_failures(name, mode, a, y, p, approach, finger_orientation))
    if np.isfinite(span):
        # The pads ride with the fingertips, which reach `gain` further once closed (#193).
        gain = 0.0
        if closed_reach is not None:
            widths, reaches = zip(*closed_reach)
            gain = max(0.0, float(np.interp(span, widths, reaches)) - finger_length)
        failed.extend(
            _grip_failures(
                name,
                mode,
                finger_length + gain - contact_distance,
                cg.realized_depth + gain,
                span,
                pad_length,
                min_encompassing_diameter,
                clearance,
                grip,
                atol,
            )
        )

    return GraspWitness(
        ok=not failed,
        palm_clearance=float(palm_clearance),
        realized_depth=float(cg.realized_depth),
        boundary_clearance=float(cg.boundary_clearance),
        span=float(span) if np.isfinite(span) else np.nan,
        contact_neg=cg.contact_lo if np.isfinite(span) else None,
        contact_pos=cg.contact_hi if np.isfinite(span) else None,
        normal_neg=cg.normal_lo if np.isfinite(span) else None,
        normal_pos=cg.normal_hi if np.isfinite(span) else None,
        minor_angle=cg.minor_angle,
        contact_plane_distance=contact_distance,
        failed=failed,
    )


def _approach_consistent(name, mode, a, p, approach) -> bool:
    """Does the declared HAND-OCCUPIED side match the pose-derived geometry (#98)?

    Checked against a shared, pose-derived local frame (palm position + axes), not
    a bare axis category (#101).
    """
    if approach in _SIGNED_AXES:
        axis, sign = _SIGNED_AXES[approach]
        # The hand occupies the -z_EE side, so the approach column points toward it,
        # and (for a centred/planar primitive) the palm lies on that side.
        if not _aligned(-a, sign * axis):
            return False
        if name == "torus" and mode == "span":
            return sign * float(p @ axis) > 0.0  # palm above/below the equator plane
        return True
    if approach == "radial":
        # Radial: perpendicular to the axis AND pointing toward it (cylinder/torus),
        # or toward the centre (sphere) -- a pose-derived local-frame predicate.
        if name in ("cylinder", "torus"):
            return _perpendicular(a, _EZ) and float((-p[:2]) @ a[:2]) > 0.0
        return float((-p) @ a) > 0.0  # sphere: toward the centre
    if approach == "tube":
        return True  # torus side: the handler verifies the inward tube normal
    return False


def _orientation_consistent(name, y, finger_orientation) -> bool:
    """Does the realized closing axis match the declared finger orientation (#98)?"""
    if finger_orientation in ("x", "y", "z"):
        return _aligned(y, {"x": _EX, "y": _EY, "z": _EZ}[finger_orientation]) or _aligned(
            -y, {"x": _EX, "y": _EY, "z": _EZ}[finger_orientation]
        )
    if finger_orientation == "diameter":
        # A diameter closes perpendicular to the primitive axis (cylinder/torus);
        # any great circle qualifies for a sphere.
        return _perpendicular(y, _EZ) if name in ("cylinder", "torus") else True
    if finger_orientation == "tangential":
        # Cylinder side closes perpendicular to the axis; the torus tangent is
        # verified by the handler + SDF contact check.
        return _perpendicular(y, _EZ) if name == "cylinder" else True
    return False


def _selector_failures(name, mode, a, y, p, approach, finger_orientation) -> List[Tuple[int, str]]:
    """Clause-8 failures for recognized declared selectors that disagree with the pose."""
    out: List[Tuple[int, str]] = []
    if approach is not None and not _approach_consistent(name, mode, a, p, approach):
        out.append((8, f"declared approach {approach!r} is inconsistent with the pose"))
    if finger_orientation is not None and not _orientation_consistent(name, y, finger_orientation):
        out.append((8, f"declared finger_orientation {finger_orientation!r} is inconsistent with the pose"))
    return out


# Grasps whose contacts lie on one closing line crossing the object (the axis plane, the
# sphere's centre, the tube centre, the torus equator), against grasps whose fingers
# grip a face from the fingertip back to the insertion depth (caps and box faces).
_LINE_CONTACT = {("sphere", "surface"), ("cylinder", "side"), ("torus", "side"), ("torus", "span")}
_ENCOMPASSABLE = {("sphere", "surface"), ("cylinder", "side")}


def _grip_failures(
    name, mode, behind_tip, realized_depth, span, pad_length, min_diameter, c, grip, atol
) -> List[Tuple[int, str]]:
    """Clause 8/9 failures for where on the finger the contacts fall (#189).

    ``behind_tip`` is the closing line's distance behind the closed fingertip, from the pose;
    ``realized_depth`` is a face contact's extent back from it.
    """
    if pad_length is None:
        return [(8, f"declared grip {grip!r} but the hand has no gripping surfaces")] if grip else []
    if grip not in ("pad", "encompassing"):
        return [(8, f"a hand with pads must declare grip 'pad' or 'encompassing', got {grip!r}")]
    line = (name, mode) in _LINE_CONTACT
    if grip == "encompassing":
        if (name, mode) not in _ENCOMPASSABLE or min_diameter is None:
            return [(8, f"({name}, {mode}) cannot be encompassed by this hand")]
        out = []
        if behind_tip < pad_length + c - atol:
            out.append((9, f"encompassing contacts {behind_tip:.6g} behind the tip are not {c:.6g} behind the pads"))
        if span < min_diameter - atol:
            out.append((9, f"object {span:.6g} across is below the minimum encompassing {min_diameter:.6g}"))
        return out
    # Pad grip: a line contact sits a clearance inside both pad ends; a face contact runs
    # from the fingertip back to the insertion depth, which must stay a clearance on the pads.
    if line and behind_tip < c - atol:
        return [(9, f"contacts {behind_tip:.6g} behind the tip are within {c:.6g} of the pad's tip end")]
    inner = behind_tip if line else realized_depth
    if inner > pad_length - c + atol:
        return [(9, f"contacts reach {inner:.6g} behind the tip, past the pads ({pad_length:.6g}) less {c:.6g}")]
    return []


def pose_from(palm, approach, close) -> np.ndarray:
    """Build an SE(3) EE pose from a palm point and approach/close axes.

    ``approach`` → ``z_EE`` and ``close`` → ``y_EE`` (re-orthonormalized against the
    approach), with ``x_EE = y_EE × z_EE``. Test helper for hand-derived poses.
    """
    z = np.asarray(approach, dtype=float)
    z = z / np.linalg.norm(z)
    yv = np.asarray(close, dtype=float)
    yv = yv - (yv @ z) * z
    yv = yv / np.linalg.norm(yv)
    xv = np.cross(yv, z)
    T = np.eye(4)
    T[:3, 0] = xv
    T[:3, 1] = yv
    T[:3, 2] = z
    T[:3, 3] = np.asarray(palm, dtype=float)
    return T
