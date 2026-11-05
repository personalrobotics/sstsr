# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""StablePlacer: generate stable placement TSRs for objects on a flat surface.

The placement contract is stated in ``docs/ARCHITECTURE.md``. In short, for **every**
pose a returned region admits, the object rests on the surface without penetrating it,
its centre of mass projects inside the contact patch, and the whole object stays within
the surface footprint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..core.utils import length_atol
from ..placement_provenance import PlacementProvenance
from ..template import TSRTemplate
from ._stable_poses import _inward_distance, _plane_basis, _rotation_to_align, stable_poses_mesh
from .shape import Box, Cylinder, Shape, Sphere

logger = logging.getLogger(__name__)

_NEG_Z = np.array([0.0, 0.0, -1.0])


def _check_positive_finite(name: str, value: float) -> float:
    """Validate a finite, strictly positive scalar, as the grasp factories do (#154)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return float(value)


def _check_finite(name: str, value: float) -> float:
    """Validate a finite scalar of any sign (#154)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class _Variant:
    """One resting orientation of a primitive, with everything the contract needs.

    The tipping angle is *derived* from its two components rather than stored: the angle
    the object must rotate about the nearest support edge before its centre of mass
    passes over it is ``arctan(support_margin / com_height)``, so keeping the components
    makes that identity structurally true for every primitive and lets the record report
    all three consistently (#160).

    ``support_margin`` is the in-plane distance from the projected centre of mass to the
    nearest support edge; ``com_height`` is the lever arm, the perpendicular distance
    from the centre of mass to the resting plane. It is **not** ``origin_height``, which
    positions the object and coincides only when the origin is the centre of mass.
    ``footprint`` is the largest horizontal distance from the object frame origin to any
    point of the resting object, which is what the free yaw sweeps.
    """

    label: str
    normal: np.ndarray  # outward object-frame normal of the resting face
    origin_height: float
    footprint: float
    support_margin: float
    com_height: float

    @property
    def margin(self) -> float:
        """The tipping angle. Resolves ``_tipping_angle`` at call time on purpose, so a
        corrupted helper is still visible through this property."""
        return _tipping_angle(self.support_margin, self.com_height)


def _tipping_angle(lever: float, height: float) -> float:
    """``arctan(lever / height)``: the tipping angle about a support edge ``lever``
    metres from the centre of mass's projection, with the mass ``height`` above the
    surface. The same quantity :mod:`_stable_poses` computes for a mesh face."""
    return float(np.arctan2(lever, height))


def _placement_record(
    primitive: str,
    *,
    face_normal,
    support_margin: float,
    com_height: float,
    footprint_radius: float,
    face_index=None,
    face_count=None,
    facet_count=None,
) -> PlacementProvenance:
    """Build one placement provenance record (#160).

    The single conversion point on the generator side: every scalar here starts life as
    a NumPy value, and a ``np.float64`` passes ``isinstance(x, float)`` while serializing
    to a tag ``yaml.safe_load`` refuses -- so an un-canonicalised record round-trips
    through a dict and then vanishes on disk. ``int()`` matters for the opposite reason:
    ``np.int64`` is *not* an ``int``, so it would raise.

    ``equilibrium`` is derived here, once. A zero support margin means the contact cannot
    resist tipping at all -- which for a sphere is neutral (it rolls), and is why the
    distinction is recorded rather than left to be guessed from a zero angle.
    """
    support_margin = float(support_margin)
    return PlacementProvenance(
        primitive=primitive,
        support_margin=support_margin,
        com_height=float(com_height),
        footprint_radius=float(footprint_radius),
        equilibrium="stable" if support_margin > 0.0 else "neutral",
        face_normal=None if face_normal is None else tuple(float(c) for c in face_normal),
        face_index=None if face_index is None else int(face_index),
        face_count=None if face_count is None else int(face_count),
        facet_count=None if facet_count is None else int(facet_count),
    )


def _check_finite_array(name: str, values: np.ndarray) -> np.ndarray:
    """Validate that every entry of an array is finite (#154)."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite, got {np.count_nonzero(~np.isfinite(values))} non-finite entry(s)")
    return values


class StablePlacer:
    """Generate stable placement TSRs for objects on a flat surface.

    Frame convention:
        Surface z points up; surface origin at the centre of the surface.
        The object frame is the caller's own frame: its origin is the point the
        returned poses place, which is the geometric centre for the primitives below
        and an arbitrary point for :meth:`place_mesh`.

    Surface extents:
        ``table_x`` and ``table_y`` bound the **object**, not just its origin. The
        sliding bounds are inset by the object's worst-case footprint over the free
        yaw, so every admitted pose keeps the whole object on the surface (#150). An
        object too large for the surface is a valid request with an empty feasible
        set, and the factory returns ``[]`` with the reason logged at ``DEBUG``.

    Args:
        table_x: Surface half-extent along x (m).
        table_y: Surface half-extent along y (m).
        reference: Reference frame name (default ``"table"``).

    Example::

        placer    = StablePlacer(table_x=0.3, table_y=0.2)
        templates = placer.place_cylinder(cylinder_radius=0.04,
                                          cylinder_height=0.12,
                                          subject="mug")
        tsr  = templates[0].instantiate(surface_pose)
        pose = tsr.sample()
    """

    def __init__(self, table_x: float, table_y: float, reference: str = "table"):
        self.table_x = _check_positive_finite("table_x", table_x)
        self.table_y = _check_positive_finite("table_y", table_y)
        self.reference = reference

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bw(self, footprint_radius: float, roll_range=None, pitch_range=None) -> np.ndarray:
        """Build the 6×2 ``Bw``: inset xy slide, z fixed, yaw free.

        ``footprint_radius`` is the largest horizontal distance from the object frame
        origin to any point of the resting object. Because yaw is free, the footprint
        sweeps a disc of that radius about the origin, so insetting each axis by it is
        exactly the condition that the object stays on the surface at every yaw (#150).
        """
        half_x = max(self.table_x - footprint_radius, 0.0)
        half_y = max(self.table_y - footprint_radius, 0.0)
        bw = np.array(
            [
                [-half_x, half_x],
                [-half_y, half_y],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [-np.pi, np.pi],
            ]
        )
        if roll_range is not None:
            bw[3] = roll_range
        if pitch_range is not None:
            bw[4] = pitch_range
        return bw

    def _fits(self, footprint_radius: float, scale: float) -> bool:
        """Whether a footprint of this radius fits on the surface at every yaw.

        Equality is feasible, as everywhere else in the library: a footprint exactly
        the size of the surface leaves the single pose at the centre, which is an exact
        interval of zero width rather than an empty one.
        """
        atol = length_atol(scale)
        return footprint_radius <= self.table_x + atol and footprint_radius <= self.table_y + atol

    def _log_empty(self, method: str, reason: str, context: str) -> List[TSRTemplate]:
        """Report an empty feasible set with a machine-readable reason (#150).

        The factory contract in ``docs/ARCHITECTURE.md``: an exception reports an
        invalid request, an empty list reports a valid request with no feasible pose.
        """
        logger.debug("%s.%s: empty feasible set (%s; %s)", type(self).__name__, method, reason, context)
        return []

    def _template(
        self,
        name,
        description,
        variant,
        R,
        origin_height,
        Bw,
        subject,
        stability_margin=None,
        provenance=None,
    ) -> TSRTemplate:
        """Assemble one template from a resting rotation and the origin's resting height.

        The height goes in ``T_ref_tsr``, **not** in ``Tw_e``: a pose is
        ``T_ref_tsr @ xyzrpy_to_trans(ξ) @ Tw_e``, so any height carried by ``Tw_e``
        would be rotated by ``ξ``'s roll and pitch and the object would sink (#149).
        With the height ahead of ``ξ`` and ``Tw_e`` a pure rotation, the resting height
        is invariant under every rotation ``Bw`` admits, so freeing roll or pitch
        rotates the object in place, as a tilt tolerance should.
        """
        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = _check_finite("origin_height", origin_height)
        Tw_e = np.eye(4)
        Tw_e[:3, :3] = R
        return TSRTemplate(
            T_ref_tsr=T_ref_tsr,
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

    def _emit(
        self,
        method: str,
        primitive: str,
        subject: str,
        variants: List["_Variant"],
        min_margin_deg: float,
        scale: float,
        name: Callable[[str], str],
        description: Callable[[str], str],
    ) -> List[TSRTemplate]:
        """Filter a primitive's resting variants and build their templates (#156).

        One policy for every primitive: drop the variants below the requested tipping
        angle, then those whose footprint exceeds the surface, and report which of the
        two emptied the result. Analytic margins make the primitives as self-describing
        as :meth:`place_mesh`, so ``min_margin_deg`` means the same thing everywhere.
        """
        min_margin_rad = np.radians(_check_finite("min_margin_deg", min_margin_deg))
        stable = [v for v in variants if v.margin >= min_margin_rad]
        if not stable:
            best = max(np.degrees(v.margin) for v in variants)
            return self._log_empty(
                method,
                "below_min_margin",
                f"most stable variant is {best:.4g}°, below the requested {min_margin_deg:.4g}°",
            )

        usable = [v for v in stable if self._fits(v.footprint, scale)]
        if not usable:
            smallest = min(v.footprint for v in stable)
            return self._log_empty(
                method,
                "exceeds_surface",
                f"smallest footprint radius {smallest:.4g} m exceeds surface half-extent "
                f"min({self.table_x:.4g}, {self.table_y:.4g}) m",
            )
        if len(usable) < len(stable):
            logger.debug(
                "%s.%s: %d of %d variants dropped (exceeds_surface)",
                type(self).__name__,
                method,
                len(stable) - len(usable),
                len(stable),
            )

        return [
            self._template(
                name=name(v.label),
                description=description(v.label),
                variant=v.label,
                R=_rotation_to_align(v.normal, _NEG_Z),
                origin_height=v.origin_height,
                Bw=self._bw(v.footprint),
                subject=subject,
                stability_margin=v.margin,
                provenance=_placement_record(
                    primitive,
                    face_normal=v.normal,
                    support_margin=v.support_margin,
                    com_height=v.com_height,
                    footprint_radius=v.footprint,
                ),
            )
            for v in usable
        ]

    # ------------------------------------------------------------------
    # Primitive placement methods
    # ------------------------------------------------------------------

    def place_cylinder(
        self,
        cylinder_radius: float,
        cylinder_height: float,
        subject: str = "object",
        min_margin_deg: float = 0.0,
    ) -> List[TSRTemplate]:
        """Return 2 placement templates: cylinder on each circular face.

        Object frame: origin at center, z = cylinder axis pointing up.
        Sideways (on the curved surface) is not stable and not returned.

        Both caps are returned because which cap faces the surface is caller-visible in
        ``variant`` and semantically distinct for a labelled object (a mug's opening up
        or down), even though the two occupy the same volume.

        Args:
            cylinder_radius: Cylinder radius (m).
            cylinder_height: Cylinder height (m).
            subject: Name of the object frame.
            min_margin_deg: Discard variants whose tipping angle is below this
                            threshold (degrees), as for :meth:`place_mesh`.

        Returns:
            Two templates, or ``[]`` if the cylinder cannot fit on the surface or is
            less stable than requested.
        """
        radius = _check_positive_finite("cylinder_radius", cylinder_radius)
        height = _check_positive_finite("cylinder_height", cylinder_height)

        # Resting on a cap the axis is vertical, so the footprint is the cap itself and
        # the support polygon is that same disc: the cylinder tips about a tangent to the
        # rim, one radius from its centre, with the mass half its height up.
        variants = [
            _Variant(
                label=label,
                normal=np.array([0.0, 0.0, sign]),
                origin_height=height / 2.0,
                footprint=radius,
                support_margin=radius,
                com_height=height / 2.0,
            )
            for sign, label in ((-1.0, "-z"), (+1.0, "+z"))
        ]
        return self._emit(
            "place_cylinder",
            "cylinder",
            subject,
            variants,
            min_margin_deg,
            scale=max(2.0 * radius, height),
            name=lambda label: f"Place cylinder {label}-face down ({subject} on {self.reference})",
            description=lambda label: (
                f"Cylinder (r={radius:.3f} m, h={height:.3f} m) "
                f"resting on {label} face on {self.reference}. Yaw free."
            ),
        )

    def place_box(
        self,
        lx: float,
        ly: float,
        lz: float,
        subject: str = "object",
        min_margin_deg: float = 0.0,
    ) -> List[TSRTemplate]:
        """Return 6 placement templates: one for each face of the box.

        Object frame: origin at center, axes aligned with box extents.
        All 6 faces are returned because opposite faces are semantically
        distinct (e.g. front vs back of a cereal box).

        Args:
            lx: Box x-extent (m).
            ly: Box y-extent (m).
            lz: Box z-extent (m).
            subject: Name of the object frame.
            min_margin_deg: Discard faces whose tipping angle is below this threshold
                            (degrees), as for :meth:`place_mesh`. A needle standing on
                            its end is in equilibrium but barely; this is how a caller
                            rejects that.

        Returns:
            Up to six templates: the faces that fit on the surface and are at least as
            stable as requested. ``[]`` if none is.
        """
        lx = _check_positive_finite("lx", lx)
        ly = _check_positive_finite("ly", ly)
        lz = _check_positive_finite("lz", lz)

        # Each face is listed with its outward normal pointing toward the surface. The
        # footprint radius is the half-diagonal of the resting face, which is what the
        # free yaw sweeps -- an axis-aligned half-extent would still overhang (#150).
        # The box tips about the nearest edge of that face, so the lever is the smaller
        # of its two half-extents and the mass sits half the third extent up.
        faces = [
            (np.array([0.0, 0.0, -1.0]), "-z", lz, lx, ly),
            (np.array([0.0, 0.0, +1.0]), "+z", lz, lx, ly),
            (np.array([0.0, -1.0, 0.0]), "-y", ly, lx, lz),
            (np.array([0.0, +1.0, 0.0]), "+y", ly, lx, lz),
            (np.array([-1.0, 0.0, 0.0]), "-x", lx, ly, lz),
            (np.array([+1.0, 0.0, 0.0]), "+x", lx, ly, lz),
        ]
        variants = [
            _Variant(
                label=label,
                normal=normal,
                origin_height=up / 2.0,
                footprint=np.hypot(a, b) / 2.0,
                support_margin=min(a, b) / 2.0,
                com_height=up / 2.0,
            )
            for normal, label, up, a, b in faces
        ]
        return self._emit(
            "place_box",
            "box",
            subject,
            variants,
            min_margin_deg,
            scale=max(lx, ly, lz),
            name=lambda label: f"Place box {label}-face down ({subject} on {self.reference})",
            description=lambda label: (
                f"Box ({lx:.3f}×{ly:.3f}×{lz:.3f} m) resting on {label} face on {self.reference}. Yaw free."
            ),
        )

    def place_sphere(
        self,
        radius: float,
        subject: str = "object",
        min_margin_deg: float = 0.0,
    ) -> List[TSRTemplate]:
        """Return one placement template: sphere on a flat surface.

        Every orientation is equally stable so roll and pitch are also free. The centre
        stays at ``z = radius`` for every admitted orientation, which the construction
        in :meth:`_template` guarantees structurally rather than only at zero tilt
        (#149).

        The reported ``stability_margin`` is ``0``: a sphere touches at one point
        directly beneath its centre, so it is in *neutral* equilibrium — it never tips,
        it rolls. That is a different thing from the knife-edge equilibrium a mesh face
        with a zero margin describes, and it is why the sphere survives the default
        ``min_margin_deg`` of 0 while any positive threshold rejects it.

        Args:
            radius: Sphere radius (m).
            subject: Name of the object frame.
            min_margin_deg: Discard the template if this exceeds 0, since a sphere's
                            tipping angle is 0.

        Returns:
            One template, or ``[]`` if the sphere cannot fit on the surface or a positive
            margin was required.
        """
        radius = _check_positive_finite("radius", radius)
        if np.radians(_check_finite("min_margin_deg", min_margin_deg)) > 0.0:
            return self._log_empty(
                "place_sphere",
                "below_min_margin",
                f"a sphere is neutrally stable (tipping angle 0°), below the requested {min_margin_deg:.4g}°",
            )
        if not self._fits(radius, 2.0 * radius):
            return self._log_empty(
                "place_sphere",
                "exceeds_surface",
                f"footprint radius {radius:.4g} m exceeds surface half-extent "
                f"min({self.table_x:.4g}, {self.table_y:.4g}) m",
            )

        return [
            self._template(
                name=f"Place sphere ({subject} on {self.reference})",
                description=(f"Sphere (r={radius:.3f} m) on {self.reference}. All orientations free."),
                variant="upright",
                R=np.eye(3),
                origin_height=radius,
                Bw=self._bw(
                    radius,
                    roll_range=np.array([-np.pi, np.pi]),
                    pitch_range=np.array([-np.pi, np.pi]),
                ),
                subject=subject,
                stability_margin=0.0,
                provenance=_placement_record(
                    "sphere",
                    face_normal=None,  # no distinguished face: every orientation rests alike
                    support_margin=0.0,  # point contact -> neutral, it rolls rather than tips
                    com_height=radius,
                    footprint_radius=radius,
                ),
            )
        ]

    def place_torus(
        self,
        major_radius: float,
        minor_radius: float,
        subject: str = "object",
        min_margin_deg: float = 0.0,
    ) -> List[TSRTemplate]:
        """Return 2 placement templates: torus flat on surface, each side down.

        Object frame: origin at center, z = torus symmetry axis pointing up.
        The torus rests on the bottom of the tube ring at z = -minor_radius.

        As for the cylinder's two caps, both sides are returned because which side faces
        the surface is caller-visible in ``variant`` and semantically distinct for a
        labelled object, even though a bare solid torus is invariant under the flip
        that separates them (#155).

        Args:
            major_radius: Distance from torus center to tube center (m).
            minor_radius: Tube radius (m).
            subject: Name of the object frame.
            min_margin_deg: Discard variants whose tipping angle is below this
                            threshold (degrees), as for :meth:`place_mesh`.

        Returns:
            Two templates, or ``[]`` if the torus cannot fit on the surface or is less
            stable than requested.
        """
        major_radius = _check_positive_finite("major_radius", major_radius)
        minor_radius = _check_positive_finite("minor_radius", minor_radius)
        if minor_radius >= major_radius:
            raise ValueError("minor_radius must be less than major_radius")

        # Lying flat, the torus touches along a circle of radius ``major_radius``, whose
        # convex hull is that disc: it tips about a tangent to that circle, with the mass
        # one tube radius up. The footprint is the ring's outer equator.
        footprint = major_radius + minor_radius
        variants = [
            _Variant(
                label=label,
                normal=np.array([0.0, 0.0, sign]),
                origin_height=minor_radius,
                footprint=footprint,
                support_margin=major_radius,
                com_height=minor_radius,
            )
            for sign, label in ((-1.0, "-z"), (+1.0, "+z"))
        ]
        return self._emit(
            "place_torus",
            "torus",
            subject,
            variants,
            min_margin_deg,
            scale=2.0 * footprint,
            name=lambda label: f"Place torus {label}-face down ({subject} on {self.reference})",
            description=lambda label: (
                f"Torus (R={major_radius:.3f} m, r={minor_radius:.3f} m) "
                f"flat, {label} face down on {self.reference}. Yaw free."
            ),
        )

    def place_mesh(
        self,
        vertices: np.ndarray,
        com: np.ndarray,
        subject: str = "object",
        min_margin_deg: float = 0.0,
    ) -> List[TSRTemplate]:
        """Return one template per stable resting face of the mesh.

        Uses the convex hull of ``vertices`` for stable-pose detection.
        Works for non-convex objects: the support polygon is the convex
        hull of the contact points.  Results are sorted by descending
        stability margin (most stable face first).

        Args:
            vertices:       (N, 3) array of object vertices in the object frame.
            com:            (3,) center of mass in the same frame.
            subject:        Name of the object frame.
            min_margin_deg: Discard faces whose stability margin is below this
                            threshold (degrees). Default 0 returns all stable faces.

        Returns:
            One template per stable face that fits on the surface, or ``[]`` when no
            face is both stable and small enough, with the reason logged at ``DEBUG``.
        """
        vertices = np.asarray(vertices, dtype=float)
        com = np.asarray(com, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if com.shape != (3,):
            raise ValueError("com must be a length-3 array")
        _check_finite_array("vertices", vertices)
        _check_finite_array("com", com)
        _check_finite("min_margin_deg", min_margin_deg)

        scale = float(np.ptp(vertices, axis=0).max()) if len(vertices) else 0.0
        min_margin_rad = float(np.radians(min_margin_deg))
        # descending stability margin: place_mesh promises most-stable-first
        stable = sorted(stable_poses_mesh(vertices, com), key=lambda f: -f.stability_margin)
        poses = [f for f in stable if f.stability_margin >= min_margin_rad]
        if not poses:
            return self._log_empty(
                "place_mesh",
                "no_stable_face" if not stable else "below_min_margin",
                f"{len(stable)} stable face(s), none with margin >= {min_margin_deg:.4g}°",
            )

        # The footprint is measured from the object frame origin, which Bw slides, and
        # not from the centroid: place_mesh accepts an arbitrary origin (#150).
        usable = []
        for face in poses:
            footprint = float(np.linalg.norm((vertices @ face.R.T)[:, :2], axis=1).max())
            if self._fits(footprint, scale):
                usable.append((face, footprint))
        if not usable:
            return self._log_empty(
                "place_mesh",
                "exceeds_surface",
                f"all {len(poses)} stable face(s) exceed surface half-extent "
                f"min({self.table_x:.4g}, {self.table_y:.4g}) m",
            )
        if len(usable) < len(poses):
            logger.debug(
                "%s.place_mesh: %d of %d stable faces dropped (exceeds_surface)",
                type(self).__name__,
                len(poses) - len(usable),
                len(poses),
            )

        templates = []
        for idx, (face, footprint) in enumerate(usable):
            deg = float(np.degrees(face.stability_margin))
            templates.append(
                self._template(
                    name=(
                        f"Place mesh face {idx + 1}/{len(usable)} ({subject} on {self.reference}, margin {deg:.1f}°)"
                    ),
                    description=(
                        f"Mesh resting on stable face {idx + 1} of {len(usable)} "
                        f"(stability margin {deg:.1f}°) on {self.reference}."
                    ),
                    variant=f"face-{idx + 1}",
                    R=face.R,
                    origin_height=face.origin_height,
                    Bw=self._bw(footprint),
                    subject=subject,
                    stability_margin=float(face.stability_margin),
                    provenance=_placement_record(
                        "mesh",
                        face_normal=face.face_normal,
                        support_margin=face.support_margin,
                        com_height=face.com_height,
                        footprint_radius=footprint,
                        face_index=idx,
                        face_count=len(usable),
                        facet_count=face.facet_count,
                    ),
                )
            )
        return templates

    # ------------------------------------------------------------------
    # An object as a whole: whatever parts it is made of
    # ------------------------------------------------------------------

    def place(self, shape: Shape, subject: str = "object", min_margin_deg: float = 0.0) -> List[TSRTemplate]:
        """Return one template per stable rest of an object described by its parts (#201).

        The caller describes the object, typically its collision model, as a
        :class:`~tsr.placement.Shape`, and need not know which method places it:

        * **One box, cylinder or sphere** whose centre of mass is its centre is placed
          exactly by :meth:`place_box`, :meth:`place_cylinder` or :meth:`place_sphere`,
          wherever the part sits in the object frame. A sphere off the object origin keeps
          the object's own roll and pitch, since rolling it would move that origin.
        * **Anything else** rests on a face of its convex hull. Candidate faces come from
          points on the parts' surfaces, and each is kept only if the **true** parts
          support it: the resting height is the parts' exact support, and the contact
          patch is the parts' own points on that plane, a cylinder's cap as its inscribed
          polygon. A curved part sampled into points would otherwise invent narrow faces
          where the real object touches along a line or at a point and rolls; those are
          not rests, so a lying cylinder or a capsule contributes none. A rest whose
          contact meets a curved part at a single tangent point between samples can be
          missed; none is invented.

        Every template frames the object origin, with ``Tw_e`` a pure rotation and the
        footprint taken about that origin. Results from the general path are sorted by
        descending stability margin, as :meth:`place_mesh`'s are.

        Args:
            shape: The object's parts and centre of mass, in the object frame.
            subject: Name of the object frame.
            min_margin_deg: Discard rests whose tipping angle is below this (degrees).

        Returns:
            The templates, or ``[]`` when no rest is stable enough and fits on the
            surface, with the reason logged at ``DEBUG``.
        """
        if not isinstance(shape, Shape):
            raise ValueError(f"shape must be a tsr.placement.Shape, got {type(shape).__name__}")
        _check_finite("min_margin_deg", min_margin_deg)
        surface = shape.surface()
        scale = float(np.ptp(surface, axis=0).max())
        part = shape.parts[0] if len(shape.parts) == 1 else None
        if isinstance(part, (Box, Cylinder, Sphere)) and np.linalg.norm(shape.com - part.centre) <= length_atol(scale):
            return self._place_part(part, subject, min_margin_deg, scale)
        return self._place_parts(shape, surface, subject, min_margin_deg, scale)

    def _place_part(self, part, subject: str, min_margin_deg: float, scale: float) -> List[TSRTemplate]:
        """A single primitive, placed by its exact factory in its own frame and re-framed
        at the object origin."""
        if isinstance(part, Box):
            primitive = "box"
            templates = self.place_box(*part.size, subject=subject, min_margin_deg=min_margin_deg)
        elif isinstance(part, Cylinder):
            primitive = "cylinder"
            templates = self.place_cylinder(part.radius, part.height, subject=subject, min_margin_deg=min_margin_deg)
        else:
            primitive = "sphere"
            templates = self.place_sphere(part.radius, subject=subject, min_margin_deg=min_margin_deg)
        R_object_part, t_object_part = part.pose[:3, :3], part.pose[:3, 3]
        centred = np.linalg.norm(t_object_part) <= length_atol(scale)
        out = []
        for t in templates:
            # The object's rotation once the part rests. A sphere's own rotation means
            # nothing, so a sphere leaves the object as it is framed.
            R = np.eye(3) if primitive == "sphere" else t.Tw_e[:3, :3] @ R_object_part.T
            height = float(t.T_ref_tsr[2, 3] - (R @ t_object_part)[2])
            footprint = part.reach(R)
            if not self._fits(footprint, scale):
                continue
            record = t.provenance
            normal = None if record.face_normal is None else R_object_part @ np.asarray(record.face_normal)
            free = np.array([-np.pi, np.pi]) if primitive == "sphere" and centred else None
            out.append(
                self._template(
                    name=t.name,
                    description=t.description,
                    variant=t.variant if normal is None else _face_label(normal, f"{primitive} {t.variant}"),
                    R=R,
                    origin_height=height,
                    Bw=self._bw(footprint, roll_range=free, pitch_range=free),
                    subject=subject,
                    stability_margin=t.stability_margin,
                    provenance=_placement_record(
                        primitive,
                        face_normal=normal,
                        support_margin=record.support_margin,
                        com_height=record.com_height,
                        footprint_radius=footprint,
                    ),
                )
            )
        if templates and not out:
            return self._log_empty("place", "exceeds_surface", "no rest's footprint about the object origin fits")
        return out

    def _place_parts(
        self, shape: Shape, surface: np.ndarray, subject: str, min_margin_deg: float, scale: float
    ) -> List[TSRTemplate]:
        """A compound, or a part placed off its centre of mass: rests on its convex hull's
        faces that the true parts support."""
        atol = length_atol(scale)
        com = shape.com
        candidates = list(stable_poses_mesh(surface, com))
        rests = []
        for face in candidates:
            n = np.asarray(face.face_normal, dtype=float)
            h = shape.support(n)
            com_height = h - float(np.dot(com, n))
            if com_height <= atol:
                continue
            contact = np.vstack([p.contact(n, h, atol) for p in shape.parts])
            basis = _plane_basis(n)
            points = contact @ basis
            try:
                polygon = points[ConvexHull(points).vertices]
            except (QhullError, ValueError):
                continue  # a line or point of contact: the object rolls, it does not rest
            d_min = _inward_distance(com @ basis, polygon)
            if d_min <= atol:
                continue
            rests.append((face, n, h, com_height, d_min, float(np.arctan2(d_min, com_height))))
        stable = [r for r in rests if r[5] >= np.radians(min_margin_deg)]
        if not stable:
            return self._log_empty(
                "place",
                "no_stable_face" if not rests else "below_min_margin",
                f"{len(rests)} stable rest(s), none with margin >= {min_margin_deg:.4g}°",
            )
        stable.sort(key=lambda r: -r[5])
        usable = []
        for face, n, h, com_height, d_min, margin in stable:
            R = _rotation_to_align(n, _NEG_Z)
            footprint = shape.reach(R)
            if self._fits(footprint, scale):
                usable.append((face, n, h, com_height, d_min, margin, R, footprint))
        if not usable:
            return self._log_empty(
                "place",
                "exceeds_surface",
                f"all {len(stable)} rest(s) exceed surface half-extent min({self.table_x:.4g}, {self.table_y:.4g}) m",
            )
        templates = []
        for index, (face, n, h, com_height, d_min, margin, R, footprint) in enumerate(usable):
            deg = float(np.degrees(margin))
            templates.append(
                self._template(
                    name=f"Place rest {index + 1}/{len(usable)} ({subject} on {self.reference}, margin {deg:.1f}°)",
                    description=(
                        f"Object resting on hull face {index + 1} of {len(usable)} "
                        f"(stability margin {deg:.1f}°) on {self.reference}."
                    ),
                    variant=f"face-{index + 1}",
                    R=R,
                    origin_height=h,
                    Bw=self._bw(footprint),
                    subject=subject,
                    stability_margin=margin,
                    provenance=_placement_record(
                        "compound",
                        face_normal=n,
                        support_margin=d_min,
                        com_height=com_height,
                        footprint_radius=footprint,
                        face_index=index,
                        face_count=len(usable),
                        facet_count=face.facet_count,
                    ),
                )
            )
        return templates


_AXIS_LABELS = {(0, -1.0): "-x", (0, 1.0): "+x", (1, -1.0): "-y", (1, 1.0): "+y", (2, -1.0): "-z", (2, 1.0): "+z"}


def _face_label(normal: np.ndarray, part_face: str) -> str:
    """The face a variant names, since the label is a claim about the face: ``"-z"`` and
    the like for an object-frame face along an axis, and otherwise the part's own face
    (``"box -z"``), which stays the same face whatever else is filtered out."""
    axis = int(np.argmax(np.abs(normal)))
    if abs(abs(normal[axis]) - 1.0) <= 1e-9:
        return _AXIS_LABELS[(axis, float(np.sign(normal[axis])))]
    return part_face
