# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""GripperBase: abstract base class for gripper hand models."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from tsr.core.utils import length_atol
from tsr.template import TSRTemplate

logger = logging.getLogger(__name__)


class GripperBase(ABC):
    """Abstract base class for gripper hand models.

    A hand model generates TSRTemplates from object geometry (grasp_* methods)
    Visualization lives in :mod:`tsr.viser` and never in a hand.

    **Gripper frame convention** (canonical for this library):

        z = approach direction  (toward object surface)
        y = finger opening direction
        x = palm normal         (right-hand rule: x = y × z)

    Poses sampled from grasp TSRs are **pre-grasp configurations**: the hand
    is open at ``preshape``, positioned so that closing the fingers achieves
    stable contact with the object. The TSR guarantees this geometrically —
    it constrains where the hand must be before closing, but does not
    explicitly verify force closure.

    The geometric meaning of a returned template — what "sound" means, the
    per-field semantics of ``TSRTemplate.provenance``, tolerances, and the
    soundness/coverage/distribution/embodied distinctions — is specified by the
    **geometric grasp contract** in ``docs/ARCHITECTURE.md``.

    To convert sampled poses to another convention, apply a fixed rotation::

        # AnyGrasp / GraspNet uses x=approach:
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        anygrasp_pose = tsr_pose @ np.block([[R, np.zeros((3,1))],
                                             [np.zeros((1,3)), [[1]]]])

    **Object coordinate conventions:**

    All grasp_cylinder_* methods expect the reference object's frame to be
    placed with:

        Cylinder::

              ^ +z
              |
            --+-- z = cylinder_height  (top face)
            | | |
            | | |  ← axis along +z
            | | |
            --+-- z = 0               (bottom face, at origin)

        Box::

              ^ +z
              |
            +-+--------+  z = box_z  (top face)
            | |        |
            | +--------+  ← centered in x, y
            | |        |
            +-+--------+  z = 0      (bottom face, at origin)
              ·
            x ∈ [-box_x/2, box_x/2]
            y ∈ [-box_y/2, box_y/2]
            z ∈ [0,        box_z   ]

        Sphere::

                  * * *
                *       *
               *    O    *   ← center at origin (0, 0, 0)
                *       *
                  * * *
            radius = object_radius

        Torus::

                 ___
               /     \\
              |   O   |     <- center at origin, axis along +z
               \\     /
                 ‾‾‾
            torus_radius R = distance from center to tube center
            tube_radius  r = tube cross-section radius
            Tube center at azimuth θ: (R·cos θ, R·sin θ, 0)
            Outer radius: R + r,  Inner radius: R - r

        The reference object pose (T_world_object) transforms this frame into
        the world. E.g., a box sitting upright on a table at position p::

            T_world_box = np.eye(4)
            T_world_box[:3, 3] = p        # bottom-center of box at p
    """

    def grasp_cylinder(
        self,
        cylinder_radius: float,
        cylinder_height: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0.0, 2 * np.pi),
        subject: str = "gripper",
        reference: str = "cylinder",
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for all cylinder grasp modes.

        Combines side, top-down, and bottom-up approaches.
        Returns 2*k + k + k = 4*k templates (default k=3: 12 templates).

        Args:
            cylinder_radius: Cylinder radius [m].
            cylinder_height: Cylinder height [m]. Bottom at z=0, top at z=height.
            preshape:        Jaw opening [m]. Defaults to 2*r + clearance.
            k:               Number of discrete depths per mode (default 3).
            clearance:       Safety buffer [m]. Defaults to 10% of finger_length.
            angle_range:     Yaw freedom (default full 360°).
            subject:         Label for the end-effector entity.
            reference:       Label for the reference object.

        Returns:
            List of (2*k + k + k) TSRTemplates.
            Empty list if preshape cannot span the cylinder.
        """
        shared = dict(
            preshape=preshape,
            k=k,
            clearance=clearance,
            angle_range=angle_range,
            subject=subject,
            reference=reference,
        )
        return (
            self.grasp_cylinder_side(cylinder_radius, cylinder_height, **shared)
            + self.grasp_cylinder_top(cylinder_radius, cylinder_height, **shared)
            + self.grasp_cylinder_bottom(cylinder_radius, cylinder_height, **shared)
        )

    @abstractmethod
    def grasp_cylinder_side(
        self,
        cylinder_radius: float,
        cylinder_height: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0.0, 2 * np.pi),
        subject: str = "gripper",
        reference: str = "cylinder",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for side-grasping a cylinder — 2*k templates.

        Returns pre-grasp configurations: hand open at preshape, positioned
        so closing the fingers contacts the cylinder surface radially.

        Args:
            cylinder_radius: Cylinder radius [m].
            cylinder_height: Cylinder height [m]. Graspable band: [clearance, height-clearance].
            preshape:        Jaw opening [m]. Defaults to 2*r + clearance.
            k:               Number of discrete approach depths (default 3).
            clearance:       Safety buffer [m] at height ends and depth limits.
                             Defaults to 10% of finger_length.
            angle_range:     Yaw freedom (default full 360°).
            subject:         Label for the end-effector entity.
            reference:       Label for the reference object.
            name:            Template name prefix.
            description:     Template description.

        Returns:
            List of 2*k TSRTemplates (k depths × 2 roll orientations).
            Empty list if preshape cannot span the cylinder.
        """
        raise NotImplementedError

    def grasp_cylinder_top(
        self,
        cylinder_radius: float,
        cylinder_height: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0.0, 2 * np.pi),
        subject: str = "gripper",
        reference: str = "cylinder",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for grasping a cylinder from above — k templates.

        Returns pre-grasp configurations with z_EE = [0,0,-1] (approach
        downward). TSR origin at z = cylinder_height (top face).

        Insertion depth is measured from the approached (top) face and spans
        ``d ∈ [m, min(finger_length, cylinder_height) − m]``, where
        ``m = max(clearance, 2 · atol(scale))`` is the edge margin (#121; ``m`` is the
        requested clearance except at or below the contract tolerance). **Which limit
        binds depends on the cylinder** (#122):

        * ``finger_length ≤ cylinder_height`` — the palm-clearance limit binds: the
          deepest pose puts the palm exactly ``m`` above the approached rim.
        * ``cylinder_height < finger_length`` — the far-cap limit binds: the deepest
          pose stops the fingertips ``m`` short of the **bottom** face, and the palm
          can stay much farther than ``m`` above the approached rim.

        A hand whose fingertips reach further once closed (``ParallelJawGripper.reach``)
        moves the far-cap limit up by that much, so the closed fingertips stop ``m``
        short of the bottom face too (#193).

        Args:
            cylinder_radius: Cylinder radius [m].
            cylinder_height: Cylinder height [m] (z of top face); also bounds the
                             insertion depth, see above.
            preshape:        Jaw opening [m]. Must exceed cylinder diameter.
                             Defaults to 2*r + clearance.
            k:               Number of discrete approach depths (default 3).
            clearance:       Safety buffer [m]. Defaults to 10% of finger_length.
            angle_range:     Yaw freedom (default full 360°).

        Returns:
            List of k TSRTemplates. Empty list if preshape cannot span the cylinder.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement grasp_cylinder_top")

    def grasp_cylinder_bottom(
        self,
        cylinder_radius: float,
        cylinder_height: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0.0, 2 * np.pi),
        subject: str = "gripper",
        reference: str = "cylinder",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for grasping a cylinder from below — k templates.

        Returns pre-grasp configurations with z_EE = [0,0,+1] (approach
        upward). TSR origin at z = 0 (bottom face).

        Insertion depth is measured from the approached (bottom) face and spans
        ``d ∈ [m, min(finger_length, cylinder_height) − m]``, where
        ``m = max(clearance, 2 · atol(scale))`` is the edge margin (#121; ``m`` is the
        requested clearance except at or below the contract tolerance). **Which limit
        binds depends on the cylinder** (#122):

        * ``finger_length ≤ cylinder_height`` — the palm-clearance limit binds: the
          deepest pose puts the palm exactly ``m`` below the approached rim.
        * ``cylinder_height < finger_length`` — the far-cap limit binds: the deepest
          pose stops the fingertips ``m`` short of the **top** face, and the palm can
          stay much farther than ``m`` below the approached rim.

        A hand whose fingertips reach further once closed (``ParallelJawGripper.reach``)
        moves the far-cap limit down by that much, so the closed fingertips stop ``m``
        short of the top face too (#193).

        Args:
            cylinder_radius: Cylinder radius [m].
            cylinder_height: Cylinder height [m]; bounds the insertion depth (see
                             above). The TSR origin is the bottom face at z = 0.
            preshape:        Jaw opening [m]. Must exceed cylinder diameter.
                             Defaults to 2*r + clearance.
            k:               Number of discrete approach depths (default 3).
            clearance:       Safety buffer [m]. Defaults to 10% of finger_length.
            angle_range:     Yaw freedom (default full 360°).

        Returns:
            List of k TSRTemplates. Empty list if preshape cannot span the cylinder.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement grasp_cylinder_bottom")

    def grasp_box(
        self,
        box_x: float,
        box_y: float,
        box_z: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        subject: str = "gripper",
        reference: str = "box",
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for all box grasp modes — 6*k templates.

        Combines top, bottom, +x/-x face, and +y/-y face approaches.
        Each face offers two finger orientations (one per face axis), filtered
        by max_aperture.  Maximum 2 orientations × 6 faces × k depths = 12*k.
        Actual count depends on box dimensions vs gripper aperture.

        Box coordinate convention::

            x ∈ [-box_x/2, +box_x/2]   (centered)
            y ∈ [-box_y/2, +box_y/2]   (centered)
            z ∈ [0,         box_z   ]   (bottom at z=0)

        Args:
            box_x:     Box width  [m] (along x-axis).
            box_y:     Box depth  [m] (along y-axis).
            box_z:     Box height [m] (along z-axis; top face at z=box_z).
            preshape:  Jaw opening [m]. Defaults to max_aperture / 2.
            k:         Number of discrete approach depths per face (default 3).
            clearance: Safety buffer [m]. Defaults to 10% of min(finger_length,
                       the box's extent along the approach) (#195).
            subject:   Label for the end-effector entity.
            reference: Label for the reference object.

        Returns:
            List of 6*k TSRTemplates.
        """
        shared = dict(
            preshape=preshape,
            k=k,
            clearance=clearance,
            subject=subject,
            reference=reference,
        )
        return (
            self.grasp_box_face_x(box_x, box_y, box_z, **shared)
            + self.grasp_box_face_y(box_x, box_y, box_z, **shared)
            + self.grasp_box_top(box_x, box_y, box_z, **shared)
            + self.grasp_box_bottom(box_x, box_y, box_z, **shared)
        )

    def grasp_box_top(
        self,
        box_x: float,
        box_y: float,
        box_z: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        subject: str = "gripper",
        reference: str = "box",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for grasping a box from above — k templates.

        Gripper approaches from above (z_EE = [0,0,-1]). TSR origin at
        z = box_z (top face). Fingers slide freely in x and y within the face
        bounds; no rotational freedom.

        Args:
            box_x:     Box width  [m].
            box_y:     Box depth  [m].
            box_z:     Box height [m] (z of top face).
            preshape:  Jaw opening [m]. Defaults to max_aperture / 2.
            k:         Number of discrete approach depths (default 3).
            clearance: Safety buffer [m]. Defaults to 10% of min(finger_length,
                       the box's extent along the approach) (#195).

        Returns:
            List of k TSRTemplates.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement grasp_box_top")

    def grasp_box_bottom(
        self,
        box_x: float,
        box_y: float,
        box_z: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        subject: str = "gripper",
        reference: str = "box",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for grasping a box from below — k templates.

        Gripper approaches from below (z_EE = [0,0,+1]). TSR origin at
        z = 0 (bottom face). Fingers slide freely in x and y within the face
        bounds; no rotational freedom.

        Args:
            box_x:     Box width  [m].
            box_y:     Box depth  [m].
            box_z:     Box height [m]; bounds the insertion depth,
                       ``d ∈ [m, min(finger_length, box_z) − m]`` with the edge margin
                       ``m = max(clearance, 2 · atol(scale))`` (#121), less how much
                       further the fingertips reach once closed on the span (#193).
                       The TSR origin is the bottom face at z = 0.
            preshape:  Jaw opening [m]. Defaults to max_aperture / 2.
            k:         Number of discrete approach depths (default 3).
            clearance: Safety buffer [m]. Defaults to 10% of min(finger_length,
                       the box's extent along the approach) (#195).

        Returns:
            List of k TSRTemplates.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement grasp_box_bottom")

    def grasp_box_face_x(
        self,
        box_x: float,
        box_y: float,
        box_z: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        subject: str = "gripper",
        reference: str = "box",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for grasping the ±x faces of a box — 2*k templates.

        k templates approach from +x (z_EE = [-1,0,0]) and k from -x
        (z_EE = [+1,0,0]). Fingers slide freely in y and z within the face
        bounds; no rotational freedom.

        Args:
            box_x:     Box width  [m] (standoff is box_x/2 + finger_length - depth).
            box_y:     Box depth  [m].
            box_z:     Box height [m].
            preshape:  Jaw opening [m]. Defaults to max_aperture / 2.
            k:         Number of discrete approach depths per face (default 3).
            clearance: Safety buffer [m]. Defaults to 10% of min(finger_length,
                       the box's extent along the approach) (#195).

        Returns:
            List of 2*k TSRTemplates (k for +x face, k for -x face).
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement grasp_box_face_x")

    def grasp_box_face_y(
        self,
        box_x: float,
        box_y: float,
        box_z: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        subject: str = "gripper",
        reference: str = "box",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for grasping the ±y faces of a box — 2*k templates.

        k templates approach from +y (z_EE = [0,-1,0]) and k from -y
        (z_EE = [0,+1,0]). Fingers slide freely in x and z within the face
        bounds; no rotational freedom.

        Args:
            box_x:     Box width  [m].
            box_y:     Box depth  [m] (standoff is box_y/2 + finger_length - depth).
            box_z:     Box height [m].
            preshape:  Jaw opening [m]. Defaults to max_aperture / 2.
            k:         Number of discrete approach depths per face (default 3).
            clearance: Safety buffer [m]. Defaults to 10% of min(finger_length,
                       the box's extent along the approach) (#195).

        Returns:
            List of 2*k TSRTemplates (k for +y face, k for -y face).
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement grasp_box_face_y")

    def grasp_sphere(
        self,
        object_radius: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0.0, 2 * np.pi),
        subject: str = "gripper",
        reference: str = "sphere",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Full-surface grasp templates for a sphere (mode ``"surface"``).

        The template's rotational bounds cover **SO(3)**: a sampled pose can
        approach the sphere from any direction (not only the equatorial plane).
        TSR origin at the sphere center; k discrete approach depths.

        This is a statement about the *set* the template covers. It is **not** a
        claim about the sampling *distribution*: ``TSR.sample`` draws independent
        uniform roll/pitch/yaw, which is not Haar-uniform on SO(3); use
        :func:`tsr.sampling.sample_haar` for Haar-uniform rotations over the
        represented set (#72). ``angle_range`` constrains the Euler yaw coordinate:
        the represented approach directions form the lune ``azimuth ∈ angle_range``,
        not a spherical cap.

        Sphere coordinate convention: center at origin, radius = object_radius.

        Args:
            object_radius: Sphere radius [m].
            preshape:      Jaw opening [m]. Defaults to 2*r + clearance.
            k:             Number of discrete approach depths (default 3).
            clearance:     Safety buffer [m]. Defaults to 10% of finger_length.
            angle_range:   Yaw freedom (default full 360°).
            subject:       Label for the end-effector entity.
            reference:     Label for the reference object.
            name:          Template name prefix.
            description:   Template description.

        Returns:
            List of k TSRTemplates. Empty list if preshape cannot span the sphere.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement grasp_sphere")

    def grasp_torus(
        self,
        torus_radius: float,
        tube_radius: float,
        preshape: Optional[float] = None,
        k: int = 3,
        n_minor: int = 5,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0.0, 2 * np.pi),
        subject: str = "gripper",
        reference: str = "torus",
        *,
        minor_angle_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for all torus grasp modes.

        Combines radial side grasps (always) and span grasps (when the outer
        diameter fits within max_aperture).

        ``minor_angle_range`` (keyword-only) selects the tube cross-section approach
        interval for the **side** grasps only (span grasps have no minor angle); it
        must lie within the externally accessible outer half ``[−π/2, +π/2]``.
        ``n_minor == 1`` samples the interval centre. See :meth:`grasp_torus_side`.

        Torus coordinate convention: center at origin, axis along +z.
            torus_radius R: distance from center to tube center.
            tube_radius  r: cross-section radius of the tube.

        Side mode (2 * k * n_minor templates):
            Approach from n_minor discrete angles α ∈ [−π/2, +π/2] in the
            tube cross-section plane. α=0 is a pure equatorial radial approach;
            α=±π/2 approaches from below/above (matching span geometry).
            k depths × 2 hand flips per angle. Full yaw covers all azimuth
            positions around the ring.

        Span mode (up to 2*k templates, silently omitted if too large):
            Approach from above and below; fingers span the full outer diameter
            2*(R+r). Only included when 2*(R+r) + clearance ≤ max_aperture.
            k depths each, full yaw covers all finger orientations.

        Args:
            torus_radius:  Major radius R [m] — center to tube center.
            tube_radius:   Minor radius r [m] — tube cross-section.
            preshape:      Jaw opening for side grasps [m]. Defaults to
                           2*r + clearance.
            k:             Number of discrete depths per mode (default 3).
            n_minor:       Discrete approach angles around the tube
                           cross-section (default 5).
            clearance:     Safety buffer [m]. Defaults to 10% of finger_length.
            angle_range:   Yaw freedom for side grasps (default full 360°).
            subject:       Label for the end-effector entity.
            reference:     Label for the reference object.

        Returns:
            List of 2*k*n_minor + up to 2*k TSRTemplates.
        """
        shared = dict(
            preshape=preshape,
            k=k,
            clearance=clearance,
            subject=subject,
            reference=reference,
        )
        return self.grasp_torus_side(
            torus_radius,
            tube_radius,
            n_minor=n_minor,
            angle_range=angle_range,
            minor_angle_range=minor_angle_range,
            **shared,
        ) + self.grasp_torus_span(torus_radius, tube_radius, **shared)

    def grasp_torus_side(
        self,
        torus_radius: float,
        tube_radius: float,
        preshape: Optional[float] = None,
        k: int = 3,
        n_minor: int = 5,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0.0, 2 * np.pi),
        subject: str = "gripper",
        reference: str = "torus",
        name: str = "",
        description: str = "",
        *,
        minor_angle_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
    ) -> List[TSRTemplate]:
        """Side grasp templates for a torus tube — 2 * k * n_minor templates.

        The gripper approaches from ``n_minor`` discrete minor angles ``α`` in the
        tube cross-section plane, sampled from ``minor_angle_range`` (keyword-only,
        default ``[−π/2, +π/2]``), with full azimuthal yaw freedom around the ring.
        Two hand flip variants per (α, depth).

        **Coverage.** ``[−π/2, +π/2]`` is the externally accessible **outer half** of
        the tube cross-section; ``minor_angle_range`` must lie within it (inner-hole
        approaches are out of scope). ``n_minor == 1`` samples the interval **centre**
        (the equator ``α = 0`` for the default range), not an endpoint.

        Args:
            torus_radius:      Major radius R [m].
            tube_radius:       Minor radius r [m].
            preshape:          Jaw opening [m]. Defaults to 2*r + clearance.
            k:                 Number of discrete approach depths (default 3).
            n_minor:           Discrete approach angles in the tube cross-section
                               (default 5).
            clearance:         Safety buffer [m]. Defaults to 10% of finger_length.
            angle_range:       Yaw freedom (default full 360°).
            minor_angle_range: Closed minor-angle interval within ``[−π/2, +π/2]``
                               (keyword-only, default the full outer half).

        Returns:
            List of 2*k*n_minor TSRTemplates. Empty (``finger_too_short`` /
            ``insufficient_clearance_band`` / ``closed_reach_too_long``) if the fingers
            cannot reach the tube with clearance, open or closed, or the straddle reason
            if the tube diameter cannot be spanned.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement grasp_torus_side")

    def grasp_torus_span(
        self,
        torus_radius: float,
        tube_radius: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        subject: str = "gripper",
        reference: str = "torus",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Span grasp templates for a torus — up to 2*k templates.

        Approach from above (+k) and below (-k) with fingers spanning the full
        outer diameter 2*(R+r). Full yaw freedom covers all finger orientations.
        Silently returns [] if the outer diameter + clearance exceeds max_aperture.

        Args:
            torus_radius:  Major radius R [m].
            tube_radius:   Minor radius r [m].
            preshape:      Jaw opening [m]. Defaults to 2*(R+r) + clearance.
            k:             Number of discrete depths per direction (default 3).
            clearance:     Safety buffer [m]. Defaults to 10% of finger_length.

        Returns:
            List of up to 2*k TSRTemplates (k top + k bottom).
            Empty list if max_aperture cannot span the outer diameter.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement grasp_torus_span")

    # ------------------------------------------------------------------
    # Shared factory contract
    #
    # Across all grasp_* factories:
    #   * An exception reports an INVALID REQUEST — arguments that are
    #     nonsensical on their own (non-positive size, k < 1, reversed
    #     angle_range). Callers should never trigger these in correct code.
    #   * An empty list reports an EMPTY FEASIBLE SET — valid arguments that
    #     simply admit no grasp (object too wide for the jaws, band too thin,
    #     etc.). Callers can sweep many objects without try/except.
    #
    # Infeasibility carries a machine-readable reason; the *public* factory
    # method emits a single debug log at its boundary via ``_empty`` rather
    # than logging at each internal early return.
    # ------------------------------------------------------------------

    @staticmethod
    def _check_positive_finite(name: str, value: float) -> None:
        """Validate a finite, strictly positive scalar. Raises ValueError otherwise (#68)."""
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a finite positive number, got {value!r}")
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number, got {value!r}")

    @classmethod
    def _check_dimensions(cls, **named: float) -> None:
        """Validate every named primitive dimension is finite and positive (#68)."""
        for name, value in named.items():
            cls._check_positive_finite(name, value)

    @staticmethod
    def _check_nonnegative_finite(name: str, value: float) -> None:
        """Validate a finite scalar >= 0 (clearance, clearance_fraction). Raises otherwise (#104)."""
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
        if not (np.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")

    @classmethod
    def _check_preshape(cls, preshape: Optional[float]) -> None:
        """A supplied preshape must be finite and positive; None uses the default (#68)."""
        if preshape is not None:
            cls._check_positive_finite("preshape", preshape)

    @staticmethod
    def _check_count(name: str, value: int) -> None:
        """Validate a discrete count (``k``, ``n_minor``): an integer >= 1 (#68)."""
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        if value < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {value!r}")

    @staticmethod
    def _check_depth_count(k: int) -> None:
        """Backwards-compatible alias for :meth:`_check_count` on ``k``."""
        GripperBase._check_count("k", k)

    @staticmethod
    def _check_angle_range(angle_range: Tuple[float, float]) -> None:
        """Validate a (min, max) yaw range: finite and ordered (min <= max) (#68)."""
        lo, hi = angle_range
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"angle_range must be finite, got {angle_range}")
        if lo > hi:
            raise ValueError(f"angle_range must be (min, max) with min <= max, got {angle_range}")

    @staticmethod
    def _sample_range(interval: Tuple[float, float], n: int) -> np.ndarray:
        """``n`` samples of a closed interval; ``n == 1`` is the CENTER, not an endpoint.

        ``np.linspace(lo, hi, 1)`` returns ``lo``, which for a symmetric minor-angle
        range would silently pick a lower approach rather than the intended equatorial
        centre. A single sample is the interval midpoint (#71).
        """
        lo, hi = interval
        if n == 1:
            return np.array([(lo + hi) / 2.0])
        return np.linspace(lo, hi, n)

    @staticmethod
    def _usable_depths(lo: float, hi: float, k: int) -> Optional[np.ndarray]:
        """``k`` approach depths in ``[lo, hi]``, or ``None`` if the band is empty (#68).

        Exact, scale-independent boundary policy (#105):

        * ``hi < lo`` -> ``None`` (empty band; a positive clearance left no interval, so
          the caller emits ``insufficient_clearance_band`` rather than reversing the
          shallow-to-deep ordering via ``np.linspace``);
        * ``hi == lo`` -> a single depth (coincident endpoints);
        * ``hi > lo`` -> up to ``k`` depths.

        The comparison is exact: a positive-width interval never collapses through a
        default (unit- or scale-dependent) ``np.isclose`` tolerance. The returned
        depths are strictly increasing (shallow to deep) and **distinct**: an interval
        only a few ulps wide can make ``np.linspace`` round samples onto the same
        float, and those duplicates are dropped, so ``len(depths)`` is the number of
        distinct depth levels that ``GraspProvenance.depth_count`` reports (#81).
        """
        if hi < lo:
            return None
        if hi == lo:
            return np.array([float(lo)])
        depths = np.linspace(lo, hi, k)
        # linspace is already nondecreasing: drop exact adjacent repeats, no sort (#119).
        keep = np.empty(depths.shape, dtype=bool)
        keep[0] = True
        keep[1:] = depths[1:] != depths[:-1]
        return depths[keep]

    @staticmethod
    def _length_atol(scale: float) -> float:
        """Scale-aware length tolerance from the geometric contract (docs/ARCHITECTURE.md).

        ``atol = 1e-9 + 1e-6 · scale``, defined once in :func:`tsr.core.utils.length_atol`
        and shared with the placement factories. The analytic oracle implements the same
        documented formula independently, so factory feasibility agrees with the
        contract (#107). ``scale`` is the primitive's characteristic dimension.
        """
        return length_atol(scale)

    def _edge_margin(self, clearance: float, scale: float) -> float:
        """Edge-facing band margin ``m = max(clearance, 2 * _length_atol(scale))`` (#121).

        A band limit that faces an **edge** of the primitive -- the approached face,
        the opposite face or cap, a lateral slide edge, a cylinder rim -- must keep the
        contacts strictly off that edge, where the surface normal is ambiguous and the
        insertion would be degenerate. ``clearance = 0`` is a valid request (#104), and
        a clearance below the contract tolerance is indistinguishable from zero, so
        edge limits use at least ``_length_atol(scale)``. This mirrors the scale-aware
        straddle margin (#107) and matches the #67 oracle, which requires a positive
        insertion and off-edge contacts.

        The floor is **twice** ``_length_atol(scale)``: the oracle compares the pose's
        *realized* insertion against ``_length_atol``, and recovering that depth from
        the pose cancels two coordinates, so a floor of exactly the tolerance loses to
        rounding. Twice the tolerance clears it and stays sub-micron at hand scale
        (0.23 µm for a 0.11 m box).

        Palm clearance and the default preshape still use the requested ``clearance``;
        only edge-facing band limits use this margin.
        """
        return max(clearance, 2.0 * self._length_atol(scale))

    @staticmethod
    def _default_preshape(object_span: float, clearance: float) -> float:
        """``object_span + clearance``, advanced until the **realized** margin meets it.

        The straddle rule is decided on the realized margin ``preshape - object_span``
        (#129), but the sum that builds a default preshape rounds to nearest, so it can
        land a hair *below* the requested clearance -- which would reject the very
        clearance the caller asked for, including the documented floor itself (#131).
        A public clearance request is met conservatively by the nearest representable
        geometry instead: step toward +inf until the margin is at least ``clearance``
        (normally one step, and none at ordinary magnitudes).

        Only *defaults* are constructed this way. An explicitly supplied preshape is
        never altered; it is judged by its own realized margin.
        """
        p = object_span + clearance
        while p - object_span < clearance:
            p = math.nextafter(p, math.inf)
        return p

    def _infeasibility_reason(
        self, preshape: float, object_span: float, scale: Optional[float] = None
    ) -> Optional[str]:
        """Why a grasp of ``object_span`` at ``preshape`` is infeasible, else None.

        ``"exceeds_aperture"`` — the jaws cannot open wider than the object;
        ``"cannot_straddle"`` — the object does not fit **strictly between** the open
        jaws by the contract's scale-aware tolerance, matching the analytic oracle's
        clause 2 (#107), rather than merely being narrower than the preshape.
        ``scale`` defaults to ``object_span / 2``.

        The test is on the **margin**, not on the two sums (#129). The jaw margin
        ``preshape - object_span`` is exact by Sterbenz whenever the two are within a
        factor of two of each other, whereas comparing ``preshape`` against
        ``object_span + 2·atol`` rounds both operands: when the margin is many orders
        of magnitude smaller than the span, that addition loses it and a margin below
        the floor is accepted.

        Each pad must clear the object by ``2 · _length_atol(scale)`` -- **twice** the
        oracle's tolerance, so a total margin of ``4 · atol``. The oracle admits a pose
        when the contacts sit inside the open jaws by ``atol``; at a total margin of
        exactly ``2 · atol`` that bound evaluates to the contact coordinate itself, so
        whether a pose certifies is decided by the rounding of ``(r + atol) - atol``
        and of the contact coordinate. Measured on a sphere at three scales, a margin
        of exactly ``2 · atol`` left ~59% of sampled poses uncertifiable, while any
        margin strictly above it left none. Doubling mirrors the edge-margin floor
        (#121), which doubles for the same reason: arithmetic eats one tolerance.
        """
        if preshape > self.max_aperture:
            return "exceeds_aperture"
        if scale is None:
            scale = object_span / 2.0
        if preshape - object_span < 4.0 * self._length_atol(scale):
            return "cannot_straddle"
        return None

    def _empty(self, method: str, reason: str, **context) -> List[TSRTemplate]:
        """Return [] and log, once at debug level, why the feasible set is empty."""
        if context:
            ctx = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in context.items())
            logger.debug("%s.%s: empty feasible set (%s; %s)", type(self).__name__, method, reason, ctx)
        else:
            logger.debug("%s.%s: empty feasible set (%s)", type(self).__name__, method, reason)
        return []
