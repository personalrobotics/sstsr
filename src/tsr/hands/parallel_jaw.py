# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Parallel jaw gripper hand models."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from tsr.grasp_provenance import GraspProvenance
from tsr.template import TSRTemplate

from .base import GripperBase

_DEPTH_LABELS = {1: ["mid"], 2: ["shallow", "deep"], 3: ["shallow", "mid", "deep"]}


def _depth_label(k: int, i: int) -> str:
    return (_DEPTH_LABELS.get(k) or [f"depth {j + 1}/{k}" for j in range(k)])[i]


class ParallelJawGripper(GripperBase):
    """Parallel jaw gripper: generates TSRTemplates from object geometry.

    Poses sampled from these TSRs are pre-grasp configurations — the hand is
    open at ``preshape``, positioned so that closing the fingers achieves
    stable contact with the object. The TSR constrains where the hand must be
    before closing; it does not explicitly verify force closure.

    Frame convention:
        z = approach direction (toward object surface)
        y = finger opening direction
        x = palm normal (right-hand rule: x = y × z)

    AnyGrasp / GraspNet uses x=approach — convert with::

        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    **Closed reach** (#193). ``finger_length`` is the fingertips' reach with the hand open.
    A linkage hand's pads move on an arc, so closed on an object its fingertips can reach
    further: :meth:`reach` gives the reach closed to a width. Every template whose depth band
    faces the far side of what the fingers span keeps the fingertips a margin inside that
    far side both open and closed -- the far-face limit is ``extent − m − (reach(span) −
    finger_length)``. The default ``reach`` is ``finger_length`` at every width (an ideal
    prismatic jaw); a hand declares its measured curve as :attr:`CLOSED_REACH`.

    Args:
        finger_length: Distance from palm to fingertip, hand open [m].
        max_aperture:  Maximum jaw opening [m].
    """

    #: Measured closed reach, ``((width, reach), ...)`` in metres with widths ascending, or
    #: ``None`` for an ideal prismatic jaw. ``width`` is the gap between the pads' contact
    #: faces -- the quantity ``max_aperture`` and a template's span measure -- and ``reach``
    #: how far the fingers extend past the palm along the approach, closed to that gap.
    CLOSED_REACH: Optional[Tuple[Tuple[float, float], ...]] = None

    def __init__(
        self,
        finger_length: float,
        max_aperture: float,
        clearance_fraction: float = 0.1,
    ):
        # Centralized gripper-parameter validation (#68): a malformed gripper is an
        # invalid request, so we raise rather than emit templates or NaNs later.
        self._check_positive_finite("finger_length", finger_length)
        self._check_positive_finite("max_aperture", max_aperture)
        self._check_nonnegative_finite("clearance_fraction", clearance_fraction)
        self.finger_length = finger_length
        self.max_aperture = max_aperture
        self.clearance_fraction = clearance_fraction

    def reach(self, width: float) -> float:
        """How far the fingertips reach past the palm along the approach, closed to ``width`` [m].

        ``width`` is the gap between the pads' contact faces, i.e. the width of the object
        the fingers close on. With no :attr:`CLOSED_REACH` this is ``finger_length`` at
        every width. With one, it is the curve interpolated linearly, held at its end values
        outside the measured widths, and never less than ``finger_length``: the open hand's
        reach already bounds every template, so a curve can only tighten it.

        Raises ``ValueError`` for a NaN, infinite, negative or nonnumeric ``width``.
        """
        self._check_nonnegative_finite("width", width)
        if self.CLOSED_REACH is None:
            return self.finger_length
        widths, reaches = zip(*self.CLOSED_REACH)
        return max(self.finger_length, float(np.interp(width, widths, reaches)))

    def _closed_reach_gain(self, span: float) -> float:
        """How much further than ``finger_length`` the fingertips reach closed on ``span`` (#193).

        A far-face depth limit subtracts this, so the fingertips stay inside the far face
        once the hand closes as well as while it is open. Never negative: a subclass whose
        ``reach`` dips below ``finger_length`` still has its open fingertips to keep inside.
        """
        return max(0.0, self.reach(span) - self.finger_length)

    def _default_clearance(self, graspable_depth: float) -> float:
        """Compute default clearance from graspable depth.

        Clearance is ``clearance_fraction`` of the graspable depth — the
        distance fingers can wrap around the object.

        Args:
            graspable_depth: max penetration depth for this grasp geometry,
                typically min(finger_length, object_radius) for side grasps
                or finger_length for top/bottom grasps.
        """
        return self.clearance_fraction * graspable_depth

    def _radial_band_reason(self, radius: float, clearance: float) -> str:
        """Classify an empty radial depth band ``[radius, min(L, 2r − g) − clearance]`` (#108).

        ``g`` is the closed-reach gain at the diameter, ``reach(2r) − L`` (#193). The band
        empties for three distinct reasons, named in this order of precedence: the fingers
        are too short to keep the palm outside with the requested clearance
        (``finger_too_short``: ``finger_length < radius + clearance``); reach is ample but
        the clearance relative to the radius empties the far-surface limit even for the open
        hand (``insufficient_clearance_band``: ``min(L, 2r) − clearance < radius``, which
        requires ``clearance > radius``); or the open hand fits but the closed fingertips
        would pass the far surface (``closed_reach_too_long``). This is diagnostics only;
        the returned feasible set is unchanged.
        """
        if self.finger_length < radius + clearance:
            return "finger_too_short"
        if min(self.finger_length, 2.0 * radius) - clearance < radius:
            return "insufficient_clearance_band"
        return "closed_reach_too_long"

    def _cap_band_reason(self, extent: float, margin: float) -> str:
        """Classify an empty approach depth band ``[m, min(L, extent − g) − m]`` (#193).

        ``insufficient_clearance_band`` when the band is empty even for the open hand
        (``min(L, extent) − m < m``); otherwise the open hand fits and the fingertips, closed
        on the span and ``g = reach(span) − L`` further on, would pass the far face:
        ``closed_reach_too_long``. Diagnostics only.
        """
        if min(self.finger_length, extent) - margin < margin:
            return "insufficient_clearance_band"
        return "closed_reach_too_long"

    def _resolve_clearance(self, clearance: Optional[float], graspable_depth: float) -> float:
        """Validate an explicit ``clearance`` (finite, >= 0) or derive the default (#104).

        A NaN/inf/negative/Boolean/nonnumeric ``clearance`` is an invalid request and
        raises ``ValueError`` identifying ``clearance`` -- before it is used to derive
        a preshape, bounds, depths, or provenance. ``clearance=0`` is valid.
        """
        if clearance is None:
            return self._default_clearance(graspable_depth)
        self._check_nonnegative_finite("clearance", clearance)
        return clearance

    def _validate(self, cylinder_radius: float, preshape: float, cylinder_height: Optional[float] = None) -> None:
        # Invalid request -> raise. (An over-wide preshape is geometric
        # infeasibility, reported by the caller via _empty; see the base class.)
        # A finite, positive radius/height/preshape is required consistently across
        # every cylinder entry point (#68).
        self._check_positive_finite("cylinder_radius", cylinder_radius)
        if cylinder_height is not None:
            self._check_positive_finite("cylinder_height", cylinder_height)
        self._check_preshape(preshape)

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
        """Side grasp templates for a cylinder — 2*k templates.

        Returns k depth levels × 2 roll orientations. Each template has a fixed
        radial offset baked into Tw_e, covering the full pre-grasp volume:

          depth 1/k : fingertips a clearance inside the surface (shallowest)
          depth k/k : palm a clearance from the surface (deepest)

        Two roll orientations per depth (180° apart around z_EE):
          roll=0 : palm normal = +y_world, fingers open in +tangential
          roll=π : palm normal = -y_world, fingers open in -tangential
        A symmetric hand produces identical poses; asymmetric hands need both.

        The radial approach direction couples with yaw and cannot be encoded in
        Bw directly. Instead, k discrete depths are baked into Tw_e so the full
        pre-grasp volume is covered without post-processing.

        The deepest pose also keeps the fingertips a clearance inside the far side of
        the cylinder once closed across its diameter, ``reach(2r)`` past the palm (#193).
        For a linkage hand that can bind before the palm limit, and on a thin cylinder it
        empties the family with ``closed_reach_too_long``.

        Args:
            clearance:  Safety buffer [m] applied to: height ends, fingertip
                        start depth, and palm stop depth. Default: 30% of
                        graspable depth (ensures stable contact).

        Returns [] if preshape <= cylinder diameter. Raises ValueError for
        invalid geometry.
        """
        self._check_depth_count(k)
        self._check_angle_range(angle_range)
        clearance = self._resolve_clearance(clearance, min(self.finger_length, cylinder_radius))
        if preshape is None:
            preshape = self._default_preshape(2.0 * cylinder_radius, clearance)
        self._validate(cylinder_radius, preshape, cylinder_height)
        reason = self._infeasibility_reason(
            preshape, 2.0 * cylinder_radius, scale=max(cylinder_radius, cylinder_height)
        )
        if reason:
            return self._empty(
                "grasp_cylinder_side",
                reason,
                preshape=preshape,
                diameter=2.0 * cylinder_radius,
                max_aperture=self.max_aperture,
            )
        # Height band limits face the cylinder rims, so they use the edge margin (#121).
        margin = self._edge_margin(clearance, max(cylinder_radius, cylinder_height))
        h0, h1 = margin, cylinder_height - margin
        # Exact-interval convention (#105): height == 2*margin is ONE valid centred
        # height (zero-width band), not an empty set; only h1 < h0 is empty.
        if h1 < h0:
            return self._empty(
                "grasp_cylinder_side", "insufficient_clearance_band", height=cylinder_height, clearance=clearance
            )

        if not name:
            name = f"{reference.title()} Cylinder Side Grasp"

        z_mid, z_half = (h0 + h1) / 2.0, (h1 - h0) / 2.0
        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = z_mid

        Bw = np.array(
            [
                [0.0, 0.0],  # x: no radial freedom
                [0.0, 0.0],  # y: no tangential freedom
                [-z_half, z_half],  # z: height range (symmetric)
                [0.0, 0.0],  # roll: fixed (encoded in Tw_e)
                [0.0, 0.0],  # pitch
                [angle_range[0], angle_range[1]],  # yaw: angular freedom
            ]
        )

        # Shallowest: fingertips at cylinder center (depth = radius).
        # Deepest: fingertips past center, limited by finger_length or far surface.
        # Radial depth band: fingertips reach the axis (depth >= radius) and the palm
        # stays a clearance outside the surface (depth <= L - clearance), as do the
        # fingertips inside the far surface, closed across the diameter as well as open
        # (depth <= 2r - (reach(2r) - L) - clearance, #193). It is empty because the
        # fingers are too short (finger_length < radius + clearance, which would place the
        # palm inside the surface, #69), because an excessive clearance relative to the
        # radius empties the far-surface limit (#108), or because the closed fingertips
        # would pass it; _radial_band_reason names the failed constraint.
        depth_min = cylinder_radius
        depth_max = (
            min(self.finger_length, 2 * cylinder_radius - self._closed_reach_gain(2 * cylinder_radius)) - clearance
        )
        depths = self._usable_depths(depth_min, depth_max, k)
        if depths is None:
            return self._empty(
                "grasp_cylinder_side",
                self._radial_band_reason(cylinder_radius, clearance),
                finger_length=self.finger_length,
                radius=cylinder_radius,
                clearance=clearance,
            )

        common = dict(
            T_ref_tsr=T_ref_tsr,
            Bw=Bw,
            task="grasp",
            subject=subject,
            reference=reference,
            preshape=np.array([preshape]),
        )
        templates = []
        for i, d in enumerate(depths):
            ro = cylinder_radius + self.finger_length - d
            dlabel = _depth_label(len(depths), i)
            Tw_e_0 = np.array(
                [
                    [0.0, 0.0, -1.0, ro],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            Tw_e_pi = np.array(
                [
                    [0.0, 0.0, -1.0, ro],
                    [0.0, -1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            for Tw_e, roll_label, roll_variant in (
                (Tw_e_0, "roll 0°", "roll0"),
                (Tw_e_pi, "roll 180°", "rollpi"),
            ):
                t_desc = description or (
                    f"{dlabel.capitalize()} side grasp on {reference}: "
                    f"standoff {ro * 1000:.0f}mm from axis, {roll_label}, "
                    f"preshape {preshape * 1000:.0f}mm"
                )
                templates.append(
                    TSRTemplate(
                        Tw_e=Tw_e,
                        name=f"{name} — {dlabel}, {roll_label}",
                        description=t_desc,
                        provenance=GraspProvenance(
                            primitive="cylinder",
                            mode="side",
                            approach="radial",
                            finger_orientation="tangential",
                            depth_index=i,
                            depth_count=len(depths),
                            depth=float(d),
                            symmetry=roll_variant,
                        ),
                        **common,
                    )
                )
        return templates

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
        """Top-down grasp templates for a cylinder — k templates.

        Gripper approaches from above (z_EE = [0,0,-1]). TSR origin at
        z = cylinder_height. Full yaw covers all finger orientations.

        Insertion depth is measured from the top face and is bounded by the
        **height** as well as the fingers (#122) — the same rule the box approach
        bands use:

            d ∈ [m, min(finger_length, cylinder_height) − m]

        where ``m = max(clearance, 2 · atol(scale))`` is the edge margin (#121), equal
        to the requested clearance except at or below the contract tolerance. The
        shallowest pose puts the fingertips ``m`` inside the top face.
        Which limit binds at the deep end depends on the cylinder:

        * ``finger_length ≤ cylinder_height`` — the palm-clearance limit binds: the
          palm ends up exactly ``m`` above the approached rim.
        * ``cylinder_height < finger_length`` — the far-cap limit binds: the
          fingertips stop ``m`` short of the bottom face, and the palm stays
          ``finger_length − (cylinder_height − m)`` above the rim, which can be much
          more than ``m``.

        The far-cap limit holds for the closed fingertips too (#193): it is
        ``cylinder_height − g − m``, where ``g = reach(2r) − finger_length`` is how much
        further they reach closed across the diameter (0 for an ideal prismatic jaw).

        Returns ``[]`` with ``insufficient_clearance_band`` when the band is empty for the
        open hand (``min(finger_length, cylinder_height) < 2 · m``), and with
        ``closed_reach_too_long`` when only the closed fingertips leave no band.
        """
        self._check_depth_count(k)
        self._check_angle_range(angle_range)
        clearance = self._resolve_clearance(clearance, self.finger_length)
        if preshape is None:
            preshape = self._default_preshape(2.0 * cylinder_radius, clearance)
        self._validate(cylinder_radius, preshape, cylinder_height)
        reason = self._infeasibility_reason(
            preshape, 2.0 * cylinder_radius, scale=max(cylinder_radius, cylinder_height)
        )
        if reason:
            return self._empty(
                "grasp_cylinder_top",
                reason,
                preshape=preshape,
                diameter=2.0 * cylinder_radius,
                max_aperture=self.max_aperture,
            )

        if not name:
            name = f"{reference.title()} Cylinder Top Grasp"

        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = cylinder_height

        Bw = np.array(
            [
                [0.0, 0.0],  # x
                [0.0, 0.0],  # y
                [0.0, 0.0],  # z: fixed at top face
                [0.0, 0.0],  # roll
                [0.0, 0.0],  # pitch
                [angle_range[0], angle_range[1]],  # yaw: full rotation
            ]
        )

        # Bounded by the height as well as the fingers (#122): a deeper fingertip
        # would pass the opposite cap, which the #67 oracle rejects (clause 4) -- and
        # so would the closed fingertips, which reach further across the diameter (#193).
        # Both limits face a cap, so they use the edge margin (#121).
        margin = self._edge_margin(clearance, max(cylinder_radius, cylinder_height))
        far_cap = cylinder_height - self._closed_reach_gain(2.0 * cylinder_radius)
        depths = self._usable_depths(margin, min(self.finger_length, far_cap) - margin, k)
        if depths is None:
            return self._empty(
                "grasp_cylinder_top",
                self._cap_band_reason(cylinder_height, margin),
                clearance=clearance,
                finger_length=self.finger_length,
                cylinder_height=cylinder_height,
            )
        common = dict(
            T_ref_tsr=T_ref_tsr,
            Bw=Bw,
            task="grasp",
            subject=subject,
            reference=reference,
            preshape=np.array([preshape]),
        )
        templates = []
        for i, d in enumerate(depths):
            h_palm = self.finger_length - d
            dlabel = _depth_label(len(depths), i)
            t_desc = description or (
                f"{dlabel.capitalize()} top grasp on {reference}: "
                f"palm {h_palm * 1000:.0f}mm above rim, preshape {preshape * 1000:.0f}mm"
            )
            # z_EE = [0,0,-1] (approach down); x = y × z = [0,1,0]×[0,0,-1] = [-1,0,0]
            Tw_e = np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, h_palm],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            templates.append(
                TSRTemplate(
                    Tw_e=Tw_e,
                    name=f"{name} — {dlabel}",
                    description=t_desc,
                    provenance=GraspProvenance(
                        primitive="cylinder",
                        mode="top",
                        approach="+z",
                        finger_orientation="diameter",
                        depth_index=i,
                        depth_count=len(depths),
                        depth=float(d),
                    ),
                    **common,
                )
            )
        return templates

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
        """Bottom-up grasp templates for a cylinder — k templates.

        Gripper approaches from below (z_EE = [0,0,+1]). TSR origin at z = 0
        (bottom face). Full yaw covers all finger orientations.

        Insertion depth is measured from the bottom face and is bounded by the
        **height** as well as the fingers (#122):

            d ∈ [m, min(finger_length, cylinder_height) − m]

        where ``m = max(clearance, 2 · atol(scale))`` is the edge margin (#121), equal
        to the requested clearance except at or below the contract tolerance. The
        shallowest pose puts the fingertips ``m`` inside the bottom face.
        Which limit binds at the deep end depends on the cylinder:

        * ``finger_length ≤ cylinder_height`` — the palm-clearance limit binds: the
          palm ends up exactly ``m`` below the approached rim.
        * ``cylinder_height < finger_length`` — the far-cap limit binds: the
          fingertips stop ``m`` short of the top face, and the palm stays
          ``finger_length − (cylinder_height − m)`` below the rim, which can be much
          more than ``m``.

        The far-cap limit holds for the closed fingertips too (#193): it is
        ``cylinder_height − g − m``, where ``g = reach(2r) − finger_length`` is how much
        further they reach closed across the diameter (0 for an ideal prismatic jaw).

        Returns ``[]`` with ``insufficient_clearance_band`` when the band is empty for the
        open hand (``min(finger_length, cylinder_height) < 2 · m``), and with
        ``closed_reach_too_long`` when only the closed fingertips leave no band.
        """
        self._check_depth_count(k)
        self._check_angle_range(angle_range)
        clearance = self._resolve_clearance(clearance, self.finger_length)
        if preshape is None:
            preshape = self._default_preshape(2.0 * cylinder_radius, clearance)
        self._validate(cylinder_radius, preshape, cylinder_height)
        reason = self._infeasibility_reason(
            preshape, 2.0 * cylinder_radius, scale=max(cylinder_radius, cylinder_height)
        )
        if reason:
            return self._empty(
                "grasp_cylinder_bottom",
                reason,
                preshape=preshape,
                diameter=2.0 * cylinder_radius,
                max_aperture=self.max_aperture,
            )

        if not name:
            name = f"{reference.title()} Cylinder Bottom Grasp"

        T_ref_tsr = np.eye(4)  # bottom face is always at z = 0

        Bw = np.array(
            [
                [0.0, 0.0],  # x
                [0.0, 0.0],  # y
                [0.0, 0.0],  # z: fixed at bottom face
                [0.0, 0.0],  # roll
                [0.0, 0.0],  # pitch
                [angle_range[0], angle_range[1]],  # yaw: full rotation
            ]
        )

        # Bounded by the height as well as the fingers (#122), open and closed (#193):
        # see grasp_cylinder_top. Both limits face a cap, so they use the edge margin (#121).
        margin = self._edge_margin(clearance, max(cylinder_radius, cylinder_height))
        far_cap = cylinder_height - self._closed_reach_gain(2.0 * cylinder_radius)
        depths = self._usable_depths(margin, min(self.finger_length, far_cap) - margin, k)
        if depths is None:
            return self._empty(
                "grasp_cylinder_bottom",
                self._cap_band_reason(cylinder_height, margin),
                clearance=clearance,
                finger_length=self.finger_length,
                cylinder_height=cylinder_height,
            )
        common = dict(
            T_ref_tsr=T_ref_tsr,
            Bw=Bw,
            task="grasp",
            subject=subject,
            reference=reference,
            preshape=np.array([preshape]),
        )
        templates = []
        for i, d in enumerate(depths):
            h_palm = self.finger_length - d
            dlabel = _depth_label(len(depths), i)
            t_desc = description or (
                f"{dlabel.capitalize()} bottom grasp on {reference}: "
                f"palm {h_palm * 1000:.0f}mm below bottom, preshape {preshape * 1000:.0f}mm"
            )
            # z_EE = [0,0,+1] (approach up); identity rotation
            Tw_e = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -h_palm],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            templates.append(
                TSRTemplate(
                    Tw_e=Tw_e,
                    name=f"{name} — {dlabel}",
                    description=t_desc,
                    provenance=GraspProvenance(
                        primitive="cylinder",
                        mode="bottom",
                        approach="-z",
                        finger_orientation="diameter",
                        depth_index=i,
                        depth_count=len(depths),
                        depth=float(d),
                    ),
                    **common,
                )
            )
        return templates

    # ── Box primitives ────────────────────────────────────────────────────────

    def _validate_box(self, box_x: float, box_y: float, box_z: float, preshape: Optional[float] = None) -> None:
        self._check_dimensions(box_x=box_x, box_y=box_y, box_z=box_z)
        self._check_preshape(preshape)

    def _box_face_templates(
        self,
        T_ref_tsr: np.ndarray,
        y_ee: np.ndarray,
        z_ee: np.ndarray,
        span_dim: float,
        slide_bw_row: int,
        slide_half: float,
        preshape_user: Optional[float],
        k: int,
        clearance: float,
        box_scale: float,
        approach_extent: float,
        subject: str,
        reference: str,
        name_prefix: str,
        description: str,
        face_label: str,
        approach: str,
        finger_orientation: str,
        mode: str = "face",
        why: Optional[List[str]] = None,
    ) -> List[TSRTemplate]:
        """k depth templates for one face × finger-orientation combo.

        Fingers open along y_ee, spanning span_dim.  The gripper slides in
        slide_bw_row (world axis 0/1/2 = x/y/z) ± slide_half from the TSR
        origin.  Returns [] if the finger span can't fit around the object or
        the required preshape exceeds max_aperture. When it returns [], the reason
        is appended to ``why`` for the public method's diagnostic (see
        :meth:`_box_empty_reason`).
        """
        self._check_depth_count(k)
        # Internal per-orientation helper: return [] silently on infeasibility for
        # THIS orientation; the public box_* method logs once at its boundary if every
        # requested orientation is empty (#70). Only a NEGATIVE slide band (the
        # perpendicular dimension is thinner than 2*clearance) removes this orientation;
        # a zero-width band (dimension exactly 2*clearance) is one feasible centered
        # pose with the requested clearance on both edges, kept as a fixed [0, 0]
        # translational Bw interval per the exact-interval convention (#105, #110).
        why = [] if why is None else why
        if slide_half < 0.0:
            why.append("insufficient_clearance_band")
            return []
        preshape = preshape_user if preshape_user is not None else self._default_preshape(span_dim, clearance)
        # Straddle feasibility uses the box's characteristic scale (matching the
        # oracle's Box.scale = max dimension) so factory and contract agree (#107).
        if self._infeasibility_reason(preshape, span_dim, scale=box_scale) is not None:
            why.append("cannot_straddle")
            return []

        Bw = np.zeros((6, 2))
        Bw[slide_bw_row, 0] = -slide_half
        Bw[slide_bw_row, 1] = slide_half

        R = np.column_stack([np.cross(y_ee, z_ee), y_ee, z_ee])

        # Insertion depth is bounded by BOTH the finger length (palm clears the
        # approached face) and the box's extent along the approach axis (the fingertip
        # clears the far face), each by the edge margin m = max(clearance, 2*atol)
        # (#70, #121). The far-face limit holds for the closed fingertips too, which reach
        # further once closed across span_dim (#193). Empty band -> [] silently;
        # the public box_* method logs once at its boundary.
        margin = self._edge_margin(clearance, box_scale)  # both limits face a box face (#121)
        far_face = approach_extent - self._closed_reach_gain(span_dim)
        depths = self._usable_depths(margin, min(self.finger_length, far_face) - margin, k)
        if depths is None:
            why.append(self._cap_band_reason(approach_extent, margin))
            return []
        common = dict(
            T_ref_tsr=T_ref_tsr,
            Bw=Bw,
            task="grasp",
            subject=subject,
            reference=reference,
            preshape=np.array([preshape]),
        )
        templates = []
        for i, d in enumerate(depths):
            h_palm = self.finger_length - d
            dlabel = _depth_label(len(depths), i)
            Tw_e = np.eye(4)
            Tw_e[:3, :3] = R
            Tw_e[:3, 3] = -z_ee * h_palm  # palm is h_palm outside the face
            t_desc = description or (
                f"{dlabel.capitalize()} {face_label} grasp on {reference}: "
                f"standoff {h_palm * 1000:.0f}mm, preshape {preshape * 1000:.0f}mm"
            )
            templates.append(
                TSRTemplate(
                    Tw_e=Tw_e,
                    name=f"{name_prefix} {face_label} — {dlabel}",
                    description=t_desc,
                    provenance=GraspProvenance(
                        primitive="box",
                        mode=mode,
                        approach=approach,
                        finger_orientation=finger_orientation,
                        depth_index=i,
                        depth_count=len(depths),
                        depth=float(d),
                        metadata={"slide_axis": "xyz"[slide_bw_row], "span": float(span_dim)},
                    ),
                    **common,
                )
            )
        return templates

    @staticmethod
    def _box_empty_reason(why: List[str]) -> str:
        """The reason a whole box family is empty, from its orientations' reasons (#70, #193).

        Each orientation stops at the first constraint it fails: its slide band
        (``insufficient_clearance_band``), then straddling its span (``cannot_straddle``),
        then the closed fingertips passing the far face (``closed_reach_too_long``). The
        family reports the furthest any orientation got, which is the constraint that
        stopped the orientation nearest to feasible.
        """
        for reason in ("closed_reach_too_long", "cannot_straddle"):
            if reason in why:
                return reason
        return "insufficient_clearance_band"

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
        """Top-down grasp templates for a box — up to 2*k templates.

        Gripper approaches from above (z_EE = [0,0,-1]). TSR origin at
        z = box_z (top face). Two finger orientations:
          - fingers along x: spans box_x, slides in y
          - fingers along y: spans box_y, slides in x
        Each orientation generates k depth templates if max_aperture allows.
        """
        self._check_depth_count(k)
        clearance = self._resolve_clearance(clearance, self.finger_length)
        self._validate_box(box_x, box_y, box_z, preshape)
        # One margin for the diagnostic precheck AND for generation (#121): prechecking
        # with the raw clearance would report cannot_straddle where the real cause is an
        # empty band under the effective margin. This is the open hand's band, which every
        # orientation shares; the closed reach depends on each orientation's span, so
        # generation judges it and the family's reason names it (#193).
        box_scale = max(box_x, box_y, box_z)
        margin = self._edge_margin(clearance, box_scale)
        if self._usable_depths(margin, min(self.finger_length, box_z) - margin, k) is None:
            return self._empty(
                "grasp_box_top", "insufficient_clearance_band", clearance=clearance, margin=margin, box_z=box_z
            )
        if preshape is not None and preshape > self.max_aperture:
            return self._empty("grasp_box_top", "exceeds_aperture", preshape=preshape, max_aperture=self.max_aperture)
        # Per-orientation feasibility (#70): span-x slides in y (band hy), span-y slides
        # in x (band hx). A thin dimension empties only its dependent orientation; keep
        # the perpendicular one instead of rejecting the whole face.
        hx, hy = box_x / 2.0 - margin, box_y / 2.0 - margin  # slide edges use the same margin

        if not name:
            name = f"{reference.title()} Box Top Grasp"

        T = np.eye(4)
        T[2, 3] = box_z
        z_ee = np.array([0.0, 0.0, -1.0])

        why: List[str] = []  # each empty orientation's reason (#193)
        kw = dict(
            preshape_user=preshape,
            k=k,
            clearance=clearance,
            box_scale=box_scale,
            approach_extent=box_z,
            subject=subject,
            reference=reference,
            name_prefix=name,
            description=description,
            why=why,
        )
        templates = self._box_face_templates(
            T,
            np.array([1.0, 0.0, 0.0]),
            z_ee,
            box_x,
            1,
            hy,
            **kw,
            face_label="+z (span-x)",
            approach="+z",
            finger_orientation="x",
            mode="top",
        ) + self._box_face_templates(
            T,
            np.array([0.0, 1.0, 0.0]),
            z_ee,
            box_y,
            0,
            hx,
            **kw,
            face_label="+z (span-y)",
            approach="+z",
            finger_orientation="y",
            mode="top",
        )
        if not templates:
            # Whole family empty: the box is too thin to slide in either orientation,
            # every orientation's span cannot straddle within the aperture, or the closed
            # fingertips would pass the bottom face.
            reason = self._box_empty_reason(why)
            return self._empty(
                "grasp_box_top", reason, box_x=box_x, box_y=box_y, clearance=clearance, max_aperture=self.max_aperture
            )
        return templates

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
        """Bottom-up grasp templates for a box — up to 2*k templates.

        Gripper approaches from below (z_EE = [0,0,+1]). TSR origin at
        z = 0 (bottom face). Two finger orientations:
          - fingers along x: spans box_x, slides in y
          - fingers along y: spans box_y, slides in x
        Each orientation generates k depth templates if max_aperture allows.
        """
        self._check_depth_count(k)
        clearance = self._resolve_clearance(clearance, self.finger_length)
        self._validate_box(box_x, box_y, box_z, preshape)
        # One margin for the diagnostic precheck AND for generation (#121): prechecking
        # with the raw clearance would report cannot_straddle where the real cause is an
        # empty band under the effective margin. This is the open hand's band, which every
        # orientation shares; the closed reach depends on each orientation's span, so
        # generation judges it and the family's reason names it (#193).
        box_scale = max(box_x, box_y, box_z)
        margin = self._edge_margin(clearance, box_scale)
        if self._usable_depths(margin, min(self.finger_length, box_z) - margin, k) is None:
            return self._empty(
                "grasp_box_bottom", "insufficient_clearance_band", clearance=clearance, margin=margin, box_z=box_z
            )
        if preshape is not None and preshape > self.max_aperture:
            return self._empty(
                "grasp_box_bottom", "exceeds_aperture", preshape=preshape, max_aperture=self.max_aperture
            )
        # Per-orientation feasibility (#70): a thin dimension empties only its dependent
        # orientation, keeping the perpendicular one.
        hx, hy = box_x / 2.0 - margin, box_y / 2.0 - margin  # slide edges use the same margin

        if not name:
            name = f"{reference.title()} Box Bottom Grasp"

        T = np.eye(4)  # bottom face is always at z=0
        z_ee = np.array([0.0, 0.0, 1.0])

        why: List[str] = []  # each empty orientation's reason (#193)
        kw = dict(
            preshape_user=preshape,
            k=k,
            clearance=clearance,
            box_scale=box_scale,
            approach_extent=box_z,
            subject=subject,
            reference=reference,
            name_prefix=name,
            description=description,
            why=why,
        )
        templates = self._box_face_templates(
            T,
            np.array([1.0, 0.0, 0.0]),
            z_ee,
            box_x,
            1,
            hy,
            **kw,
            face_label="-z (span-x)",
            approach="-z",
            finger_orientation="x",
            mode="bottom",
        ) + self._box_face_templates(
            T,
            np.array([0.0, 1.0, 0.0]),
            z_ee,
            box_y,
            0,
            hx,
            **kw,
            face_label="-z (span-y)",
            approach="-z",
            finger_orientation="y",
            mode="bottom",
        )
        if not templates:
            reason = self._box_empty_reason(why)
            return self._empty(
                "grasp_box_bottom",
                reason,
                box_x=box_x,
                box_y=box_y,
                clearance=clearance,
                max_aperture=self.max_aperture,
            )
        return templates

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
        """Grasp templates for the ±x faces of a box — up to 4*k templates.

        Two approach directions (+x and -x), each with two finger orientations:
          - fingers along y: spans box_y, slides in z
          - fingers along z: spans box_z, slides in y
        Each valid orientation (max_aperture allows the span) generates k templates.
        """
        self._check_depth_count(k)
        clearance = self._resolve_clearance(clearance, self.finger_length)
        self._validate_box(box_x, box_y, box_z, preshape)
        # One margin for the diagnostic precheck AND for generation (#121): prechecking
        # with the raw clearance would report cannot_straddle where the real cause is an
        # empty band under the effective margin. This is the open hand's band, which every
        # orientation shares; the closed reach depends on each orientation's span, so
        # generation judges it and the family's reason names it (#193).
        box_scale = max(box_x, box_y, box_z)
        margin = self._edge_margin(clearance, box_scale)
        if self._usable_depths(margin, min(self.finger_length, box_x) - margin, k) is None:
            return self._empty(
                "grasp_box_face_x", "insufficient_clearance_band", clearance=clearance, margin=margin, box_x=box_x
            )
        if preshape is not None and preshape > self.max_aperture:
            return self._empty(
                "grasp_box_face_x", "exceeds_aperture", preshape=preshape, max_aperture=self.max_aperture
            )
        # Per-orientation feasibility (#70): span-y slides in z (band hz_half), span-z
        # slides in y (band hy). A thin dimension empties only its dependent orientation.
        hy = box_y / 2.0 - margin  # slide edges use the same margin
        hz_half = box_z / 2.0 - margin

        if not name:
            name = f"{reference.title()} Box X-Face Grasp"

        T_pos = np.eye(4)
        T_pos[0, 3] = box_x / 2.0
        T_pos[2, 3] = box_z / 2.0
        T_neg = np.eye(4)
        T_neg[0, 3] = -box_x / 2.0
        T_neg[2, 3] = box_z / 2.0

        why: List[str] = []  # each empty orientation's reason (#193)
        kw = dict(
            preshape_user=preshape,
            k=k,
            clearance=clearance,
            box_scale=box_scale,
            approach_extent=box_x,
            subject=subject,
            reference=reference,
            name_prefix=name,
            description=description,
            why=why,
        )
        templates = []
        for T_ref, z_ee, sign in (
            (T_pos, np.array([-1.0, 0.0, 0.0]), "+x"),
            (T_neg, np.array([1.0, 0.0, 0.0]), "-x"),
        ):
            templates += self._box_face_templates(
                T_ref,
                np.array([0.0, 1.0, 0.0]),
                z_ee,
                box_y,
                2,
                hz_half,
                **kw,
                face_label=f"{sign} (span-y)",
                approach=sign,
                finger_orientation="y",
            )
            templates += self._box_face_templates(
                T_ref,
                np.array([0.0, 0.0, 1.0]),
                z_ee,
                box_z,
                1,
                hy,
                **kw,
                face_label=f"{sign} (span-z)",
                approach=sign,
                finger_orientation="z",
            )
        if not templates:
            reason = self._box_empty_reason(why)
            return self._empty(
                "grasp_box_face_x",
                reason,
                box_y=box_y,
                box_z=box_z,
                clearance=clearance,
                max_aperture=self.max_aperture,
            )
        return templates

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
        """Grasp templates for the ±y faces of a box — up to 4*k templates.

        Two approach directions (+y and -y), each with two finger orientations:
          - fingers along x: spans box_x, slides in z
          - fingers along z: spans box_z, slides in x
        Each valid orientation (max_aperture allows the span) generates k templates.
        """
        self._check_depth_count(k)
        clearance = self._resolve_clearance(clearance, self.finger_length)
        self._validate_box(box_x, box_y, box_z, preshape)
        # One margin for the diagnostic precheck AND for generation (#121): prechecking
        # with the raw clearance would report cannot_straddle where the real cause is an
        # empty band under the effective margin. This is the open hand's band, which every
        # orientation shares; the closed reach depends on each orientation's span, so
        # generation judges it and the family's reason names it (#193).
        box_scale = max(box_x, box_y, box_z)
        margin = self._edge_margin(clearance, box_scale)
        if self._usable_depths(margin, min(self.finger_length, box_y) - margin, k) is None:
            return self._empty(
                "grasp_box_face_y", "insufficient_clearance_band", clearance=clearance, margin=margin, box_y=box_y
            )
        if preshape is not None and preshape > self.max_aperture:
            return self._empty(
                "grasp_box_face_y", "exceeds_aperture", preshape=preshape, max_aperture=self.max_aperture
            )
        # Per-orientation feasibility (#70): span-x slides in z (band hz_half), span-z
        # slides in x (band hx). A thin dimension empties only its dependent orientation.
        hx = box_x / 2.0 - margin  # slide edges use the same margin
        hz_half = box_z / 2.0 - margin

        if not name:
            name = f"{reference.title()} Box Y-Face Grasp"

        T_pos = np.eye(4)
        T_pos[1, 3] = box_y / 2.0
        T_pos[2, 3] = box_z / 2.0
        T_neg = np.eye(4)
        T_neg[1, 3] = -box_y / 2.0
        T_neg[2, 3] = box_z / 2.0

        why: List[str] = []  # each empty orientation's reason (#193)
        kw = dict(
            preshape_user=preshape,
            k=k,
            clearance=clearance,
            box_scale=box_scale,
            approach_extent=box_y,
            subject=subject,
            reference=reference,
            name_prefix=name,
            description=description,
            why=why,
        )
        templates = []
        for T_ref, z_ee, sign in (
            (T_pos, np.array([0.0, -1.0, 0.0]), "+y"),
            (T_neg, np.array([0.0, 1.0, 0.0]), "-y"),
        ):
            templates += self._box_face_templates(
                T_ref,
                np.array([1.0, 0.0, 0.0]),
                z_ee,
                box_x,
                2,
                hz_half,
                **kw,
                face_label=f"{sign} (span-x)",
                approach=sign,
                finger_orientation="x",
            )
            templates += self._box_face_templates(
                T_ref,
                np.array([0.0, 0.0, 1.0]),
                z_ee,
                box_z,
                0,
                hx,
                **kw,
                face_label=f"{sign} (span-z)",
                approach=sign,
                finger_orientation="z",
            )
        if not templates:
            reason = self._box_empty_reason(why)
            return self._empty(
                "grasp_box_face_y",
                reason,
                box_x=box_x,
                box_z=box_z,
                clearance=clearance,
                max_aperture=self.max_aperture,
            )
        return templates

    # ── Sphere primitives ─────────────────────────────────────────────────────

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
        """Full-sphere grasp templates — k templates.

        Approach from any direction in 3-D. Each template's rotational Bw box
        (ZYX, ``R = Rz(yaw) Ry(pitch) Rx(roll)``) covers the orientation set:

          roll  ∈ [0, 2π]        — finger orientation around the approach axis
          pitch ∈ [-π/2, π/2]   — elevation of the approach direction (no double-cover)
          yaw   ∈ angle_range    — azimuth of the approach direction (default full 360°)

        **Coverage.** With the default ``angle_range`` the set is all of SO(3). A
        restricted ``angle_range`` bounds the Euler yaw coordinate: the represented
        set is every orientation whose approach direction lies in the **lune**
        ``azimuth ∈ angle_range`` (the wedge between two meridians, poles included),
        with any spin about the approach axis. It is not a spherical cap.

        **Distribution.** Coverage says nothing about sampling. ``TSR.sample``
        draws roll/pitch/yaw uniformly, which is **not** Haar-uniform on SO(3) and
        concentrates approach directions near the poles. Use
        :func:`tsr.sampling.sample_haar` for rotations that are Haar-uniform over the
        represented set, i.e. approach directions uniform by area over the sphere
        or lune (#72).

        TSR origin at sphere center. k discrete depths baked into Tw_e.

        Returns [] if preshape <= sphere diameter. Raises ValueError for
        invalid geometry.
        """
        self._check_depth_count(k)
        self._check_angle_range(angle_range)
        self._check_dimensions(object_radius=object_radius)
        self._check_preshape(preshape)
        clearance = self._resolve_clearance(clearance, min(self.finger_length, object_radius))
        if preshape is None:
            preshape = self._default_preshape(2.0 * object_radius, clearance)
        reason = self._infeasibility_reason(preshape, 2.0 * object_radius, scale=object_radius)
        if reason:
            return self._empty(
                "grasp_sphere", reason, preshape=preshape, diameter=2.0 * object_radius, max_aperture=self.max_aperture
            )

        if not name:
            name = f"{reference.title()} Sphere Grasp"

        T_ref_tsr = np.eye(4)  # origin at sphere center

        Bw = np.array(
            [
                [0.0, 0.0],  # x: no translational freedom
                [0.0, 0.0],  # y
                [0.0, 0.0],  # z
                [0.0, 2 * np.pi],  # roll: full rotation around approach
                [-np.pi / 2, np.pi / 2],  # pitch: full elevation (no double-cover)
                [angle_range[0], angle_range[1]],  # yaw: azimuthal freedom
            ]
        )

        # Radial depth band, identical to the cylinder-side derivation (#69, #108, #193):
        # empty because the fingers are too short (finger_length < radius + clearance,
        # palm inside the sphere), because an excessive clearance relative to the
        # radius empties the far-surface limit, or because the fingertips closed across
        # the diameter would pass it; _radial_band_reason names the cause.
        depth_min = object_radius
        depth_max = min(self.finger_length, 2 * object_radius - self._closed_reach_gain(2 * object_radius)) - clearance
        depths = self._usable_depths(depth_min, depth_max, k)
        if depths is None:
            return self._empty(
                "grasp_sphere",
                self._radial_band_reason(object_radius, clearance),
                finger_length=self.finger_length,
                radius=object_radius,
                clearance=clearance,
            )

        common = dict(
            T_ref_tsr=T_ref_tsr,
            Bw=Bw,
            task="grasp",
            subject=subject,
            reference=reference,
            preshape=np.array([preshape]),
        )
        if angle_range[1] - angle_range[0] >= 2 * np.pi:
            coverage = "full SO(3)"
        else:
            coverage = f"azimuth lune {np.degrees(angle_range[0]):.0f}°–{np.degrees(angle_range[1]):.0f}°"
        templates = []
        for i, d in enumerate(depths):
            ro = object_radius + self.finger_length - d
            dlabel = _depth_label(len(depths), i)
            # Approach along -x in TSR frame; standoff ro baked into Tw_e.
            # Bw roll/pitch/yaw rotates this to any direction on the sphere.
            Tw_e = np.array(
                [
                    [0.0, 0.0, -1.0, ro],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            t_desc = description or (
                f"{dlabel.capitalize()} sphere grasp on {reference}: "
                f"standoff {ro * 1000:.0f}mm from center, {coverage}, "
                f"preshape {preshape * 1000:.0f}mm"
            )
            templates.append(
                TSRTemplate(
                    Tw_e=Tw_e,
                    name=f"{name} — {dlabel}",
                    description=t_desc,
                    provenance=GraspProvenance(
                        primitive="sphere",
                        mode="surface",
                        approach="radial",
                        finger_orientation="diameter",
                        depth_index=i,
                        depth_count=len(depths),
                        depth=float(d),
                    ),
                    **common,
                )
            )
        return templates

    # ── Torus primitives ─────────────────────────────────────────────────────

    def _validate_torus(self, torus_radius: float, tube_radius: float, preshape: Optional[float] = None) -> None:
        self._check_dimensions(torus_radius=torus_radius, tube_radius=tube_radius)
        self._check_preshape(preshape)
        if tube_radius >= torus_radius:
            raise ValueError(
                f"tube_radius ({tube_radius}) must be < torus_radius ({torus_radius}) "
                "to avoid a self-intersecting torus"
            )

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

        The gripper approaches the tube from ``n_minor`` discrete minor angles ``α``
        in the tube cross-section plane (spanned by the radial and vertical axes),
        sampled from ``minor_angle_range`` (default ``[−π/2, +π/2]``):

            α = −π/2  from below    (matches span-bottom geometry)
            α =  0    from outside  (pure radial equatorial approach)
            α = +π/2  from above    (matches span-top geometry)

        **Coverage.** ``[−π/2, +π/2]`` is the externally accessible **outer half** of
        the tube cross-section (the side facing away from the ring axis). Inner-hole
        approaches (``|α| > π/2``) require the hand to fit through the ring hole and
        are **out of scope**: ``minor_angle_range`` must lie within ``[−π/2, +π/2]``.
        The represented set is exactly the sampled minor angles crossed with the full
        azimuth (``angle_range``) around the ring.

        ``n_minor == 1`` samples the **centre** of ``minor_angle_range`` (the pure
        equatorial approach ``α = 0`` for the default range), not an endpoint.

        Two hand flip variants (fingers open in ±tangential direction) per
        (α, depth). Gripper position in TSR frame at depth d, minor angle α:
            tx = R + (r + fl − d) · cos α ,  tz = (r + fl − d) · sin α
        z_EE = (−cos α, 0, −sin α)   (points toward the tube centre).

        The depth band applies the same reach/clearance rule as the cylinder-side
        and sphere families: ``d ∈ [tube_radius, min(2·tube_radius − g, finger_length) −
        clearance]`` keeps the palm a clearance outside the tube surface, and the
        fingertips a clearance inside its far side, open and closed across it
        (``g = reach(2·tube_radius) − finger_length``, #193).

        Args:
            n_minor:           Discrete approach angles across the tube cross-section
                               (default 5).
            minor_angle_range: Closed minor-angle interval within the outer half
                               (default ``[−π/2, +π/2]``).
            clearance:         Safety buffer [m]. Defaults to 10% of finger_length.

        Returns 2*k*n_minor TSRTemplates. Returns [] with ``finger_too_short`` if
        ``finger_length < tube_radius + clearance``, with ``insufficient_clearance_band``
        if reach is ample but ``clearance > tube_radius`` empties the far-surface limit,
        with ``closed_reach_too_long`` if only the closed fingertips would pass it,
        or with the straddle reason if the tube diameter cannot be spanned. Raises
        ValueError for invalid geometry, including a ``minor_angle_range`` endpoint
        strictly outside ``[−π/2, +π/2]``.
        """
        self._check_depth_count(k)
        self._check_angle_range(angle_range)
        self._check_count("n_minor", n_minor)
        self._check_angle_range(minor_angle_range)
        # Exact closed-interval policy (#105, #113): endpoints inside or exactly on
        # [-pi/2, pi/2] are valid; anything strictly outside raises. No tolerance.
        if minor_angle_range[0] < -np.pi / 2 or minor_angle_range[1] > np.pi / 2:
            raise ValueError(
                f"minor_angle_range must lie within the externally accessible outer half "
                f"[-pi/2, pi/2]; inner-hole approaches are out of scope, got {minor_angle_range}"
            )
        self._validate_torus(torus_radius, tube_radius, preshape)
        clearance = self._resolve_clearance(clearance, min(self.finger_length, tube_radius))
        if preshape is None:
            preshape = self._default_preshape(2.0 * tube_radius, clearance)
        reason = self._infeasibility_reason(preshape, 2.0 * tube_radius, scale=torus_radius + tube_radius)
        if reason:
            return self._empty(
                "grasp_torus_side",
                reason,
                preshape=preshape,
                tube_diameter=2.0 * tube_radius,
                max_aperture=self.max_aperture,
            )

        if not name:
            name = f"{reference.title()} Torus Side Grasp"

        T_ref_tsr = np.eye(4)  # origin at torus center

        Bw = np.array(
            [
                [0.0, 0.0],  # x: no freedom
                [0.0, 0.0],  # y: no freedom
                [0.0, 0.0],  # z: baked into Tw_e
                [0.0, 0.0],  # roll: baked into Tw_e
                [0.0, 0.0],  # pitch: baked into Tw_e
                [angle_range[0], angle_range[1]],  # yaw: full azimuthal freedom
            ]
        )

        # Reach/clearance band (same rule as cylinder-side and sphere, #69/#71/#193):
        # fingertips reach the tube centre (d >= tube_radius), the palm stays a
        # clearance outside the tube surface (d <= L - clearance), and so do the
        # fingertips inside its far side, closed across the tube as well as open
        # (d <= 2r - (reach(2r) - L) - clearance). The band is empty because the fingers
        # are too short (finger_length < tube_radius + clearance), because an excessive
        # clearance relative to the tube radius empties the far-surface limit, or because
        # the closed fingertips would pass it; _radial_band_reason names the failed
        # constraint (#114).
        far_side = 2 * tube_radius - self._closed_reach_gain(2 * tube_radius)
        depths = self._usable_depths(tube_radius, min(far_side, self.finger_length) - clearance, k)
        if depths is None:
            return self._empty(
                "grasp_torus_side",
                self._radial_band_reason(tube_radius, clearance),
                finger_length=self.finger_length,
                tube_radius=tube_radius,
                clearance=clearance,
            )
        minor_angles = self._sample_range(minor_angle_range, n_minor)

        common = dict(
            T_ref_tsr=T_ref_tsr,
            Bw=Bw,
            task="grasp",
            subject=subject,
            reference=reference,
            preshape=np.array([preshape]),
        )
        templates = []
        for mi, alpha in enumerate(minor_angles):
            ca, sa = np.cos(alpha), np.sin(alpha)
            a_label = f"α={np.degrees(alpha):.0f}°"
            for i, d in enumerate(depths):
                # Distance from tube center to palm
                ro_minor = tube_radius + self.finger_length - d
                # Gripper position in TSR frame (before yaw rotation)
                tx = torus_radius + ro_minor * ca
                tz = ro_minor * sa
                dlabel = _depth_label(len(depths), i)
                # z_EE = (−cosα, 0, −sinα); y_EE in radial-vertical plane ⊥ z_EE
                # y_EE ⊥ z_EE in span{x̂,ẑ}: y_EE = (−sinα, 0, cosα)
                # x_EE = y_EE × z_EE = (0, −1, 0)  [same for all α]
                # Flip π: y_EE = (+sinα, 0, −cosα), x_EE = (0, +1, 0)
                Tw_e_0 = np.array(
                    [
                        [0.0, -sa, -ca, tx],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, ca, -sa, tz],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                Tw_e_pi = np.array(
                    [
                        [0.0, sa, -ca, tx],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -ca, -sa, tz],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                for Tw_e, flip_label, flip_variant in (
                    (Tw_e_0, "flip 0°", "flip0"),
                    (Tw_e_pi, "flip 180°", "flippi"),
                ):
                    t_desc = description or (
                        f"{dlabel.capitalize()} torus side grasp on {reference}: "
                        f"{a_label}, ro={ro_minor * 1000:.0f}mm from tube center, "
                        f"{flip_label}, preshape {preshape * 1000:.0f}mm"
                    )
                    templates.append(
                        TSRTemplate(
                            Tw_e=Tw_e,
                            name=f"{name} — {a_label}, {dlabel}, {flip_label}",
                            description=t_desc,
                            provenance=GraspProvenance(
                                primitive="torus",
                                mode="side",
                                approach="tube",
                                finger_orientation="tangential",
                                depth_index=i,
                                depth_count=len(depths),
                                depth=float(d),
                                symmetry=flip_variant,
                                metadata={
                                    "minor_index": mi,
                                    "minor_count": len(minor_angles),
                                    "minor_angle": float(alpha),
                                },
                            ),
                            **common,
                        )
                    )
        return templates

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

        Approach from above and below, fingers spanning the outer diameter
        2*(R+r). Full yaw freedom covers all finger orientations. Silently
        returns [] if the outer diameter + clearance exceeds max_aperture.
        """
        self._check_depth_count(k)
        self._validate_torus(torus_radius, tube_radius, preshape)
        clearance = self._resolve_clearance(clearance, self.finger_length)
        if preshape is None:
            preshape = self._default_preshape(2.0 * (torus_radius + tube_radius), clearance)
        outer_diameter = 2.0 * (torus_radius + tube_radius)
        reason = self._infeasibility_reason(preshape, outer_diameter)
        if reason:
            return self._empty(
                "grasp_torus_span",
                reason,
                preshape=preshape,
                outer_diameter=outer_diameter,
                max_aperture=self.max_aperture,
            )

        if not name:
            name = f"{reference.title()} Torus Span Grasp"

        # Bw: yaw free [0, 2π] for full finger-orientation freedom; all else fixed
        Bw = np.zeros((6, 2))
        Bw[5, 1] = 2 * np.pi

        # Reach band (#71): the fingertip must reach the equatorial plane (z = 0, the
        # widest outer diameter), which needs an insertion d >= tube_radius, while the
        # palm stands off a clearance above the tube top (d <= finger_length -
        # clearance). Empty when finger_length < tube_radius + clearance. The fingers
        # pass outside the torus rather than into it, so no far face bounds the depth and
        # the closed reach does not enter (#193).
        depths = self._usable_depths(tube_radius, self.finger_length - clearance, k)
        if depths is None:
            return self._empty(
                "grasp_torus_span",
                "finger_too_short",
                finger_length=self.finger_length,
                tube_radius=tube_radius,
                clearance=clearance,
            )
        templates = []
        for i, d in enumerate(depths):
            h_palm = self.finger_length - d
            dlabel = _depth_label(len(depths), i)

            # Top: z_EE = [0,0,-1]; TSR origin at tube top (z = +tube_r)
            T_top = np.eye(4)
            T_top[2, 3] = tube_radius
            Tw_e_top = np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, h_palm],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            t_desc = description or (
                f"{dlabel.capitalize()} torus span top on {reference}: "
                f"palm {h_palm * 1000:.0f}mm above torus, preshape {preshape * 1000:.0f}mm"
            )
            templates.append(
                TSRTemplate(
                    T_ref_tsr=T_top,
                    Bw=Bw,
                    Tw_e=Tw_e_top,
                    task="grasp",
                    subject=subject,
                    reference=reference,
                    preshape=np.array([preshape]),
                    name=f"{name} top — {dlabel}",
                    description=t_desc,
                    provenance=GraspProvenance(
                        primitive="torus",
                        mode="span",
                        approach="+z",
                        finger_orientation="diameter",
                        depth_index=i,
                        depth_count=len(depths),
                        depth=float(d),
                    ),
                )
            )

            # Bottom: z_EE = [0,0,+1]; TSR origin at tube bottom (z = -tube_r)
            T_bot = np.eye(4)
            T_bot[2, 3] = -tube_radius
            Tw_e_bot = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -h_palm],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            t_desc = description or (
                f"{dlabel.capitalize()} torus span bottom on {reference}: "
                f"palm {h_palm * 1000:.0f}mm below torus, preshape {preshape * 1000:.0f}mm"
            )
            templates.append(
                TSRTemplate(
                    T_ref_tsr=T_bot,
                    Bw=Bw,
                    Tw_e=Tw_e_bot,
                    task="grasp",
                    subject=subject,
                    reference=reference,
                    preshape=np.array([preshape]),
                    name=f"{name} bottom — {dlabel}",
                    description=t_desc,
                    provenance=GraspProvenance(
                        primitive="torus",
                        mode="span",
                        approach="-z",
                        finger_orientation="diameter",
                        depth_index=i,
                        depth_count=len(depths),
                        depth=float(d),
                    ),
                )
            )
        return templates


class Robotiq2F140(ParallelJawGripper):
    """Robotiq 2F-140 parallel gripper.

    Fixed hardware parameters: ``FINGER_LENGTH = 0.114 m``, ``MAX_APERTURE = 0.128 m``.

    ``FINGER_LENGTH`` is the palm→pad-tip reach along the approach axis — the same
    convention as :class:`Robotiq2F85` and :class:`FrankaHand`. Measured from
    geodude_assets ``2f140.xml`` (fully open): the ``grasp_site`` (TSR palm) is at
    ``base_mount + 0.100 m`` (6 mm past the 0.094 m housing forward edge, rounded up
    for clearance) and the pad tip is at ``base_mount + 0.214 m``, so
    ``FINGER_LENGTH = 0.214 − 0.100 = 0.114 m``.

    ``MAX_APERTURE = 0.128 m`` is the inner-face-to-inner-face gap between the pads
    at full open (measured from 2f140.xml), which is what ``preshape ≤ MAX_APERTURE``
    needs — the object must fit *between* the inner faces. The manufacturer's nominal
    0.140 m is the outer/advertised figure and overstates the usable gap (#60), the
    same distinction as the :class:`Robotiq2F85`'s 0.085 m.

    ``CLOSED_REACH`` is the fingertips' reach as the hand closes (#193). The 2F-140 is a
    four-bar hand whose pads swing forward as they close: closed on a 72 mm object the
    fingertips reach about 132 mm past the palm, 18 mm more than ``FINGER_LENGTH``, and
    closed to nothing 139 mm. Measured from geodude_assets ``2f140.xml`` (395991d, MuJoCo
    3.14.0) with ``tools/finger_reach.py --palm-offset 0.100``: the width is the gap between
    the pads' contact faces, the reach how far the fingers' collision geometry (mesh links
    bounded by their own-frame AABBs) extends past the palm. The fully open point is the
    model's rest configuration (128.4 mm, 114.3 mm), which matches ``MAX_APERTURE`` and
    ``FINGER_LENGTH``. Eleven of the tool's levels; linear interpolation between them is
    within 0.1 mm of every measured level. The curve is the model's and has not been checked
    against Robotiq's finger drawings.

    Outputs poses in the canonical TSR EE frame (z=approach, y=finger-opening,
    x=palm normal). The corresponding MuJoCo model (geodude_assets 2f140.xml)
    defines a ``grasp_site`` at the palm with this orientation baked in — use
    that site as the arm's ``ee_site`` so IK targets the canonical frame
    directly.
    """

    FINGER_LENGTH = 0.114
    MAX_APERTURE = 0.128  # inner-face gap at full open (nominal/outer is 0.140); see #60
    CLOSED_REACH = (  # (pad gap, reach past the palm) [m]; see the class docstring
        (0.0001, 0.1387),
        (0.0125, 0.1384),
        (0.0252, 0.1377),
        (0.0375, 0.1367),
        (0.0501, 0.1352),
        (0.0622, 0.1334),
        (0.0744, 0.1311),
        (0.0846, 0.1288),
        (0.0970, 0.1256),
        (0.1075, 0.1223),
        (0.1284, 0.1143),
    )

    # Distance from base_mount origin to the TSR palm (grasp_site) along the
    # approach axis, for callers placing an ee_site in an MjSpec (cf. 2F-85).
    PALM_OFFSET_FROM_BASE_MOUNT = 0.100

    def __init__(self):
        super().__init__(
            finger_length=self.FINGER_LENGTH,
            max_aperture=self.MAX_APERTURE,
        )


class Robotiq2F85(ParallelJawGripper):
    """Robotiq 2F-85 parallel gripper.

    Fixed hardware parameters measured from mujoco_menagerie's
    ``robotiq_2f85/2f85.xml`` via the disembodied-hand visualizer
    (``mj_manipulator/scripts/visualize_grasps.py``), which walks each
    collision geom's AABB along the approach axis:

    - ``FINGER_LENGTH = 0.059 m`` — distance from the **forward edge of
      the base housing** (the TSR "palm") to the **pad
      tip** (finger reach along approach; the palm→pad-tip convention shared by
      all the named grippers).
      The 2F-85's ``base`` body is a chunky housing that extends ~94 mm
      past the base_mount plate along the approach axis — the
      mechanism (drivers/couplers/spring links) sits inside that block.
      FINGER_LENGTH must be measured *from the forward edge of the
      housing*, not from base_mount, so the TSR's "everything behind
      the palm is clear space" assumption holds.
    - ``MAX_APERTURE = 0.085 m`` — inner-face to inner-face distance
      between the pads when fully open. This is what the TSR actually
      needs for ``preshape ≤ MAX_APERTURE`` (the object has to fit
      *between* the inner faces, not between outer edges). Old
      ``0.098 m`` was outer-face to outer-face and would claim the
      gripper fits objects it can't.

    ``CLOSED_REACH`` is the fingertips' reach as the hand closes (#193). Like the 2F-140,
    the 2F-85 is a four-bar hand whose pads swing forward as they close: closed on a 40 mm
    object the fingertips reach about 71 mm past the palm, 12 mm more than
    ``FINGER_LENGTH``. Measured from the same ``2f85.xml`` (mujoco_menagerie feadf76,
    MuJoCo 3.14.0) with ``tools/finger_reach.py --palm-offset 0.094``, as for the 2F-140:
    the width is the gap between the pads' contact faces, the reach how far the fingers'
    collision geometry extends past the palm. Fully open it reads 85.4 mm and 59.2 mm,
    matching ``MAX_APERTURE`` and ``FINGER_LENGTH``. Eleven of the tool's levels; linear
    interpolation between them is within 0.1 mm of every measured level.

    Outputs poses in the canonical TSR EE frame (z=approach,
    y=finger-opening, x=palm normal). The arm's ``ee_site`` (a.k.a.
    ``grasp_site``) must be placed at the forward edge of the base
    housing — i.e., at ``base_mount + [0, 0, 0.094]`` in the 2F-85
    frame, before the -90° about-z alignment rotation. See
    ``mj_manipulator/demos/iiwa14_setup.py`` for the canonical attach.
    """

    FINGER_LENGTH = 0.059
    MAX_APERTURE = 0.085
    CLOSED_REACH = (  # (pad gap, reach past the palm) [m]; see the class docstring
        (0.0004, 0.0726),
        (0.0098, 0.0728),
        (0.0191, 0.0725),
        (0.0284, 0.0720),
        (0.0375, 0.0711),
        (0.0464, 0.0698),
        (0.0551, 0.0682),
        (0.0635, 0.0663),
        (0.0716, 0.0641),
        (0.0808, 0.0610),
        (0.0854, 0.0593),
    )

    # Distance from base_mount origin to the TSR palm along the
    # approach axis. Callers placing a grasp_site in an MjSpec should
    # offset it by this value (plus ``base_mount.pos``) so the arm's
    # ee_site lands on the TSR palm, not inside the housing.
    PALM_OFFSET_FROM_BASE_MOUNT = 0.094

    def __init__(self):
        super().__init__(
            finger_length=self.FINGER_LENGTH,
            max_aperture=self.MAX_APERTURE,
        )


class FrankaHand(ParallelJawGripper):
    """Franka Emika Panda hand (parallel jaw gripper).

    Fixed hardware parameters measured from the menagerie ``hand.xml``
    model via ``mj_manipulator/scripts/validate_gripper.py``, which walks
    each collision geom's AABB along the approach axis:

    - ``FINGER_LENGTH = 0.037 m`` — distance from the **forward edge of
      the hand body** (the TSR "palm") to the **pad
      tip** (finger reach along approach; the palm→pad-tip convention shared by
      all the named grippers).
      The menagerie ``hand`` body extends 16.9 mm past the finger-joint
      origin along the approach axis (the metal collar around the
      finger mounts). ``FINGER_LENGTH`` must be measured from *that*
      forward edge so the TSR's "everything behind the palm is clear
      space" assumption holds — same failure mode that bit the 2F-85.
    - ``MAX_APERTURE = 0.080 m`` — 2 × 40 mm joint range. Unchanged.

    The canonical EE frame (z=approach, y=finger-opening, x=palm normal)
    matches the Franka hand body axes directly. The ``add_franka_ee_site``
    helper in ``mj_manipulator.arms.franka`` places a ``grasp_site`` at
    ``hand + [0, 0, 0.0753]`` (= finger-joint origin + 17 mm forward)
    with identity orientation. Use that site as the arm's ``ee_site``.
    """

    FINGER_LENGTH = 0.037  # palm (hand-body forward edge) → pad tip [m]
    MAX_APERTURE = 0.080  # 2 × 40 mm joint range [m]

    # Distance from the ``hand`` body origin to the TSR palm along the
    # approach axis. Callers placing a grasp_site in an MjSpec should
    # offset it by ``finger_joint_offset + PALM_OFFSET_FROM_HAND``.
    # For menagerie hand.xml, finger_joint_offset = 0.0584 m, so the
    # canonical grasp_site position is hand + [0, 0, 0.0753].
    PALM_OFFSET_FROM_HAND = 0.0753

    def __init__(self):
        super().__init__(
            finger_length=self.FINGER_LENGTH,
            max_aperture=self.MAX_APERTURE,
        )
