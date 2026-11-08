# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""OpeningPlacer: release an object into a container through its opening (#200).

The release contract is stated in ``docs/ARCHITECTURE.md``. In short, for **every** pose
a returned region admits, the object is in one of its resting orientations, the whole
object lies inside the opening by the clearance, and its lowest point is within the
requested heights of the rim plane.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from ..core.utils import length_atol
from ..template import TSRTemplate
from .shape import Shape
from .stable_placer import StablePlacer, _check_finite, _check_positive_finite

logger = logging.getLogger(__name__)

#: The surface ``StablePlacer`` finds rests on: wide enough that it drops no rest for its
#: footprint, which the opening then bounds instead.
_UNBOUNDED_M = 1.0e6


class OpeningPlacer:
    """Release regions for an object into a container through a declared opening.

    Frame convention:
        The opening frame is centred on the rim plane, with z pointing out of the
        container, as ``prl_assets``' ``placement.openings`` declares it. The container
        is taken to be the opening's section, straight down to its floor ``depth``
        below the rim; walls that narrow below the rim are not the opening's to know.

    An object is released only in a resting orientation, as :meth:`StablePlacer.place`
    finds them: released as it rests, it falls as it rests. Where it comes to rest is not
    part of the contract.

    Args:
        depth: From the rim plane down to the floor (m).
        size: A rectangle's full extents ``(x, y)`` (m), or
        radius: a circle's radius (m), the largest circle inside the rim. Exactly one.
        reference: Reference frame name (default ``"opening"``).

    Example::

        placer    = OpeningPlacer(size=(0.238, 0.238), depth=0.757)
        templates = placer.place(shape, lowest=(0.02, 0.10), clearance=0.026)
        tsr  = templates[0].instantiate(T_world_opening)
        pose = tsr.sample()
    """

    def __init__(
        self,
        *,
        depth: float,
        size: Optional[Tuple[float, float]] = None,
        radius: Optional[float] = None,
        reference: str = "opening",
    ):
        self.depth = _check_positive_finite("depth", depth)
        if (size is None) == (radius is None):
            raise ValueError("give exactly one of size (a rectangle) or radius (a circle)")
        if size is not None:
            if len(size) != 2:
                raise ValueError(f"size must be (x, y), got {size!r}")
            self.size: Optional[Tuple[float, float]] = (
                _check_positive_finite("size[0]", size[0]),
                _check_positive_finite("size[1]", size[1]),
            )
            self.radius: Optional[float] = None
        else:
            self.size = None
            self.radius = _check_positive_finite("radius", radius)
        self.reference = reference

    def _halves(self, footprint: float, clearance: float) -> Tuple[float, float]:
        """The half-extents of the ``xy`` box the object origin may slide in, so that a
        footprint of this radius, swept through every yaw, stays ``clearance`` inside the
        opening: the rectangle inset by both, or the square inscribed in the circle of
        the radius left. Negative when it does not fit."""
        if self.size is not None:
            return (self.size[0] / 2 - clearance - footprint, self.size[1] / 2 - clearance - footprint)
        half = (self.radius - clearance - footprint) / math.sqrt(2.0)
        return (half, half)

    def place(
        self,
        shape: Shape,
        *,
        lowest: Tuple[float, float],
        clearance: float = 0.0,
        subject: str = "object",
        min_margin_deg: float = 0.0,
    ) -> List[TSRTemplate]:
        """Return one template per resting orientation that fits through the opening.

        Every template frames the object origin, as :meth:`StablePlacer.place`'s do: the
        same resting rotation in ``Tw_e``, free yaw, and the sliding bounds inset by the
        footprint about that origin, here within the opening less ``clearance``. The
        ``z`` bounds move the object's lowest point through ``lowest``.

        Args:
            shape: The object's parts and centre of mass, in the object frame.
            lowest: ``(low, high)``, the heights of the object's lowest point above the
                rim plane (m); negative is below the rim, and no lower than the floor.
            clearance: How far inside the opening the whole object stays (m).
            subject: Name of the object frame.
            min_margin_deg: Discard rests whose tipping angle is below this (degrees).

        Returns:
            The templates, most stable first for a compound, or ``[]`` when no resting
            orientation fits through the opening, with the reason logged at ``DEBUG``.
        """
        if not isinstance(shape, Shape):
            raise ValueError(f"shape must be a tsr.placement.Shape, got {type(shape).__name__}")
        if len(lowest) != 2:
            raise ValueError(f"lowest must be (low, high), got {lowest!r}")
        low, high = _check_finite("lowest[0]", lowest[0]), _check_finite("lowest[1]", lowest[1])
        if low > high:
            raise ValueError(f"lowest must have low <= high, got {lowest!r}")
        if low < -self.depth:
            raise ValueError(f"lowest[0] = {low!r} is below the floor, {self.depth!r} below the rim")
        clearance = _check_finite("clearance", clearance)
        if clearance < 0.0:
            raise ValueError(f"clearance must be non-negative, got {clearance!r}")

        rests = StablePlacer(_UNBOUNDED_M, _UNBOUNDED_M, reference=self.reference).place(
            shape, subject=subject, min_margin_deg=min_margin_deg
        )
        atol = length_atol(float(np.ptp(shape.surface(), axis=0).max()))
        templates = []
        for t in rests:
            hx, hy = self._halves(t.provenance.footprint_radius, clearance)
            if hx < -atol or hy < -atol:
                continue
            Bw = np.array(t.Bw, dtype=float)
            Bw[0] = (-max(hx, 0.0), max(hx, 0.0))
            Bw[1] = (-max(hy, 0.0), max(hy, 0.0))
            Bw[2] = (low, high)
            templates.append(
                TSRTemplate(
                    T_ref_tsr=t.T_ref_tsr,
                    Tw_e=t.Tw_e,
                    Bw=Bw,
                    task="place",
                    subject=subject,
                    reference=self.reference,
                    name=f"Release {t.variant} ({subject} into {self.reference})",
                    description=(
                        f"Object released as it rests on {t.variant}, inside {self.reference} by "
                        f"{clearance:.4g} m, its lowest point {low:.4g} to {high:.4g} m above the rim."
                    ),
                    variant=t.variant,
                    stability_margin=t.stability_margin,
                    provenance=t.provenance,
                )
            )
        if rests and not templates:
            smallest = min(t.provenance.footprint_radius for t in rests)
            logger.debug(
                "%s.place: empty feasible set (exceeds_opening; smallest footprint radius %.4g m "
                "with clearance %.4g m does not fit %s)",
                type(self).__name__,
                smallest,
                clearance,
                f"size {self.size}" if self.size is not None else f"radius {self.radius:.4g} m",
            )
        return templates
