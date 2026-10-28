# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""The fingertips stay inside the far face once the hand closes (issue #193).

A linkage hand's fingertips reach further as it closes, ``reach(width)`` past the palm
rather than ``finger_length``. A template that keeps the open fingertips a clearance inside
the far face of what they span could otherwise drive them through it on closing -- into the
table, for a box top grasp.

The invariant is geometric and checked from the poses alone: for every pose of every
template, the palm is a clearance outside the near face, and the fingertips are a clearance
inside the far face both open (``finger_length`` past the palm) and closed on the span
(``reach(span)`` past the palm). Near face, far face and span are the solid's extent along
the approach and the closing axis, projected from its own geometry; nothing is read from the
template's provenance.
"""

import itertools
import logging
import unittest

import numpy as np

from tsr.hands import FrankaHand, ParallelJawGripper, Robotiq2F85, Robotiq2F140

HANDS = (Robotiq2F140(), Robotiq2F85(), FrankaHand())
CLEARANCES = (0.002, 0.005)
BOX_SIDES = (0.015, 0.03, 0.06, 0.1, 0.16)
CYLINDER_RADII = (0.008, 0.015, 0.025, 0.04, 0.06)
CYLINDER_HEIGHTS = (0.02, 0.05, 0.12)


def _box_extent(dims, u):
    """``(min, max)`` of a box (``tsr.hands.base`` convention) projected onto unit ``u``."""
    bx, by, bz = dims
    corners = np.array(list(itertools.product((-bx / 2, bx / 2), (-by / 2, by / 2), (0.0, bz))))
    s = corners @ u
    return s.min(), s.max()


def _cylinder_extent(dims, u):
    """``(min, max)`` of a cylinder, axis +z over ``[0, h]``, projected onto unit ``u``."""
    r, h = dims
    ends = (0.0, u[2] * h)
    rim = r * np.sqrt(max(0.0, 1.0 - u[2] ** 2))
    return min(ends) - rim, max(ends) + rim


def _sphere_extent(dims, u):
    (r,) = dims
    return -r, r


def _bw_samples(t):
    """The Bw midpoint and every free coordinate's extremes."""
    lo, hi = t.Bw[:, 0], t.Bw[:, 1]
    mid = (lo + hi) / 2.0
    samples = [mid]
    for i in range(6):
        if hi[i] > lo[i]:
            for v in (lo[i], hi[i]):
                s = mid.copy()
                s[i] = v
                samples.append(s)
    return samples


def _violations(hand, extent, dims, templates, clearance):
    """Every pose at which the palm or the open or closed fingertips break the clearance."""
    scale = max(dims) if len(dims) > 1 else dims[0]
    tol = 1e-9 + 1e-6 * scale  # the contract's length tolerance
    out = []
    for t in templates:
        tsr = t.instantiate(np.eye(4))
        for xi in _bw_samples(t):
            pose = tsr.to_transform(xi)
            closing, approach, palm = pose[:3, 1], pose[:3, 2], pose[:3, 3]
            near, far = extent(dims, approach)
            lo, hi = extent(dims, closing)
            p = palm @ approach
            gaps = {
                "palm outside the near face": near - p,
                "open fingertips inside the far face": far - (p + hand.finger_length),
                "closed fingertips inside the far face": far - (p + hand.reach(hi - lo)),
            }
            for what, gap in gaps.items():
                if gap < clearance - tol:
                    out.append(f"{t.name}: {what} by {gap * 1e3:.3f} mm < {clearance * 1e3:g} mm")
    return out


class TestClosedFingertipsStayInside(unittest.TestCase):
    """Over a grid of boxes and cylinders, for each hand, every template keeps the clearance."""

    def _check(self, hand, family, extent, dims, clearance, counts):
        templates = getattr(hand, family)(*dims, clearance=clearance)
        counts[family] = counts.get(family, 0) + len(templates)
        self.assertEqual(_violations(hand, extent, dims, templates, clearance), [], (type(hand).__name__, dims))

    def test_boxes(self):
        families = ("grasp_box_top", "grasp_box_bottom", "grasp_box_face_x", "grasp_box_face_y")
        for hand in HANDS:
            counts = {}
            for dims, c, family in itertools.product(itertools.product(BOX_SIDES, repeat=3), CLEARANCES, families):
                self._check(hand, family, _box_extent, dims, c, counts)
            # Not vacuous: the grid reaches every family on every hand.
            self.assertTrue(all(counts.get(f) for f in families), (type(hand).__name__, counts))

    def test_cylinders_and_spheres(self):
        families = ("grasp_cylinder_side", "grasp_cylinder_top", "grasp_cylinder_bottom")
        for hand in HANDS:
            counts = {}
            for r, h, c in itertools.product(CYLINDER_RADII, CYLINDER_HEIGHTS, CLEARANCES):
                for family in families:
                    self._check(hand, family, _cylinder_extent, (r, h), c, counts)
                self._check(hand, "grasp_sphere", _sphere_extent, (r,), c, counts)
            self.assertTrue(all(counts.get(f) for f in families + ("grasp_sphere",)), (type(hand).__name__, counts))


class TestClosedReachDiagnostic(unittest.TestCase):
    """A family emptied only by the closed reach says so (#193)."""

    def test_reason_names_the_closed_reach(self):
        hand = Robotiq2F140()
        # The same jaw without the linkage: the open hand fits each of these.
        prismatic = ParallelJawGripper(finger_length=hand.finger_length, max_aperture=hand.max_aperture)
        cases = (
            ("grasp_box_top", (0.072, 0.088, 0.028)),  # a gelatin box: the fingertips reach the table
            ("grasp_cylinder_side", (0.015, 0.1)),
            ("grasp_cylinder_top", (0.02, 0.025)),
        )
        for family, dims in cases:
            self.assertTrue(getattr(prismatic, family)(*dims), family)
            with self.assertLogs("tsr.hands.base", level=logging.DEBUG) as logs:
                self.assertEqual(getattr(hand, family)(*dims), [], family)
            self.assertIn("closed_reach_too_long", "\n".join(logs.output), family)


if __name__ == "__main__":
    unittest.main()
