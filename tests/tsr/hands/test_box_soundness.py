#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Per-orientation box grasp feasibility and soundness (issue #70).

Each box face supports two finger-opening orientations; a thin dimension must
remove only the dependent orientation, not the perpendicular one. Every emitted
pose is certified against the independent #67 analytic oracle, and structural
provenance (mode / approach / finger orientation) is read from the record -- never
by parsing display names.
"""

import unittest

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tsr.hands import ParallelJawGripper

from ._grasp_oracle import Box, certify

dims = st.floats(min_value=0.01, max_value=0.2, allow_nan=False, width=64)


def _bw_samples(t):
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


def _assert_all_sound(testcase, box, templates, gripper, clearance):
    for t in templates:
        prov = t.provenance
        testcase.assertEqual(prov.primitive, "box")
        tsr = t.instantiate(np.eye(4))
        for xyz in _bw_samples(t):
            w = certify(
                box,
                tsr.to_transform(xyz),
                finger_length=gripper.finger_length,
                max_aperture=gripper.max_aperture,
                preshape=float(t.preshape[0]),
                clearance=clearance,
                mode=prov.mode,
                approach=prov.approach,
                finger_orientation=prov.finger_orientation,
            )
            testcase.assertTrue(w.ok, (prov.mode, prov.approach, prov.finger_orientation, w.failed))


class TestThinBoxRetainsOrientation(unittest.TestCase):
    """A thin dimension removes only the dependent orientation (#70)."""

    def test_reproduction_retains_span_x(self):
        g = ParallelJawGripper(finger_length=0.055, max_aperture=0.14)
        templates = g.grasp_box_top(0.003, 0.060, 0.050)  # box_x thin: span-y (slides in x) infeasible
        self.assertTrue(templates)
        orientations = {t.provenance.finger_orientation for t in templates}
        self.assertEqual(orientations, {"x"})  # only the span-x orientation survives
        # The default margin is 10% of min(finger_length, box_z), the extent approached (#195).
        _assert_all_sound(self, Box(0.003, 0.060, 0.050), templates, g, 0.1 * min(0.055, 0.050))

    def test_thin_dimension_removes_only_dependent_orientation(self):
        # For a box top grasp, a thin box_x removes span-y (slides in x); a thin box_y
        # removes span-x (slides in y). The perpendicular orientation is retained.
        g = ParallelJawGripper(finger_length=0.055, max_aperture=0.14)
        thin_x = {t.provenance.finger_orientation for t in g.grasp_box_top(0.003, 0.06, 0.05)}
        thin_y = {t.provenance.finger_orientation for t in g.grasp_box_top(0.06, 0.003, 0.05)}
        self.assertEqual(thin_x, {"x"})
        self.assertEqual(thin_y, {"y"})

    def test_both_thin_returns_empty(self):
        g = ParallelJawGripper(finger_length=0.055, max_aperture=0.14)
        self.assertEqual(g.grasp_box_top(0.003, 0.003, 0.05), [])


class TestBoxSoundnessProperty(unittest.TestCase):
    """Every emitted box pose has a valid oracle witness (#70)."""

    @settings(max_examples=60, deadline=None)
    @given(fl=dims, ma=dims, bx=dims, by=dims, bz=dims, c=st.floats(0.0, 0.03, allow_nan=False), k=st.integers(1, 3))
    def test_all_box_faces_are_oracle_sound(self, fl, ma, bx, by, bz, c, k):
        g = ParallelJawGripper(finger_length=fl, max_aperture=ma)
        box = Box(bx, by, bz)
        _assert_all_sound(self, box, g.grasp_box(bx, by, bz, clearance=c, k=k), g, c)

    @settings(max_examples=40, deadline=None)
    @given(bx=dims, by=dims, bz=dims)
    def test_no_invalid_orientation_survives(self, bx, by, bz):
        # Every emitted template's declared finger orientation is one whose slide band
        # is a positive width -- i.e. the perpendicular in-face dimension exceeds 2c.
        g = ParallelJawGripper(finger_length=0.055, max_aperture=0.30)
        c = 0.1 * 0.055
        for t in g.grasp_box(bx, by, bz, clearance=c):
            slide_axis = t.provenance.metadata["slide_axis"]
            slide_dim = {"x": bx, "y": by, "z": bz}[slide_axis]
            # A zero-width band (dim == 2c) is feasible (#110); only a negative one is not.
            self.assertGreaterEqual(slide_dim / 2.0 - c, 0.0, (t.provenance.mode, slide_axis))


class TestBoxSymmetry(unittest.TestCase):
    """Axis permutation and opposite-face symmetries (#70)."""

    G = ParallelJawGripper(finger_length=0.055, max_aperture=0.30)

    def test_axis_swap_invariance(self):
        # Swapping box_x and box_y is a 90-degree rotation about z: the total number of
        # feasible templates is invariant, and the swapped set is oracle-sound.
        for bx, by, bz in [(0.05, 0.07, 0.09), (0.003, 0.06, 0.05), (0.12, 0.02, 0.08)]:
            c = 0.1 * 0.055
            a = self.G.grasp_box(bx, by, bz, clearance=c)
            b = self.G.grasp_box(by, bx, bz, clearance=c)
            self.assertEqual(len(a), len(b), (bx, by, bz))
            _assert_all_sound(self, Box(by, bx, bz), b, self.G, c)

    def test_opposite_faces_are_mirror_consistent(self):
        bx, by, bz, c = 0.05, 0.07, 0.09, 0.1 * 0.055
        self.assertEqual(
            len(self.G.grasp_box_top(bx, by, bz, clearance=c)), len(self.G.grasp_box_bottom(bx, by, bz, clearance=c))
        )
        # +x and -x faces of grasp_box_face_x are mirror images -> equal orientation sets.
        fx = self.G.grasp_box_face_x(bx, by, bz, clearance=c)
        pos = {t.provenance.finger_orientation for t in fx if t.provenance.approach == "+x"}
        neg = {t.provenance.finger_orientation for t in fx if t.provenance.approach == "-x"}
        self.assertEqual(pos, neg)


class TestBoxApertureClearanceBoundary(unittest.TestCase):
    """Aperture and clearance boundary behaviour on both sides (#70)."""

    def test_aperture_boundary_per_orientation(self):
        # A preshape between box_x and box_y keeps only the orientation whose span fits.
        g = ParallelJawGripper(finger_length=0.055, max_aperture=0.30)
        bx, by, bz, c = 0.04, 0.08, 0.05, 0.005
        mid = 0.06  # bx < mid < by
        templates = g.grasp_box_top(bx, by, bz, preshape=mid, clearance=c)
        self.assertEqual({t.provenance.finger_orientation for t in templates}, {"x"})  # only span-x = 0.04 fits
        _assert_all_sound(self, Box(bx, by, bz), templates, g, c)

    def test_slide_band_boundary(self):
        # box_x / 2 == clearance is the boundary for the span-y (slides-in-x) orientation.
        g = ParallelJawGripper(finger_length=0.055, max_aperture=0.30)
        by, bz = 0.08, 0.05
        c = 0.01
        # box_x/2 just below c -> span-y removed; just above -> span-y present.
        below = g.grasp_box_top(2 * c - 1e-4, by, bz, clearance=c)
        above = g.grasp_box_top(2 * c + 1e-4, by, bz, clearance=c)
        self.assertNotIn("y", {t.provenance.finger_orientation for t in below})
        self.assertIn("y", {t.provenance.finger_orientation for t in above})


class TestZeroWidthSlideBand(unittest.TestCase):
    """An in-face dimension exactly 2*clearance yields one fixed centered pose (#110)."""

    G = ParallelJawGripper(finger_length=0.055, max_aperture=0.30)
    C = 0.01

    @staticmethod
    def _orientations(templates):
        return {t.provenance.finger_orientation for t in templates}

    def test_zero_width_boundary_matrix(self):
        c = self.C
        two_c = 2 * c  # exactly representable: box_dim/2 == c
        big = 0.08  # a comfortably feasible perpendicular dimension
        # (label, call(slide_dim) -> templates, dependent orientation, Box(slide_dim))
        cases = [
            ("top", lambda d: self.G.grasp_box_top(d, big, 0.05, clearance=c), "y", lambda d: Box(d, big, 0.05)),
            ("bottom", lambda d: self.G.grasp_box_bottom(d, big, 0.05, clearance=c), "y", lambda d: Box(d, big, 0.05)),
            ("face_x", lambda d: self.G.grasp_box_face_x(big, d, 0.05, clearance=c), "z", lambda d: Box(big, d, 0.05)),
            ("face_y", lambda d: self.G.grasp_box_face_y(d, big, 0.05, clearance=c), "z", lambda d: Box(d, big, 0.05)),
        ]
        for label, call, dep, box_of in cases:
            below = self._orientations(call(np.nextafter(two_c, -np.inf)))
            at_templates = call(two_c)
            above = self._orientations(call(np.nextafter(two_c, np.inf)))
            self.assertNotIn(dep, below, (label, "below"))  # negative band -> removed
            self.assertIn(dep, self._orientations(at_templates), (label, "at"))  # zero-width -> kept
            self.assertIn(dep, above, (label, "above"))  # positive band -> kept
            # The zero-width-band templates have a fixed slide coordinate and are sound.
            kept = [t for t in at_templates if t.provenance.finger_orientation == dep]
            for t in kept:
                slide_row = "xyz".index(t.provenance.metadata["slide_axis"])
                np.testing.assert_array_equal(t.Bw[slide_row], [0.0, 0.0])
            _assert_all_sound(self, box_of(two_c), at_templates, self.G, c)


if __name__ == "__main__":
    unittest.main()
