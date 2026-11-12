# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""KinovaJaco2TwoFinger grasps ADA's cylinder, and nothing wider than its aperture (#185).

That the fingers close on what these templates pose is checked in the hand's MuJoCo model by
``tools/grasp_closure.py``, outside this suite, which carries no simulator. That the closed
fingertips stay inside the far face is the property in ``test_closed_reach.py``; for its
enveloping grasps (#204), that they reach past the cylinder's middle instead. Which cylinders
it envelops is measured in the model by ``tools/finger_reach.py --envelop``, and whether the
enveloping and fingertip grasps keep a free cylinder seated by ``tools/grasp_hold.py``.
"""

import unittest

import numpy as np

from tsr.hands import FrankaHand, KinovaJaco2TwoFinger, Robotiq2F85, Robotiq2F140, default_registry
from tsr.hands._conformance import validate_builtin_provenance


class TestKinovaJaco2TwoFinger(unittest.TestCase):
    def test_grasps_adas_cylinder_from_side_top_and_bottom(self):
        hand = KinovaJaco2TwoFinger()
        modes = {t.provenance.mode for t in hand.grasp_cylinder(0.025, 0.10)}
        self.assertEqual(modes, {"side", "top", "bottom"})

    def test_no_grasp_of_a_cylinder_wider_than_the_aperture(self):
        hand = KinovaJaco2TwoFinger()
        radius = hand.max_aperture / 2 + 0.001
        self.assertEqual(hand.grasp_cylinder(radius, 0.10), [])

    def test_envelops_the_cylinders_it_declares_with_the_palm_a_clearance_away(self):
        # #204: within ENVELOPING_DIAMETERS, side grasps include an enveloping family whose palm
        # is the side grasp's clearance from the near surface, measured from each pose.
        # The range is measured with the palm at the side grasp's default clearance, a tenth of
        # the radius here, so that is the clearance enveloping grasps are emitted with.
        hand = KinovaJaco2TwoFinger()
        narrowest, _ = hand.ENVELOPING_DIAMETERS
        for diameter in (narrowest - 0.002, narrowest, 0.040, 0.066):
            r = diameter / 2
            clearance = 0.1 * r
            templates = hand.grasp_cylinder_side(r, 0.12)
            enveloping = [t for t in templates if t.provenance.grip == "enveloping"]
            if diameter < narrowest:
                self.assertEqual(enveloping, [], diameter)
                continue
            self.assertEqual(len(enveloping), 2, diameter)  # both rolls
            wider = hand.grasp_cylinder_side(r, 0.12, clearance=2 * clearance)
            self.assertEqual([t for t in wider if t.provenance.grip == "enveloping"], [], diameter)
            for t in enveloping:
                validate_builtin_provenance(t.provenance)
                pose = t.instantiate(np.eye(4)).to_transform(t.Bw.mean(axis=1))
                palm, approach = pose[:3, 3], pose[:3, 2]
                axis_distance = float(np.hypot(palm[0], palm[1]))
                self.assertAlmostEqual(axis_distance - r, clearance, places=9)
                self.assertAlmostEqual(float(approach[2]), 0.0, places=9)  # radial, across the axis

    def test_no_enveloping_grasp_where_the_side_preshape_exceeds_the_aperture(self):
        # The declared range reaches 72 mm, but a side grasp opens to the diameter and a
        # clearance; from 69.5 mm that is wider than the hand opens, so nothing is emitted.
        hand = KinovaJaco2TwoFinger()
        self.assertTrue([t for t in hand.grasp_cylinder_side(0.069 / 2, 0.12) if t.provenance.grip == "enveloping"])
        for diameter in (0.070, hand.ENVELOPING_DIAMETERS[1]):
            self.assertEqual(hand.grasp_cylinder_side(diameter / 2, 0.12), [], diameter)

    def test_other_hands_do_not_envelop(self):
        for hand in (Robotiq2F140(), Robotiq2F85(), FrankaHand()):
            grips = {t.provenance.grip for t in hand.grasp_cylinder_side(0.02, 0.12)}
            self.assertNotIn("enveloping", grips, type(hand).__name__)

    def test_registered_for_cylinders(self):
        generate = default_registry.get("kinova_jaco2_two_finger", "cylinder", "grasp")
        self.assertTrue(generate(KinovaJaco2TwoFinger(), cylinder_radius=0.025, cylinder_height=0.10))


if __name__ == "__main__":
    unittest.main()
