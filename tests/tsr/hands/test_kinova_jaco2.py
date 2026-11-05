# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""KinovaJaco2TwoFinger grasps ADA's cylinder, and nothing wider than its aperture (#185).

That the fingers close on what these templates pose is checked in the hand's MuJoCo model by
``tools/grasp_closure.py``, outside this suite, which carries no simulator. That the closed
fingertips stay inside the far face is the property in ``test_closed_reach.py``.
"""

import unittest

from tsr.hands import KinovaJaco2TwoFinger, default_registry


class TestKinovaJaco2TwoFinger(unittest.TestCase):
    def test_grasps_adas_cylinder_from_side_top_and_bottom(self):
        hand = KinovaJaco2TwoFinger()
        modes = {t.provenance.mode for t in hand.grasp_cylinder(0.025, 0.10)}
        self.assertEqual(modes, {"side", "top", "bottom"})

    def test_no_grasp_of_a_cylinder_wider_than_the_aperture(self):
        hand = KinovaJaco2TwoFinger()
        radius = hand.max_aperture / 2 + 0.001
        self.assertEqual(hand.grasp_cylinder(radius, 0.10), [])

    def test_registered_for_cylinders(self):
        generate = default_registry.get("kinova_jaco2_two_finger", "cylinder", "grasp")
        self.assertTrue(generate(KinovaJaco2TwoFinger(), cylinder_radius=0.025, cylinder_height=0.10))


if __name__ == "__main__":
    unittest.main()
