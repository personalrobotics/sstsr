# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Tests for Robotiq2F85."""

import unittest

import numpy as np

from tsr.hands import ParallelJawGripper, Robotiq2F85


class TestRobotiq2F85(unittest.TestCase):
    def setUp(self):
        self.gripper = Robotiq2F85()

    def test_fixed_params(self):
        # Measured from mujoco_menagerie/robotiq_2f85/2f85.xml via
        # mj_manipulator/scripts/validate_gripper.py (AABB of each
        # collision geom along the approach axis). FINGER_LENGTH is
        # from the TSR "palm" (= forward edge of the base housing, 94 mm
        # past base_mount) to the pad tip (finger reach along approach);
        # MAX_APERTURE is pad-inner-face to pad-inner-face at full open.
        self.assertAlmostEqual(self.gripper.finger_length, 0.059)
        self.assertAlmostEqual(self.gripper.max_aperture, 0.085)
        self.assertAlmostEqual(self.gripper.PALM_OFFSET_FROM_BASE_MOUNT, 0.094)

    def test_is_subclass_of_parallel_jaw(self):
        self.assertIsInstance(self.gripper, ParallelJawGripper)

    def test_outputs_canonical_frames(self):
        # Robotiq2F85 should produce identical Tw_e as a plain
        # ParallelJawGripper with the same hardware parameters.
        base = ParallelJawGripper(
            finger_length=0.059, max_aperture=0.085, pad_length=0.0375, min_encompassing_diameter=0.043
        )
        # Use a can-sized cylinder (33 mm radius, 115 mm height)
        t_base = base.grasp_cylinder_side(0.033, 0.115)
        t_robotiq = self.gripper.grasp_cylinder_side(0.033, 0.115)

        self.assertEqual(len(t_base), len(t_robotiq))
        for tb, tr in zip(t_base, t_robotiq):
            np.testing.assert_allclose(tr.Tw_e, tb.Tw_e, atol=1e-10)

    def test_tw_e_is_valid_se3(self):
        templates = self.gripper.grasp_cylinder_side(0.033, 0.115)
        for t in templates:
            R = t.Tw_e[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_can_grasp_66mm_can(self):
        """A 66 mm diameter soda can fits within the 85 mm max aperture."""
        templates = self.gripper.grasp_cylinder_side(0.033, 0.115)
        self.assertGreater(len(templates), 0)

    def test_cannot_grasp_wide_object(self):
        """Objects wider than max_aperture are infeasible -> return []."""
        # Preshape 95 mm > max_aperture 85 mm
        self.assertEqual(self.gripper.grasp_cylinder_side(0.048, 0.115, preshape=0.095), [])
