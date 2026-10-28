# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Tests for Robotiq2F140."""

import unittest

import numpy as np

from tsr.hands import ParallelJawGripper, Robotiq2F140


class TestRobotiq2F140(unittest.TestCase):
    def setUp(self):
        self.gripper = Robotiq2F140()

    def test_fixed_params(self):
        # finger_length = palm(grasp_site, base_mount + 0.100) -> pad tip, from
        # geodude_assets 2f140.xml (same palm->pad-tip convention as 2F-85/Franka).
        self.assertAlmostEqual(self.gripper.finger_length, 0.114)
        # inner-face gap at full open (not the 0.140 nominal/outer spec); see #60
        self.assertAlmostEqual(self.gripper.max_aperture, 0.128)
        self.assertAlmostEqual(self.gripper.PALM_OFFSET_FROM_BASE_MOUNT, 0.100)

    def test_is_subclass_of_parallel_jaw(self):
        self.assertIsInstance(self.gripper, ParallelJawGripper)

    def test_outputs_canonical_frames(self):
        # Robotiq2F140 now outputs canonical TSR EE poses (no frame correction).
        # It should produce identical Tw_e as a plain ParallelJawGripper with
        # the same hardware parameters, its closed reach among them (#193).
        base = ParallelJawGripper(finger_length=0.114, max_aperture=0.128)
        base.CLOSED_REACH = Robotiq2F140.CLOSED_REACH
        t_base = base.grasp_cylinder_side(0.040, 0.10)
        t_robotiq = self.gripper.grasp_cylinder_side(0.040, 0.10)

        self.assertEqual(len(t_base), len(t_robotiq))
        for tb, tr in zip(t_base, t_robotiq):
            np.testing.assert_allclose(tr.Tw_e, tb.Tw_e, atol=1e-10)

    def test_tw_e_is_valid_se3(self):
        templates = self.gripper.grasp_cylinder_side(0.040, 0.10)
        for t in templates:
            R = t.Tw_e[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)
