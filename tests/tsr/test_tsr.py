# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

from unittest import TestCase

import numpy
from numpy import pi

from tsr import FRAME_ATOL
from tsr.tsr import TSR


class TsrTest(TestCase):
    def test_sample_xyzrpy(self):
        # Test zero-intervals.
        Bw = [
            [0.0, 0.0],  # X
            [1.0, 1.0],  # Y
            [-1.0, -1.0],  # Z
            [0.0, 0.0],  # roll
            [pi, pi],  # pitch
            [-pi, -pi],
        ]  # yaw
        tsr = TSR(Bw=Bw)
        s = tsr.sample_xyzrpy()

        Bw = numpy.array(Bw)
        # For zero-intervals, the sampled value should be exactly equal to the bound
        # Note: angles get wrapped, so pi becomes -pi
        expected = Bw[:, 0].copy()
        expected[4] = -pi  # pitch gets wrapped from pi to -pi
        self.assertTrue(numpy.allclose(s, expected, atol=1e-10))

        # Test simple non-zero intervals
        Bw = [
            [-0.1, 0.1],  # X
            [-0.1, 0.1],  # Y
            [-0.1, 0.1],  # Z
            [-pi / 4, pi / 4],  # roll
            [-pi / 4, pi / 4],  # pitch
            [-pi / 4, pi / 4],
        ]  # yaw
        tsr = TSR(Bw=Bw)
        s = tsr.sample_xyzrpy()

        Bw = numpy.array(Bw)
        self.assertTrue(numpy.all(s >= Bw[:, 0]))
        self.assertTrue(numpy.all(s <= Bw[:, 1]))

    def test_tsr_creation(self):
        """Test basic TSR creation."""
        T0_w = numpy.eye(4)
        Tw_e = numpy.eye(4)
        Bw = numpy.zeros((6, 2))
        Bw[2, :] = [0.0, 0.02]  # Allow vertical movement
        Bw[5, :] = [-pi, pi]  # Allow any yaw rotation

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        self.assertIsInstance(tsr.T0_w, numpy.ndarray)
        self.assertIsInstance(tsr.Tw_e, numpy.ndarray)
        self.assertIsInstance(tsr.Bw, numpy.ndarray)
        self.assertEqual(tsr.T0_w.shape, (4, 4))
        self.assertEqual(tsr.Tw_e.shape, (4, 4))
        self.assertEqual(tsr.Bw.shape, (6, 2))

    def test_tsr_sampling(self):
        """Test TSR sampling functionality."""
        T0_w = numpy.eye(4)
        Tw_e = numpy.eye(4)
        Bw = numpy.zeros((6, 2))
        Bw[2, :] = [0.0, 0.02]  # Allow vertical movement
        Bw[5, :] = [-pi, pi]  # Allow any yaw rotation

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Test sampling
        pose = tsr.sample()
        self.assertIsInstance(pose, numpy.ndarray)
        self.assertEqual(pose.shape, (4, 4))

        # Test xyzrpy sampling
        xyzrpy = tsr.sample_xyzrpy()
        self.assertIsInstance(xyzrpy, numpy.ndarray)
        self.assertEqual(xyzrpy.shape, (6,))

    def test_tsr_validation(self):
        """Test TSR validation."""
        T0_w = numpy.eye(4)
        Tw_e = numpy.eye(4)
        Bw = numpy.zeros((6, 2))
        Bw[2, :] = [0.0, 0.02]  # Allow vertical movement
        Bw[5, :] = [-pi, pi]  # Allow any yaw rotation

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Test valid xyzrpy
        valid_xyzrpy = numpy.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0])
        self.assertTrue(all(tsr.is_valid(valid_xyzrpy)))

        # Test invalid xyzrpy (outside bounds)
        invalid_xyzrpy = numpy.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])  # z too large
        self.assertFalse(all(tsr.is_valid(invalid_xyzrpy)))

    def test_tsr_contains(self):
        """Test TSR containment checking."""
        T0_w = numpy.eye(4)
        Tw_e = numpy.eye(4)
        Bw = numpy.zeros((6, 2))
        Bw[2, :] = [0.0, 0.02]  # Allow vertical movement
        Bw[5, :] = [-pi, pi]  # Allow any yaw rotation

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Test contained transform
        contained_transform = numpy.eye(4)
        contained_transform[2, 3] = 0.01  # Within z bounds
        self.assertTrue(tsr.contains(contained_transform))

        # Test non-contained transform
        non_contained_transform = numpy.eye(4)
        non_contained_transform[2, 3] = 0.1  # Outside z bounds
        self.assertFalse(tsr.contains(non_contained_transform))

    def test_tsr_distance(self):
        """Test TSR distance calculation."""
        T0_w = numpy.eye(4)
        Tw_e = numpy.eye(4)
        Bw = numpy.zeros((6, 2))
        Bw[2, :] = [0.0, 0.02]  # Allow vertical movement
        Bw[5, :] = [-pi, pi]  # Allow any yaw rotation

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Test distance to contained transform
        contained_transform = numpy.eye(4)
        contained_transform[2, 3] = 0.01
        distance, bwopt = tsr.distance(contained_transform)
        self.assertEqual(distance, 0.0)

        # Test distance to non-contained transform
        non_contained_transform = numpy.eye(4)
        non_contained_transform[2, 3] = 0.1
        distance, bwopt = tsr.distance(non_contained_transform)
        self.assertGreater(distance, 0.0)

    def test_contains_with_non_identity_frames(self):
        """Test TSR containment with non-identity T0_w and Tw_e.

        This test ensures contains() correctly transforms to the TSR frame
        before checking bounds. With identity frames, bugs in frame handling
        are invisible.
        """
        # Non-identity T0_w: TSR origin offset from world origin
        T0_w = numpy.eye(4)
        T0_w[0, 3] = 1.0  # TSR origin at x=1
        T0_w[1, 3] = 2.0  # TSR origin at y=2

        # Non-identity Tw_e: end-effector offset from TSR frame
        Tw_e = numpy.eye(4)
        Tw_e[2, 3] = 0.5  # End-effector 0.5m above TSR frame

        Bw = numpy.array(
            [
                [-0.1, 0.1],  # X bounds
                [-0.1, 0.1],  # Y bounds
                [-0.1, 0.1],  # Z bounds
                [-pi / 4, pi / 4],  # roll bounds
                [-pi / 4, pi / 4],  # pitch bounds
                [-pi / 4, pi / 4],  # yaw bounds
            ]
        )

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Generate a contained transform using to_transform (round-trip test)
        valid_bw = numpy.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
        contained_transform = tsr.to_transform(valid_bw)

        self.assertTrue(tsr.contains(contained_transform))

        # A transform at the world origin should NOT be contained
        # (it's far from the TSR which is centered at x=1, y=2)
        world_origin = numpy.eye(4)
        self.assertFalse(tsr.contains(world_origin))

    def test_distance_with_non_identity_frames(self):
        """Test TSR distance with non-identity T0_w and Tw_e.

        Verifies that distance() correctly handles frame transformations
        and returns 0 for contained transforms.
        """
        # Rotated T0_w: TSR frame rotated 90 degrees about Z
        T0_w = numpy.array([[0, -1, 0, 0.5], [1, 0, 0, 0.5], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

        # Tw_e with rotation and translation
        Tw_e = numpy.array([[0, 0, 1, 0.1], [1, 0, 0, 0], [0, 1, 0, 0.05], [0, 0, 0, 1]], dtype=float)

        Bw = numpy.array(
            [
                [-0.05, 0.05],
                [-0.05, 0.05],
                [-0.05, 0.05],
                [-pi / 6, pi / 6],
                [-pi / 6, pi / 6],
                [-pi / 6, pi / 6],
            ]
        )

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Generate contained transform via to_transform
        valid_bw = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        contained_transform = tsr.to_transform(valid_bw)

        distance, bwopt = tsr.distance(contained_transform)
        self.assertEqual(distance, 0.0)
        self.assertTrue(tsr.contains(contained_transform))

        # Test at bounds edge
        edge_bw = numpy.array([0.05, 0.05, 0.05, 0.0, 0.0, 0.0])
        edge_transform = tsr.to_transform(edge_bw)

        distance, bwopt = tsr.distance(edge_transform)
        self.assertEqual(distance, 0.0)
        self.assertTrue(tsr.contains(edge_transform))

    def test_closest_transform(self):
        """closest_transform returns the closest in-bounds *world-frame* pose (#53)."""
        T0_w = numpy.array([[0, -1, 0, 0.5], [1, 0, 0, 0.5], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        Tw_e = numpy.array([[0, 0, 1, 0.1], [1, 0, 0, 0], [0, 1, 0, 0.05], [0, 0, 0, 1]], dtype=float)
        Bw = numpy.array([[-0.05, 0.05]] * 3 + [[-pi / 6, pi / 6]] * 3)
        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # A contained pose is its own closest transform, at distance 0.
        inside = tsr.to_transform(numpy.zeros(6))
        dist, T = tsr.closest_transform(inside)
        self.assertEqual(dist, 0.0)
        numpy.testing.assert_allclose(T, inside, atol=1e-9)

        # An outside pose projects to an in-bounds world pose, matching the
        # hand-composed T0_w @ xyzrpy_to_trans(bwopt) @ Tw_e.
        outside = inside.copy()
        outside[0, 3] += 1.0  # 1 m away in world x
        dist, T = tsr.closest_transform(outside)
        self.assertGreater(dist, 0.0)
        self.assertTrue(tsr.contains(T))
        _, bwopt = tsr.distance(outside)
        expected = T0_w @ TSR.xyzrpy_to_trans(bwopt) @ Tw_e
        numpy.testing.assert_allclose(T, expected, atol=1e-12)

    def test_contains_distance_consistency(self):
        """Test that contains() and distance() are consistent.

        For any transform:
        - contains(t) == True  implies distance(t) == 0
        - contains(t) == False implies distance(t) > 0
        """
        # Use non-trivial frames to catch frame-handling bugs
        T0_w = numpy.array([[1, 0, 0, 0.3], [0, 1, 0, -0.2], [0, 0, 1, 0.1], [0, 0, 0, 1]], dtype=float)

        Tw_e = numpy.array([[0, -1, 0, 0], [1, 0, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

        Bw = numpy.array(
            [
                [-0.1, 0.1],
                [-0.1, 0.1],
                [-0.1, 0.1],
                [-pi / 4, pi / 4],
                [-pi / 4, pi / 4],
                [-pi / 4, pi / 4],
            ]
        )

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Test multiple random samples - all should be contained with distance 0
        for _ in range(10):
            sample_bw = tsr.sample_xyzrpy()
            sample_transform = tsr.to_transform(sample_bw)

            is_contained = tsr.contains(sample_transform)
            distance, _ = tsr.distance(sample_transform)

            self.assertTrue(is_contained, f"Sampled transform should be contained, bw={sample_bw}")
            self.assertEqual(
                distance,
                0.0,
                f"Distance should be 0 for contained transform, bw={sample_bw}",
            )

        # Test transforms outside the TSR
        outside_transforms = [
            numpy.eye(4),  # World origin - likely outside
            numpy.diag([1, 1, 1, 1]).astype(float),  # Another identity
        ]
        # Add a clearly outside transform
        far_away = numpy.eye(4)
        far_away[0, 3] = 100.0  # Very far in x
        outside_transforms.append(far_away)

        for t in outside_transforms:
            is_contained = tsr.contains(t)
            distance, _ = tsr.distance(t)

            # If not contained, distance must be > 0
            if not is_contained:
                self.assertGreater(
                    distance,
                    0.0,
                    "Non-contained transform should have positive distance",
                )

    def test_roundtrip_to_transform_to_xyzrpy(self):
        """Test that to_transform and to_xyzrpy are inverses.

        This validates the frame transformations are consistent.
        """
        T0_w = numpy.array([[0, 0, 1, 1.0], [0, 1, 0, 0], [-1, 0, 0, 0.5], [0, 0, 0, 1]], dtype=float)

        Tw_e = numpy.eye(4)
        Tw_e[0, 3] = 0.2

        Bw = numpy.array(
            [
                [-0.1, 0.1],
                [-0.1, 0.1],
                [-0.1, 0.1],
                [-pi / 4, pi / 4],
                [-pi / 4, pi / 4],
                [-pi / 4, pi / 4],
            ]
        )

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Test round-trip: xyzrpy -> transform -> xyzrpy
        original_bw = numpy.array([0.05, -0.03, 0.02, 0.1, -0.1, 0.2])
        transform = tsr.to_transform(original_bw)
        recovered_bw = tsr.to_xyzrpy(transform)

        numpy.testing.assert_array_almost_equal(
            original_bw,
            recovered_bw,
            decimal=10,
            err_msg="Round-trip xyzrpy -> transform -> xyzrpy failed",
        )

    def test_outer_interval_bounds(self):
        """Test TSR with outer interval RPY bounds (wrapping around ±pi).

        An outer interval like [3*pi/4, -3*pi/4] for yaw means yaw values
        in the 'back hemisphere' (|yaw| > 3*pi/4). This tests that such
        intervals are handled correctly.
        """
        # Outer interval for yaw: values near ±pi (back hemisphere)
        Bw = numpy.array(
            [
                [-0.1, 0.1],
                [-0.1, 0.1],
                [-0.1, 0.1],
                [-pi / 4, pi / 4],
                [-pi / 4, pi / 4],
                [3 * pi / 4, -3 * pi / 4],  # Outer interval: |yaw| > 3*pi/4
            ]
        )

        tsr = TSR(Bw=Bw)

        # Verify _Bw_cont has correct interval size (pi/2, not negative)
        yaw_interval = tsr._Bw_cont[5, 1] - tsr._Bw_cont[5, 0]
        self.assertGreater(
            yaw_interval,
            0,
            "Outer interval should produce positive continuous interval",
        )
        self.assertAlmostEqual(
            yaw_interval,
            pi / 2,
            places=10,
            msg="Outer interval [3*pi/4, -3*pi/4] should have size pi/2",
        )

        # Test sampling produces values in the outer interval
        for _ in range(10):
            sample = tsr.sample_xyzrpy()
            yaw = sample[5]
            self.assertTrue(
                abs(yaw) > 3 * pi / 4 - 0.01,
                f"Sampled yaw {yaw} should be in outer interval (|yaw| > 3*pi/4)",
            )

        # Test contains: transform with yaw near pi should be contained
        valid_bw = numpy.array([0, 0, 0, 0, 0, 0.9 * pi])
        trans_in = tsr.to_transform(valid_bw)
        self.assertTrue(
            tsr.contains(trans_in),
            "Transform with yaw=0.9*pi should be in outer interval",
        )

        # Test contains: transform with yaw=0 should NOT be contained
        # (yaw=0 is not in the outer interval [3*pi/4, -3*pi/4])
        trans_yaw0 = numpy.eye(4)
        self.assertFalse(
            tsr.contains(trans_yaw0),
            "Transform with yaw=0 should NOT be in outer interval",
        )

        # Test distance consistency for outer interval
        distance, _ = tsr.distance(trans_in)
        self.assertEqual(
            distance,
            0.0,
            "Distance should be 0 for contained transform in outer interval",
        )

    def test_rpy_roundtrip_near_singularity(self):
        """Test rot_to_rpy / rpy_to_rot round-trip near pitch = ±pi/2.

        The RPY decomposition has a gimbal lock singularity at pitch = ±pi/2.
        The code has special-case logic for this; verify it produces a
        consistent rotation matrix round-trip.
        """
        test_pitches = [
            pi / 2 - 1e-6,  # just below singularity
            pi / 2,  # exact singularity
            pi / 2 + 1e-6,  # just above singularity
            -pi / 2 - 1e-6,
            -pi / 2,
            -pi / 2 + 1e-6,
        ]
        for pitch in test_pitches:
            R = TSR.rpy_to_rot([0.3, pitch, 0.5])
            rpy = TSR.rot_to_rpy(R)
            R2 = TSR.rpy_to_rot(rpy)
            numpy.testing.assert_allclose(R, R2, atol=1e-6, err_msg=f"RPY round-trip failed at pitch={pitch}")

    def test_rpy_roundtrip_general(self):
        """Test rot_to_rpy / rpy_to_rot round-trip for general rotations."""
        test_rpys = [
            [0, 0, 0],
            [pi / 4, pi / 6, pi / 3],
            [-pi / 3, pi / 4, -pi / 6],
            [pi, 0, pi],
            [0.1, -0.2, 0.3],
        ]
        for rpy in test_rpys:
            R = TSR.rpy_to_rot(rpy)
            rpy2 = TSR.rot_to_rpy(R)
            R2 = TSR.rpy_to_rot(rpy2)
            numpy.testing.assert_allclose(R, R2, atol=1e-10, err_msg=f"RPY round-trip failed for rpy={rpy}")


class ConstructionContractTest(TestCase):
    """A malformed region is rejected at construction (#162).

    This is a contract shared with another implementation: pycbirrt's native TSR
    runtime is checked differentially against this one and applies the same rules at
    the same ``FRAME_ATOL``, so what one accepts the other must accept. Everything here
    is stated in terms a second implementation can reproduce -- which argument, which
    condition, which tolerance -- rather than in terms of this code's internals.

    Without it a non-finite frame, a rotation block that is not a rotation, or a NaN in
    ``Bw`` is accepted and surfaces much later as a NaN distance, a sample nothing can
    use, or a projection that never converges.
    """

    VALID = dict(T0_w=numpy.eye(4), Tw_e=numpy.eye(4), Bw=numpy.zeros((6, 2)))

    def _rejects(self, argument, **override):
        kwargs = dict(self.VALID)
        kwargs.update(override)
        with self.assertRaises(ValueError) as caught:
            TSR(**kwargs)
        self.assertIn(argument, str(caught.exception))

    # -- frames --------------------------------------------------------------

    def test_a_frame_must_be_finite(self):
        for argument in ("T0_w", "Tw_e"):
            for bad in (numpy.nan, numpy.inf, -numpy.inf):
                frame = numpy.eye(4)
                frame[0, 3] = bad
                self._rejects(argument, **{argument: frame})

    def test_a_frame_must_be_4x4(self):
        for argument in ("T0_w", "Tw_e"):
            self._rejects(argument, **{argument: numpy.eye(3)})
            self._rejects(argument, **{argument: numpy.zeros((4, 3))})

    def test_a_frame_must_be_homogeneous(self):
        frame = numpy.eye(4)
        frame[3] = [0.0, 0.0, 1.0, 1.0]
        self._rejects("T0_w", T0_w=frame)

    def test_a_frame_rotation_block_must_be_a_rotation(self):
        scaled = numpy.eye(4)
        scaled[:3, :3] *= 1.5  # orthogonal but not orthonormal
        self._rejects("Tw_e", Tw_e=scaled)

        sheared = numpy.eye(4)
        sheared[0, 1] = 0.3
        self._rejects("Tw_e", Tw_e=sheared)

    def test_a_reflection_is_not_a_rotation(self):
        reflected = numpy.eye(4)
        reflected[0, 0] = -1.0  # orthonormal, but determinant -1
        with self.assertRaises(ValueError) as caught:
            TSR(Tw_e=reflected)
        self.assertIn("determinant", str(caught.exception))

    def test_the_tolerance_is_the_documented_one(self):
        """The number both implementations must agree on, probed from either side.

        A rotation block scaled by ``1 + d`` has ``max|R Rᵀ - I| ≈ 2d``, so the
        boundary sits at ``d = FRAME_ATOL/2``.
        """
        inside, outside = numpy.eye(4), numpy.eye(4)
        inside[:3, :3] *= 1.0 + FRAME_ATOL / 4.0
        outside[:3, :3] *= 1.0 + FRAME_ATOL * 4.0
        TSR(Tw_e=inside)  # must not raise
        with self.assertRaises(ValueError):
            TSR(Tw_e=outside)

    # -- bounds --------------------------------------------------------------

    def test_bounds_must_be_a_finite_6x2(self):
        self._rejects("Bw", Bw=numpy.zeros((6, 3)))
        self._rejects("Bw", Bw=numpy.zeros((5, 2)))
        for bad in (numpy.nan, numpy.inf, -numpy.inf):
            Bw = numpy.zeros((6, 2))
            Bw[2, 1] = bad
            self._rejects("Bw", Bw=Bw)

    def test_translation_bounds_must_be_ordered(self):
        Bw = numpy.zeros((6, 2))
        Bw[1] = [1.0, -1.0]
        self._rejects("Bw", Bw=Bw)

    def test_an_outer_rotational_interval_is_valid(self):
        """``hi < lo`` on a rotational row wraps through ±π and stays expressible."""
        Bw = numpy.zeros((6, 2))
        Bw[5] = [3 * pi / 4, -3 * pi / 4]
        tsr = TSR(Bw=Bw)
        for _ in range(64):
            yaw = tsr.sample_xyzrpy()[5]
            self.assertTrue(abs(yaw) >= 3 * pi / 4 - 1e-9, f"yaw {yaw} left the outer interval")

    def test_a_free_dimension_is_still_requested_with_nan(self):
        """The check is on ``Bw``; NaN in the *argument* still means "sample this"."""
        Bw = numpy.zeros((6, 2))
        Bw[0] = [-1.0, 1.0]
        tsr = TSR(Bw=Bw)
        request = numpy.zeros(6)
        request[0] = numpy.nan
        self.assertTrue(-1.0 <= tsr.sample_xyzrpy(request)[0] <= 1.0)

    # -- no false rejection --------------------------------------------------

    def test_the_frames_this_library_produces_are_accepted(self):
        """The failure mode a too-tight tolerance would cause: every template the
        factories emit, instantiated, must construct."""
        from tsr.hands import ParallelJawGripper, Robotiq2F85
        from tsr.placement import StablePlacer

        gripper = ParallelJawGripper(finger_length=0.08, max_aperture=0.14)
        placer = StablePlacer(table_x=0.60, table_y=0.60)
        templates = (
            gripper.grasp_cylinder(0.04, 0.12)
            + gripper.grasp_box(0.06, 0.05, 0.10)
            + gripper.grasp_sphere(0.05)
            + gripper.grasp_torus(0.05, 0.012)
            + Robotiq2F85().grasp_cylinder(0.03, 0.10)
            + placer.place_cylinder(0.04, 0.12)
            + placer.place_box(0.10, 0.08, 0.06)
            + placer.place_sphere(0.05)
            + placer.place_torus(0.05, 0.012)
        )
        self.assertGreater(len(templates), 20)
        pose = numpy.eye(4)
        pose[:3, :3] = TSR(Bw=numpy.zeros((6, 2))).to_transform(numpy.zeros(6))[:3, :3]
        for template in templates:
            template.instantiate(pose)  # must not raise


class DistanceWrapsToTheNearestRepresentative(TestCase):
    """`distance` reports the true minimum displacement over equivalent RPY (#186).

    RPY is periodic and double-covering, so a rotation has many representatives. The
    displacement formula has to measure the one nearest the bound interval. It used to wrap
    each angle into ``[lower, lower + 2*pi)``, which for an angle slightly *below* a bound
    narrower than a full turn lands almost a full turn *above* the upper bound -- so a pose
    0.006 rad outside a zero-width bound measured about pi instead of 0.006.

    Found downstream in personalrobotics/ssrobot#129, measuring where Geodude's hand ended
    after executing a planned TSR goal.
    """

    def test_a_pose_just_under_a_rotation_bound_is_millimetres_away(self):
        """The exact case from the issue."""
        Bw = numpy.array([[0, 0], [0, 0], [-0.02, 0.02], [0, 0], [0, 0], [-pi, pi]])
        tsr = TSR(numpy.eye(4), numpy.eye(4), Bw)
        pose = TSR.xyzrpy_to_trans(numpy.array([0.0011, -0.003, 0.0078, -0.006, -0.0036, -0.2026]))

        dist, _ = tsr.distance(pose)
        # the true displacement is the translation plus the two small rotation excesses
        expected = numpy.linalg.norm([0.0011, -0.003, 0.0, -0.006, -0.0036, 0.0])
        self.assertAlmostEqual(dist, expected, places=9)
        self.assertLess(dist, 0.01)  # not ~4.44, which is what wrapping from `lower` gave

    def test_just_below_and_just_above_a_zero_width_bound_agree(self):
        """A zero-width bound is the sharpest case: the two sides must be symmetric.

        Nothing about being on the low side of a bound makes a pose further away, so equal
        offsets either way must measure equally. Wrapping from the lower edge broke exactly
        this symmetry, and only on the low side.
        """
        Bw = numpy.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        tsr = TSR(numpy.eye(4), numpy.eye(4), Bw)
        for offset in (0.5, 0.05, 0.006, 0.002):
            below, _ = tsr.distance(TSR.xyzrpy_to_trans(numpy.array([0, 0, 0, -offset, 0, 0])))
            above, _ = tsr.distance(TSR.xyzrpy_to_trans(numpy.array([0, 0, 0, offset, 0, 0])))
            self.assertAlmostEqual(below, above, places=9, msg=f"asymmetric at offset {offset}")
            self.assertAlmostEqual(below, offset, places=9)

    def test_distance_is_the_minimum_over_every_representative(self):
        """The property the regression violated, checked against an independent oracle.

        The oracle enumerates the nine equivalent RPY triples and, for each, every ``2*pi``
        shift within a few turns, then takes the least displacement. It shares no wrapping
        code with the implementation -- that is the part that was wrong -- so agreement means
        the implementation really found the minimum rather than merely a candidate.
        """
        rng = numpy.random.default_rng(186)

        def oracle(tsr, pose):
            local = numpy.linalg.inv(tsr.T0_w) @ pose @ numpy.linalg.inv(tsr.Tw_e)
            xyz = local[:3, 3]
            r, p, y = TSR.rot_to_rpy(local[:3, :3])
            triples = [(r, p, y)]
            for dr in (pi, -pi):
                for dp in (pi, -pi):
                    for dy in (pi, -pi):
                        triples.append((r + dr, dp - p, y + dy))
            lo, hi = tsr._Bw_cont[:, 0], tsr._Bw_cont[:, 1]
            best = numpy.inf
            for triple in triples:
                for shifts in [
                    (a, b, c) for a in (-2, -1, 0, 1, 2) for b in (-2, -1, 0, 1, 2) for c in (-2, -1, 0, 1, 2)
                ]:
                    angles = numpy.array(triple) + 2 * pi * numpy.array(shifts)
                    dw = numpy.hstack((xyz, angles))
                    dx = numpy.where(dw < lo, dw - lo, numpy.where(dw > hi, dw - hi, 0.0))
                    best = min(best, float(numpy.linalg.norm(dx)))
            return best

        for _ in range(25):
            widths = rng.choice([0.0, 0.2, 1.0, 2 * pi], 3)
            lows = rng.uniform(-pi, pi, 3)
            Bw = numpy.vstack(
                [
                    numpy.column_stack([rng.uniform(-0.05, 0, 3), rng.uniform(0, 0.05, 3)]),
                    numpy.column_stack([lows, lows + widths]),
                ]
            )
            tsr = TSR(numpy.eye(4), numpy.eye(4), Bw)
            pose = TSR.xyzrpy_to_trans(numpy.concatenate([rng.uniform(-0.1, 0.1, 3), rng.uniform(-pi, pi, 3)]))
            if tsr.contains(pose):
                continue  # distance short-circuits to exactly 0 there
            dist, _ = tsr.distance(pose)
            self.assertAlmostEqual(dist, oracle(tsr, pose), places=7)
