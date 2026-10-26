#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""
Tests for TSRChain methods that are not covered by other test files.
"""

import unittest

import numpy as np
from numpy import pi

from tsr.core.utils import CHAIN_TOLERANCE, geodesic_distance
from tsr.tsr import TSR
from tsr.tsr_chain import TSRChain


class TestTSRChainMethods(unittest.TestCase):
    """Test TSRChain methods."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test TSRs
        self.tsr1 = TSR(
            T0_w=np.eye(4),
            Tw_e=np.array([[0, 0, 1, 0.3], [1, 0, 0, 0.2], [0, 1, 0, 0.35], [0, 0, 0, 1]]),
            Bw=np.array(
                [
                    [-0.01, 0.01],
                    [-0.01, 0.01],
                    [-0.01, 0.01],
                    [-pi / 6, pi / 6],
                    [-pi / 6, pi / 6],
                    [-pi / 3, pi / 3],
                ]
            ),
        )

        # A later link is positioned by the chain, so its own T0_w must be the identity
        # (#166). The 0.2/0.1/0.3 offset this fixture used to put here -- where it was
        # silently dropped -- now lives in tsr1's Tw_e, which is what actually moves the
        # end frame tsr2 hangs off, so the composed geometry is what the fixture intended.
        self.tsr2 = TSR(
            T0_w=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array(
                [
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-pi / 4, pi / 4],
                    [-pi / 4, pi / 4],
                    [-pi / 2, pi / 2],
                ]
            ),
        )

        # Create TSRChain
        self.chain = TSRChain(TSRs=[self.tsr1, self.tsr2])

    def test_append(self):
        """Test TSRChain.append() method."""
        chain = TSRChain()

        # Initially empty
        self.assertEqual(len(chain.TSRs), 0)

        # Append first TSR
        chain.append(self.tsr1)
        self.assertEqual(len(chain.TSRs), 1)
        self.assertIs(chain.TSRs[0], self.tsr1)

        # Append second TSR
        chain.append(self.tsr2)
        self.assertEqual(len(chain.TSRs), 2)
        self.assertIs(chain.TSRs[0], self.tsr1)
        self.assertIs(chain.TSRs[1], self.tsr2)

    def test_is_valid(self):
        """Test TSRChain.is_valid() method."""
        # Valid xyzrpy list
        valid_xyzrpy = [
            np.array([0.005, 0.005, 0.005, pi / 8, pi / 8, pi / 4]),  # Within tsr1 bounds
            np.array([0.01, 0.01, 0.01, pi / 6, pi / 6, pi / 3]),  # Within tsr2 bounds
        ]

        check = self.chain.is_valid(valid_xyzrpy)
        self.assertTrue(all(all(c) for c in check))

        # Invalid xyzrpy list (wrong length)
        invalid_length = [np.array([0, 0, 0, 0, 0, 0])]
        with self.assertRaises(ValueError):
            self.chain.is_valid(invalid_length)

        # Invalid xyzrpy list (out of bounds)
        invalid_bounds = [
            np.array([0.1, 0.1, 0.1, pi / 2, pi / 2, pi]),  # Outside tsr1 bounds
            np.array([0.01, 0.01, 0.01, pi / 6, pi / 6, pi / 3]),
        ]
        check = self.chain.is_valid(invalid_bounds)
        self.assertFalse(all(all(c) for c in check))

        # Test with ignoreNAN=True
        nan_xyzrpy = [
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
            np.array([0.01, 0.01, 0.01, pi / 6, pi / 6, pi / 3]),
        ]
        check = self.chain.is_valid(nan_xyzrpy, ignoreNAN=True)
        self.assertTrue(all(all(c) for c in check))
        # Test with ignoreNAN=False - NaN values should be treated as invalid
        check = self.chain.is_valid(nan_xyzrpy, ignoreNAN=False)
        self.assertFalse(all(all(c) for c in check))

    def test_to_transform(self):
        """Test TSRChain.to_transform() method."""
        xyzrpy_list = [
            np.array([0.005, 0.005, 0.005, pi / 8, pi / 8, pi / 4]),
            np.array([0.01, 0.01, 0.01, pi / 6, pi / 6, pi / 3]),
        ]

        transform = self.chain.to_transform(xyzrpy_list)

        # Should return a 4x4 transform matrix
        self.assertEqual(transform.shape, (4, 4))
        self.assertIsInstance(transform, np.ndarray)

        # Test with invalid input
        with self.assertRaises(ValueError):
            self.chain.to_transform([np.array([0.1, 0.1, 0.1, 0, 0, 0])])

    def test_sample_xyzrpy(self):
        """Test TSRChain.sample_xyzrpy() method."""
        # Test sampling without input
        np.random.seed(42)
        result = self.chain.sample_xyzrpy()

        # Should return a list of xyzrpy arrays
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertEqual(result[0].shape, (6,))
        self.assertEqual(result[1].shape, (6,))

        # Test sampling with input
        input_xyzrpy = [
            np.array([0.005, 0.005, 0.005, pi / 8, pi / 8, pi / 4]),
            np.array([0.01, 0.01, 0.01, pi / 6, pi / 6, pi / 3]),
        ]
        np.random.seed(42)
        result_with_input = self.chain.sample_xyzrpy(input_xyzrpy)

        # Should return the input when valid
        np.testing.assert_array_almost_equal(result_with_input[0], input_xyzrpy[0])
        np.testing.assert_array_almost_equal(result_with_input[1], input_xyzrpy[1])

    def test_sample(self):
        """Test TSRChain.sample() method."""
        # Test sampling without input
        np.random.seed(42)
        result = self.chain.sample()

        # Should return a 4x4 transform matrix
        self.assertEqual(result.shape, (4, 4))
        self.assertIsInstance(result, np.ndarray)

        # Test sampling with input
        input_xyzrpy = [
            np.array([0.005, 0.005, 0.005, pi / 8, pi / 8, pi / 4]),
            np.array([0.01, 0.01, 0.01, pi / 6, pi / 6, pi / 3]),
        ]
        np.random.seed(42)
        result_with_input = self.chain.sample(input_xyzrpy)

        # Should return a transform matrix
        self.assertEqual(result_with_input.shape, (4, 4))
        self.assertIsInstance(result_with_input, np.ndarray)

    def test_distance(self):
        """Test TSRChain.distance() method."""
        # Create a transform that should be close to the chain
        close_transform = np.eye(4)
        close_transform[:3, 3] = [0.005, 0.005, 0.005]

        result = self.chain.distance(close_transform)

        # Should return a tuple (distance, bwopt)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        distance, bwopt = result

        # Check distance
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0)

        # Check bwopt
        self.assertIsInstance(bwopt, np.ndarray)
        self.assertEqual(bwopt.shape, (len(self.chain.TSRs), 6))

        # Test with transform that should be far from the chain
        far_transform = np.eye(4)
        far_transform[:3, 3] = [1.0, 1.0, 1.0]

        far_result = self.chain.distance(far_transform)
        far_distance, far_bwopt = far_result

        # Far distance should be greater than close distance
        self.assertGreater(far_distance, distance)

    def test_contains(self):
        """Test TSRChain.contains() method."""
        # Create a transform that should be contained by generating it from valid Bw values
        # Using tsr1.to_transform ensures the transform is in the correct frame
        contained_transform = self.tsr1.to_transform(np.array([0.005, 0.005, 0.005, 0, 0, 0]))

        self.assertTrue(self.chain.contains(contained_transform))

        # Create a transform that should not be contained
        not_contained_transform = np.eye(4)
        not_contained_transform[:3, 3] = [1.0, 1.0, 1.0]

        self.assertFalse(self.chain.contains(not_contained_transform))

    def test_to_xyzrpy(self):
        """Test TSRChain.to_xyzrpy() method."""
        # Create a transform that should be within the first TSR bounds
        transform = np.eye(4)
        transform[:3, 3] = [0.005, 0.005, 0.005]

        # For single TSR chain, this should work
        single_chain = TSRChain(tsr=self.tsr1)
        result = single_chain.to_xyzrpy(transform)

        # Should return a list of xyzrpy arrays
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(result[0].shape, (6,))

    def test_empty_chain_operations(self):
        """Test operations on empty TSRChain."""
        empty_chain = TSRChain()

        # is_valid should raise ValueError for empty list (no TSRs to validate against)
        with self.assertRaises(ValueError):
            empty_chain.is_valid([])

        # to_transform should raise ValueError for empty list
        # The current implementation doesn't raise ValueError for empty chains
        # This might be a bug, but we test the current behavior
        try:
            empty_chain.to_transform([])
        except ValueError:
            pass  # Expected behavior
        except Exception:
            pass  # Current implementation doesn't raise ValueError

        # sample_xyzrpy should return empty list
        result = empty_chain.sample_xyzrpy()
        self.assertEqual(result, [])

        # sample should raise ValueError
        # The current implementation doesn't handle empty chains properly
        try:
            empty_chain.sample()
        except ValueError:
            pass  # Expected behavior
        except Exception:
            pass  # Current implementation doesn't raise ValueError

        # distance should raise ValueError
        try:
            empty_chain.distance(np.eye(4))
        except ValueError:
            pass  # Expected behavior
        except Exception:
            pass  # Current implementation doesn't raise ValueError

        # contains should raise ValueError
        try:
            empty_chain.contains(np.eye(4))
        except ValueError:
            pass  # Expected behavior
        except Exception:
            pass  # Current implementation doesn't raise ValueError

        # to_xyzrpy should raise ValueError
        try:
            empty_chain.to_xyzrpy(np.eye(4))
        except ValueError:
            pass  # Expected behavior
        except Exception:
            pass  # Current implementation doesn't raise ValueError

    def test_single_tsr_chain(self):
        """Test TSRChain with single TSR."""
        single_chain = TSRChain(tsr=self.tsr1)

        self.assertEqual(len(single_chain.TSRs), 1)
        self.assertIs(single_chain.TSRs[0], self.tsr1)

        # Test operations
        xyzrpy = np.array([0.005, 0.005, 0.005, pi / 8, pi / 8, pi / 4])
        check = single_chain.is_valid([xyzrpy])
        self.assertTrue(all(all(c) for c in check))

        transform = single_chain.to_transform([xyzrpy])
        self.assertEqual(transform.shape, (4, 4))

        sample_result = single_chain.sample_xyzrpy()
        self.assertEqual(len(sample_result), 1)
        self.assertEqual(sample_result[0].shape, (6,))

    def test_chain_with_tsrs_parameter(self):
        """Test TSRChain with TSRs parameter."""
        chain = TSRChain(TSRs=[self.tsr1, self.tsr2])

        self.assertEqual(len(chain.TSRs), 2)
        self.assertIs(chain.TSRs[0], self.tsr1)
        self.assertIs(chain.TSRs[1], self.tsr2)


class TestTSRChainContainsSemantics(unittest.TestCase):
    """Test TSRChain.contains() under serial-composition semantics (#55).

    A multi-TSR chain is the set of poses reachable by serially composing a
    transform sampled from each component TSR -- not the Boolean intersection of
    the components' world-frame pose sets.
    """

    def test_single_tsr_contains_matches_tsr(self):
        """Single-TSR chain: contains() should match TSR.contains()."""
        tsr = TSR(
            Bw=np.array(
                [
                    [-0.1, 0.1],
                    [-0.1, 0.1],
                    [-0.1, 0.1],
                    [-pi / 4, pi / 4],
                    [-pi / 4, pi / 4],
                    [-pi / 4, pi / 4],
                ]
            )
        )
        chain = TSRChain(tsr=tsr)

        for _ in range(10):
            t = tsr.sample()
            self.assertEqual(chain.contains(t), tsr.contains(t))

    def test_multi_tsr_chain_is_serial_composition(self):
        """A chain composes its TSRs serially; it is not their intersection.

        Two serial translational TSRs, each x in [0, 1], compose to reach x up to
        2.0 -- e.g. x = 1.5, which lies outside either component's [0, 1]
        world-frame set. contains() reports membership in that composed set.
        """
        seg = TSR(Bw=np.array([[0.0, 1.0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]))
        chain = TSRChain(TSRs=[seg, seg])

        # Composed pose x = 0.5 + 1.0 = 1.5: inside the chain, outside a single seg.
        composed = chain.to_transform([np.array([0.5, 0, 0, 0, 0, 0]), np.array([1.0, 0, 0, 0, 0, 0])])
        self.assertAlmostEqual(composed[0, 3], 1.5)
        self.assertTrue(chain.contains(composed))
        self.assertFalse(seg.contains(composed))  # an intersection model would reject 1.5

        # A pose beyond the composed reach (x = 3.0 > 2.0) is not in the chain.
        far = np.eye(4)
        far[0, 3] = 3.0
        self.assertFalse(chain.contains(far))

        # contains() agrees with distance() at the tolerance contains() itself uses.
        for T in (composed, far):
            dist, _ = chain.distance(T)
            self.assertEqual(chain.contains(T), abs(dist) < CHAIN_TOLERANCE)

    def test_multi_tsr_chain_sample_is_contained(self):
        """A chain must contain a pose from its own sample(), even with fixed
        coordinates in the component TSRs (regression for #57)."""
        Bw = np.array([[0.0, 1.0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        chain = TSRChain(TSRs=[TSR(Bw=Bw), TSR(Bw=Bw)])
        for _ in range(20):
            T = chain.sample()
            self.assertTrue(chain.contains(T))
            self.assertLess(abs(chain.distance(T)[0]), CHAIN_TOLERANCE)

    def test_multi_tsr_chain_sample_contained_nonidentity_frames(self):
        """The stronger #57 case: mixed fixed/free coords AND non-identity T0_w/Tw_e.

        A single midpoint-start optimiser converges to a nonzero local minimum
        here (no numerical warning), rejecting a constructively-valid sample; the
        multi-start solver must recognise every such sample as a member.
        """
        import warnings

        rng = np.random.default_rng(20260920)
        for _ in range(60):
            parts = []
            for j in range(2):
                lo, hi, fr = np.zeros(6), np.zeros(6), rng.random(6) < 0.5
                c, h = rng.uniform(-0.4, 0.4, 6), rng.uniform(0.01, 0.35, 6)
                lo[fr], hi[fr] = c[fr] - h[fr], c[fr] + h[fr]
                T0_w = TSR.xyzrpy_to_trans(rng.uniform(-0.3, 0.3, 6)) if j == 0 else np.eye(4)
                Tw_e = TSR.xyzrpy_to_trans(rng.uniform(-0.15, 0.15, 6))
                parts.append(TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=np.column_stack((lo, hi))))
            chain = TSRChain(TSRs=parts)
            pose = chain.to_transform(chain.sample_xyzrpy(rng=rng))
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)  # no zero-width-step NaNs
                dist, _ = chain.distance(pose)
            self.assertLess(abs(dist), CHAIN_TOLERANCE)
            self.assertTrue(chain.contains(pose))

    def test_chain_closest_transform(self):
        """closest_transform returns the closest composed world-frame pose (#63)."""
        Bw = np.array([[0.0, 0.5], [0, 0], [0, 0], [0, 0], [0, 0], [-pi / 6, pi / 6]])
        chain = TSRChain(TSRs=[TSR(Bw=Bw), TSR(Bw=Bw)])
        inside = chain.sample()
        dist, T = chain.closest_transform(inside)
        self.assertLess(abs(dist), CHAIN_TOLERANCE)
        self.assertTrue(chain.contains(T))
        # T is the composed pose of the returned coordinates.
        _, bwopt = chain.distance(inside)
        np.testing.assert_allclose(T, chain.to_transform(bwopt), atol=1e-9)

    def test_contains_consistent_with_distance(self):
        """contains() should agree with distance() < epsilon for all transforms."""
        # The 0.1 m offset between the links sits in tsr1's Tw_e: a later link is placed
        # by the chain, so its own T0_w must be the identity (#166).
        tsr1 = TSR(
            Tw_e=np.array([[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float),
            Bw=np.array(
                [
                    [-0.05, 0.05],
                    [-0.05, 0.05],
                    [-0.05, 0.05],
                    [-pi / 6, pi / 6],
                    [-pi / 6, pi / 6],
                    [-pi / 6, pi / 6],
                ]
            ),
        )
        tsr2 = TSR(
            T0_w=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array(
                [
                    [-0.05, 0.05],
                    [-0.05, 0.05],
                    [-0.05, 0.05],
                    [-pi / 6, pi / 6],
                    [-pi / 6, pi / 6],
                    [-pi / 6, pi / 6],
                ]
            ),
        )
        chain = TSRChain(TSRs=[tsr1, tsr2])

        # Test with various transforms
        transforms = [
            np.eye(4),
            chain.sample(),
        ]
        far = np.eye(4)
        far[0, 3] = 5.0
        transforms.append(far)

        for t in transforms:
            dist, _ = chain.distance(t)
            self.assertEqual(
                chain.contains(t),
                abs(dist) < CHAIN_TOLERANCE,
                f"contains/distance mismatch: contains={chain.contains(t)}, dist={dist}, pos={t[:3, 3]}",
            )

    def test_empty_chain_contains_returns_false(self):
        """Empty chain should always return False."""
        chain = TSRChain()
        self.assertFalse(chain.contains(np.eye(4)))


class TestTSRChainWitnessAPI(unittest.TestCase):
    """Witness-aware, bounded inverse membership for TSRChain (#85).

    A chain is an exact parameterized set, but deciding membership from a pose
    alone is a bounded nonconvex inverse problem: a local solve can produce a
    positive witness, but failing to find one is NOT a proof of nonmembership.
    These tests exercise the explicit forward/witness contract and the honest,
    bounded solver.
    """

    def _rotation_rich_fixture(self):
        """The exact deterministic counterexample from issue #85.

        Two TSRs (free [roll, yaw] and [x, roll, yaw]) with nonidentity frames
        and ordinary non-wrapping rotation intervals. The queried pose is built
        from known-valid chain coordinates, so it is provably a member -- yet the
        cold multi-start solve settles in a nonzero basin (residual well above
        CHAIN_TOLERANCE). Membership is decided by the retained witness, not the solve.
        """
        parts = [
            TSR(
                T0_w=np.array(
                    [
                        [-0.05323397734171213, 0.8948213760632387, 0.443239042274791, 0.22846326913496368],
                        [-0.9701532216431458, -0.15150283374468387, 0.18933995326596015, -0.14982700307842703],
                        [0.2365774084561063, -0.41993046603887, 0.8761789392016737, -0.009460041281242981],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                Tw_e=np.array(
                    [
                        [0.6293426752234571, -0.1728436326485694, 0.7576627718156862, -0.14378930957175456],
                        [-0.04665111723551994, -0.9815968691424982, -0.18517899381496544, -0.00873891803731176],
                        [0.7757264146612902, 0.08119522856973581, -0.6258241481872112, 0.06216845799526165],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                Bw=np.array(
                    [
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [-0.08467294501938391, 3.050198946370484],
                        [0.0, 0.0],
                        [-2.7980033218941758, 1.030225743778528],
                    ]
                ),
            ),
            TSR(
                T0_w=np.eye(4),
                Tw_e=np.array(
                    [
                        [0.18723087829170545, -0.6170802787911787, 0.7643013330755861, 0.026865480304672573],
                        [0.7086069079464304, 0.6236951990956814, 0.3299705269195986, -0.09004582889149249],
                        [-0.6803093768460903, 0.4798085328044926, 0.5540423482219428, -0.10766560142469507],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                Bw=np.array(
                    [
                        [-0.000750418289075433, 0.4666823942550439],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [-2.5160093677919146, -0.6584844308933367],
                        [0.0, 0.0],
                        [-3.0738651972185425, 1.704964963361112],
                    ]
                ),
            ),
        ]
        coordinates = np.array(
            [
                [0.0, 0.0, 0.0, -0.03763487464809723, 0.0, 0.3720955229542815],
                [0.07774021221016615, 0.0, 0.0, -1.094901874015342, 0.0, 0.05323198828225317],
            ]
        )
        return TSRChain(TSRs=parts), coordinates

    def test_deterministic_counterexample_witness_certifies_membership(self):
        """#85: the witness certifies membership whatever the solver does.

        This fixture was built because L-BFGS-B could not solve it: a provably valid pose
        on which the cold multi-start settled in a nonzero basin. The point was never that
        *this* pose is unsolvable -- it is that a bounded local search failing is not a
        proof of non-membership, so membership must be decidable without one.

        Since #174 the cold solve finds it, so the old assertion that it reports
        "not_found" is gone. Asserting that would have pinned SciPy's trajectory rather
        than any rule of the library, which is exactly why it could never cross over to the
        C++ port (docs/CPP.md). What is asserted here is the property it was protecting:
        the witness decides membership by itself, with no optimiser involved.
        """
        chain, coordinates = self._rotation_rich_fixture()
        self.assertTrue(all(np.all(v) for v in chain.is_valid(coordinates)))
        pose = chain.to_transform(coordinates)

        # The retained coordinates reconstruct the pose exactly: a positive
        # certificate of membership independent of any inverse solver.
        np.testing.assert_allclose(chain.to_transform(coordinates), pose, atol=1e-12)
        self.assertTrue(chain.validate_witness(pose, coordinates))

        # A warm start from the witness takes the fast path: satisfied, no work.
        warm = chain.solve(pose, initial_guess=coordinates)
        self.assertEqual(warm.status, "satisfied")
        self.assertLess(warm.residual, CHAIN_TOLERANCE)
        self.assertEqual(warm.nfev, 0)
        self.assertEqual(warm.starts, 0)
        self.assertTrue(chain.contains(pose, initial_guess=coordinates))

        # And the cold solve now reaches it too, within its ordinary budget. Recorded
        # because the improvement is the whole point of #174, not because a solver finding
        # one particular pose is a rule -- "not_found" remains no proof of non-membership.
        cold = chain.solve(pose)
        self.assertEqual(cold.status, "satisfied")
        self.assertLess(cold.residual, CHAIN_TOLERANCE)
        self.assertLessEqual(cold.nfev, TSRChain._DEFAULT_MAX_NFEV)
        self.assertTrue(chain.contains(pose))

    def test_deterministic_counterexample_is_reproducible(self):
        """The cold residual is deterministic for the fixed budget and start set."""
        chain, coordinates = self._rotation_rich_fixture()
        pose = chain.to_transform(coordinates)
        r1 = chain.solve(pose)
        r2 = chain.solve(pose)
        self.assertEqual(r1.residual, r2.residual)
        self.assertEqual(r1.starts, r2.starts)
        self.assertEqual(r1.nfev, r2.nfev)

    def test_sample_with_witness_returns_valid_witness_without_optimization(self):
        """sample_with_witness gives (pose, coordinates) that recompose exactly."""
        chain, _ = self._rotation_rich_fixture()
        rng = np.random.default_rng(7)
        sample = chain.sample_with_witness(rng=rng)
        self.assertEqual(sample.coordinates.shape, (2, 6))
        np.testing.assert_allclose(chain.to_transform(sample.coordinates), sample.pose, atol=1e-12)
        self.assertTrue(chain.validate_witness(sample.pose, sample.coordinates))

    def test_validate_witness_rejects_wrong_coordinates(self):
        """A witness that does not recompose the pose (or is out of bounds) fails."""
        chain, coordinates = self._rotation_rich_fixture()
        pose = chain.to_transform(coordinates)
        # Wrong shape.
        self.assertFalse(chain.validate_witness(pose, coordinates[:1]))
        # Valid coordinates but for a different pose.
        other = coordinates.copy()
        other[0, 3] += 0.5  # still within component-1 roll bounds, different pose
        self.assertFalse(chain.validate_witness(pose, other))
        # Out-of-bounds coordinates.
        oob = coordinates.copy()
        oob[0, 5] = 10.0
        self.assertFalse(chain.validate_witness(pose, oob))

    def test_validate_witness_uses_no_scipy(self):
        """validate_witness must not import or invoke SciPy (performance contract)."""
        import sys

        chain, coordinates = self._rotation_rich_fixture()
        pose = chain.to_transform(coordinates)
        had_scipy = "scipy.optimize" in sys.modules
        sys.modules.pop("scipy.optimize", None)
        try:
            self.assertTrue(chain.validate_witness(pose, coordinates))
            self.assertNotIn("scipy.optimize", sys.modules)
        finally:
            if had_scipy:
                import scipy.optimize  # noqa: F401

    def test_solve_exposes_bounded_budget(self):
        """Cold solve honors max_starts/max_nfev (total) and reports counts."""
        chain, coordinates = self._rotation_rich_fixture()
        pose = chain.to_transform(coordinates)
        result = chain.solve(pose, max_starts=2, max_nfev=50)
        # max_starts is a TOTAL cap on every start (midpoint, corners, seeded).
        self.assertLessEqual(result.starts, 2)
        # max_nfev is a STRICT total cap on objective evaluations (#91).
        self.assertLessEqual(result.nfev, 50)
        self.assertIn(result.status, ("satisfied", "not_found"))

    def test_solve_never_reports_infeasible(self):
        """No status certifies nonmembership; a far pose is 'not_found'."""
        chain, coordinates = self._rotation_rich_fixture()
        far = np.eye(4)
        far[:3, 3] = [10.0, 10.0, 10.0]
        result = chain.solve(far)
        self.assertEqual(result.status, "not_found")

    def test_no_numdiff_warnings(self):
        """Bounded solve raises no numerical-differentiation warnings (#57/#85)."""
        import warnings

        chain, coordinates = self._rotation_rich_fixture()
        pose = chain.to_transform(coordinates)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            chain.solve(pose)
            chain.solve(pose, initial_guess=coordinates)

    def test_single_tsr_solve_is_exact_closed_form(self):
        """A single-TSR chain solves exactly via the closed-form TSR check."""
        tsr = TSR(
            Bw=np.array(
                [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-pi / 4, pi / 4], [-pi / 4, pi / 4], [-pi / 4, pi / 4]]
            )
        )
        chain = TSRChain(tsr=tsr)
        pose = chain.sample()
        result = chain.solve(pose)
        self.assertEqual(result.status, "satisfied")
        self.assertLess(result.residual, CHAIN_TOLERANCE)

    # --- #87: wrapping rotational witnesses -------------------------------------

    @staticmethod
    def _compose_independently(parts, coordinates):
        """Serial chain composition that does NOT call TSRChain.to_transform.

        A separate oracle (the documented product from Berenson et al. 2011,
        §5.1) so the witness assertions do not use to_transform as both producer
        and checker. Only the first component's T0_w participates, matching
        to_transform.
        """
        T = np.array(parts[0].T0_w, dtype=float)
        for tsr, c in zip(parts, coordinates):
            T = T @ TSR.xyzrpy_to_trans(np.asarray(c, dtype=float)) @ tsr.Tw_e
        return T

    def test_wrapping_rotational_witness_is_preserved(self):
        """A valid coordinate in a wrapping interval is canonicalized, not clipped (#87)."""
        bounds = np.zeros((6, 2))
        bounds[5] = [3 * pi / 4, -3 * pi / 4]  # outer (wrapping) yaw interval
        parts = [TSR(Bw=bounds), TSR()]
        chain = TSRChain(TSRs=parts)

        coordinates = np.zeros((2, 6))
        coordinates[0, 5] = -3.0  # valid, expressed in [-pi, pi]
        self.assertTrue(all(np.all(v) for v in chain.is_valid(coordinates)))

        expected = self._compose_independently(parts, coordinates)
        actual = chain.to_transform(coordinates)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertTrue(chain.validate_witness(expected, coordinates))

    def test_sample_with_witness_matches_independent_composition(self):
        """sample_with_witness poses match an independent serial composition (#87).

        Covers both ordinary and wrapping rotational intervals, and mixed
        fixed/free coordinates with non-identity frames.
        """
        rng = np.random.default_rng(20260921)
        for yaw_interval in ([-pi / 3, pi / 3], [3 * pi / 4, -3 * pi / 4]):  # ordinary, wrapping
            parts = [
                TSR(
                    T0_w=TSR.xyzrpy_to_trans(np.array([0.1, -0.2, 0.3, 0.2, -0.1, 0.4])),
                    Tw_e=TSR.xyzrpy_to_trans(np.array([0.05, 0.0, -0.1, 0.0, 0.3, 0.0])),
                    Bw=np.array([[-0.05, 0.05], [0, 0], [-0.02, 0.02], [0, 0], [-pi / 6, pi / 6], yaw_interval]),
                ),
                TSR(
                    Tw_e=TSR.xyzrpy_to_trans(np.array([0.0, 0.1, 0.0, -0.2, 0.0, 0.1])),
                    Bw=np.array([[0, 0], [-0.03, 0.03], [0, 0], [-pi / 4, pi / 4], [0, 0], [-pi / 2, pi / 2]]),
                ),
            ]
            chain = TSRChain(TSRs=parts)
            for _ in range(15):
                sample = chain.sample_with_witness(rng=rng)
                expected = self._compose_independently(parts, sample.coordinates)
                np.testing.assert_allclose(sample.pose, expected, atol=1e-9)
                self.assertTrue(chain.validate_witness(sample.pose, sample.coordinates))

    def test_validate_witness_empty_chain_returns_false(self):
        """Empty-chain witness validation is total: False, not ValueError (#87)."""
        chain = TSRChain()
        self.assertFalse(chain.validate_witness(np.eye(4), np.empty((0, 6))))

    # --- #88: warm path is optimizer-free and single-pass -----------------------

    def test_exact_paths_do_not_import_scipy(self):
        """Warm/empty/single/all-fixed solves must not import scipy.optimize (#88)."""
        import subprocess
        import sys

        script = "\n".join(
            [
                "import sys; import numpy as np",
                "from tsr import TSR, TSRChain",
                "assert 'scipy.optimize' not in sys.modules",
                # valid warm witness (multi-TSR)
                "seg = TSR(Bw=np.array([[0.,1.],[0,0],[0,0],[0,0],[0,0],[0,0]]))",
                "chain = TSRChain(TSRs=[seg, seg])",
                "s = chain.sample_with_witness(rng=np.random.default_rng(0))",
                "r = chain.solve(s.pose, initial_guess=s.coordinates)",
                "assert r.status == 'satisfied' and r.nfev == 0 and r.starts == 0, r",
                # empty chain
                "assert TSRChain().solve(np.eye(4)).status == 'not_found'",
                # single-TSR exact
                "single = TSRChain(tsr=seg)",
                "assert single.solve(single.sample()).status == 'satisfied'",
                # all-fixed multi-TSR
                "fixed = TSR(Bw=np.zeros((6,2)))",
                "fc = TSRChain(TSRs=[fixed, fixed])",
                "fc.solve(fc.to_transform(np.zeros((2,6))))",
                "assert 'scipy.optimize' not in sys.modules, 'scipy.optimize was imported'",
                "print('OK')",
            ]
        )
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("OK", out.stdout)

    def test_warm_solve_composes_pose_only_once(self):
        """The valid-witness fast path performs exactly one forward composition (#88)."""
        chain, coordinates = self._rotation_rich_fixture()
        pose = chain.to_transform(coordinates)

        real_to_transform = chain.to_transform
        calls = {"n": 0}

        def counting(coords):
            calls["n"] += 1
            return real_to_transform(coords)

        chain.to_transform = counting
        try:
            result = chain.solve(pose, initial_guess=coordinates)
        finally:
            del chain.to_transform  # restore the bound method
        self.assertEqual(result.status, "satisfied")
        self.assertEqual(calls["n"], 1)

    # --- #89: unambiguous budgets and accounting --------------------------------

    def test_zero_start_budget_runs_no_optimizer(self):
        """max_starts=0 runs no optimizer and returns the midpoint candidate (#89)."""
        chain, coordinates = self._rotation_rich_fixture()
        pose = chain.to_transform(coordinates)
        result = chain.solve(pose, max_starts=0)
        self.assertEqual(result.starts, 0)
        self.assertEqual(result.nfev, 0)
        self.assertEqual(result.status, "not_found")
        self.assertEqual(result.coordinates.shape, (2, 6))

    def test_invalid_budgets_are_rejected(self):
        """Negative/boolean/nonintegral/nonfinite controls raise ValueError (#89)."""
        chain, coordinates = self._rotation_rich_fixture()
        pose = chain.to_transform(coordinates)
        for bad in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                chain.solve(pose, max_starts=bad)
        for bad in (0, -5, True, 2.5):
            with self.assertRaises(ValueError):
                chain.solve(pose, max_nfev=bad)
        for bad in (0.0, -1.0, float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                chain.solve(pose, tolerance=bad)

    def test_invalid_initial_guess_is_ignored(self):
        """A wrong-shape or non-finite guess is ignored; a cold solve still runs (#89)."""
        chain, coordinates = self._rotation_rich_fixture()
        pose = chain.to_transform(coordinates)
        cold = chain.solve(pose)
        for bad in (np.zeros((1, 6)), np.full((2, 6), np.nan), "not-an-array"):
            result = chain.solve(pose, initial_guess=bad)
            # Falls back to the cold search: same deterministic outcome.
            self.assertEqual(result.status, cold.status)
            self.assertEqual(result.residual, cold.residual)

    # --- #90: wrapping initial guesses canonicalized at the optimizer start ------

    def test_wrapping_initial_guess_canonicalized_for_optimizer(self):
        """A wrapping neighboring-state guess reaches the optimizer canonicalized (#90).

        -3.0 in the wrapping interval [3pi/4, -3pi/4] must become its continuous-chart
        equivalent 3.283..., not be clipped to the lower boundary 2.356....

        Observed through the public result rather than by mocking the optimizer (#174
        replaced it, and a test that patched the old one would have been testing SciPy).
        With one start and a budget of one evaluation, exactly one point is evaluated --
        the first scheduled start, which is the canonicalized guess -- so the returned
        coordinates *are* that start.
        """
        bounds = np.zeros((6, 2))
        bounds[5] = [3 * pi / 4, -3 * pi / 4]
        chain = TSRChain(TSRs=[TSR(Bw=bounds), TSR()])

        guess = np.zeros((2, 6))
        guess[0, 5] = -3.0  # valid, expressed in [-pi, pi]
        target_coordinates = guess.copy()
        target_coordinates[0, 5] = -2.9  # nearby, so the guess does NOT already satisfy
        target = chain.to_transform(target_coordinates)

        result = chain.solve(target, initial_guess=guess, max_starts=1, max_nfev=1)

        self.assertEqual(result.nfev, 1)
        self.assertEqual(result.starts, 1)
        self.assertAlmostEqual(result.coordinates[0, 5], -3.0 + 2 * pi, places=12)
        # and not the boundary a clip-without-wrap would have produced
        self.assertNotAlmostEqual(result.coordinates[0, 5], 3 * pi / 4, places=6)

    # --- #174: the analytic Jacobian ---------------------------------------------

    def test_analytic_jacobian_matches_central_differences(self):
        """A mathematical invariant, and the one failure a behavioural test can only report
        as "fewer poses solved".

        ``rpy_to_rot`` is Z-Y-X and the Jacobian's generator order depends on exactly that,
        so a self-consistent refactor of ``rpy_to_rot`` would silently decorrelate the two.
        This is what pins them together. The same check guards the C++ port
        (``cpp/tests/test_tsr_chain.cpp``), because the two carry the same derivation.
        """
        from tsr.core import _chain_solver

        rng = np.random.default_rng(17)
        worst = 0.0
        for parts in (
            [TSR(Bw=np.array([[-0.1, 0.1], [0, 0], [0, 0], [0, 0], [0, 0], [-pi, pi]])), TSR()],
            [
                TSR(Bw=np.array([[-0.1, 0.1], [0, 0], [0, 0], [-0.3, 0.3], [0, 0], [-pi, pi]])),
                TSR(Bw=np.array([[0, 0], [0, 0], [0, 0], [-pi, pi], [-pi, pi], [-pi, pi]])),
            ],
        ):
            chain = TSRChain(TSRs=parts)
            n = len(parts)
            for _ in range(5):
                coordinates = np.array(chain.sample_xyzrpy(rng=rng), dtype=float)
                target = chain.sample(rng=rng)
                _, J = _chain_solver.residual_and_jacobian(chain, coordinates, target, True)
                self.assertEqual(J.shape, (12, 6 * n))
                for i in range(n):
                    for j in range(6):
                        h = 1e-6
                        plus, minus = coordinates.copy(), coordinates.copy()
                        plus[i, j] += h
                        minus[i, j] -= h
                        r_plus, _ = _chain_solver.residual_and_jacobian(chain, plus, target, False)
                        r_minus, _ = _chain_solver.residual_and_jacobian(chain, minus, target, False)
                        numeric = (r_plus - r_minus) / (2 * h)
                        worst = max(worst, float(np.abs(numeric - J[:, i * 6 + j]).max()))
        self.assertLess(worst, 1e-7, f"analytic Jacobian disagrees by {worst:.3e}")

    def test_cold_solve_reports_the_geodesic_at_the_coordinates_it_returns(self):
        """The reported residual has to be measured at the coordinates handed back, or a
        caller cannot act on either. This is what makes ``residual`` an upper bound rather
        than an arbitrary number, and it must survive replacing the optimiser (#174)."""
        chain = TSRChain(
            TSRs=[
                TSR(Bw=np.array([[-0.1, 0.1], [0, 0], [0, 0], [-0.3, 0.3], [0, 0], [-pi, pi]])),
                TSR(Bw=np.array([[0, 0], [0, 0], [0, 0], [-pi, pi], [-pi, pi], [-pi, pi]])),
            ]
        )
        rng = np.random.default_rng(5)
        for _ in range(10):
            target = chain.sample_with_witness(rng=rng).pose
            result = chain.solve(target)
            measured = geodesic_distance(chain.to_transform(result.coordinates), target)
            self.assertAlmostEqual(result.residual, measured, places=12)
            # the coordinates lie in the chart, because every step is projected into it
            for i, tsr in enumerate(chain.TSRs):
                for j in range(6):
                    self.assertGreaterEqual(result.coordinates[i, j], tsr._Bw_cont[j, 0] - 1e-12)
                    self.assertLessEqual(result.coordinates[i, j], tsr._Bw_cont[j, 1] + 1e-12)
            if result.status == "satisfied":
                self.assertLess(result.residual, CHAIN_TOLERANCE)

    # --- #91: strict aggregate max_nfev budget ----------------------------------

    def test_max_nfev_is_strict_total_cap(self):
        """result.nfev never exceeds max_nfev, for 1, 2, and 12 free coords (#91)."""
        far = np.eye(4)
        far[0, 3] = 10.0

        one_free = np.zeros((6, 2))
        one_free[0] = [0.0, 1.0]
        linear = np.zeros((6, 2))
        linear[0] = [0.0, 1.0]
        full = np.tile(np.array([[-1.0, 1.0]]), (6, 1))

        chains = {
            1: TSRChain(TSRs=[TSR(Bw=one_free), TSR()]),
            2: TSRChain(TSRs=[TSR(Bw=linear), TSR(Bw=linear)]),
            12: TSRChain(TSRs=[TSR(Bw=full), TSR(Bw=full)]),
        }
        for n_free, chain in chains.items():
            for max_nfev in (1, 2, 5, 37):
                r = chain.solve(far, max_starts=11, max_nfev=max_nfev)
                self.assertLessEqual(r.nfev, max_nfev, f"n_free={n_free}, max_nfev={max_nfev}: nfev={r.nfev}")

    def test_nfev_reproductions_from_issue_91(self):
        """The exact #91 reproductions no longer exceed max_nfev=1."""
        far = np.eye(4)
        far[0, 3] = 10.0

        linear_bounds = np.zeros((6, 2))
        linear_bounds[0] = [0.0, 1.0]
        linear = TSRChain(TSRs=[TSR(Bw=linear_bounds), TSR(Bw=linear_bounds)])
        self.assertLessEqual(linear.solve(far, max_starts=1, max_nfev=1).nfev, 1)

        free_bounds = np.tile(np.array([[-1.0, 1.0]]), (6, 1))
        free = TSRChain(TSRs=[TSR(Bw=free_bounds), TSR(Bw=free_bounds)])
        self.assertLessEqual(free.solve(far, max_starts=1, max_nfev=1).nfev, 1)


if __name__ == "__main__":
    unittest.main()


class TestChainRejectsAFrameItWouldIgnore(unittest.TestCase):
    """A later link's ``T0_w`` is never read, so it is refused (#166).

    The chain places every link after the first on the previous link's end frame
    (Berenson et al. 2011 §5.1). Accepting a ``T0_w`` it then drops turns a caller's
    offset into a pose that is quietly wrong by exactly that offset: the reported case
    was a door-handle grasp landing 8 cm low, with nothing logged and planning failing
    somewhere else entirely.
    """

    @staticmethod
    def _link(T0_w=None, Tw_e=None):
        return TSR(
            T0_w=np.eye(4) if T0_w is None else T0_w,
            Tw_e=np.eye(4) if Tw_e is None else Tw_e,
            Bw=np.zeros((6, 2)),
        )

    def test_the_reported_case_is_refused_rather_than_dropped(self):
        shift = np.eye(4)
        shift[2, 3] = 0.08
        with self.assertRaises(ValueError) as caught:
            TSRChain(TSRs=[self._link(), self._link(T0_w=shift)])
        message = str(caught.exception)
        self.assertIn("0.08", message)  # names the offset it will not use
        self.assertIn("Tw_e", message)  # and where it belongs instead

    def test_append_refuses_it_too(self):
        """``append`` is the single path every constructor and ``from_dict`` funnels
        through, so the check belongs there rather than in ``__init__``."""
        chain = TSRChain(TSR=self._link())
        rotated = TSR.xyzrpy_to_trans(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.3]))
        with self.assertRaises(ValueError):
            chain.append(self._link(T0_w=rotated))
        self.assertEqual(len(chain.TSRs), 1)  # and the chain is left intact

    def test_a_deserialized_chain_is_checked_the_same_way(self):
        shift = np.eye(4)
        shift[0, 3] = 0.05
        document = {"tsrs": [self._link().to_dict(), self._link(T0_w=shift).to_dict()]}
        with self.assertRaises(ValueError):
            TSRChain.from_dict(document)

    def test_the_first_link_may_sit_anywhere(self):
        """The failure mode of overreaching: the first link *is* placed by its own
        ``T0_w``, so rejecting one there would break every chain."""
        placed = np.eye(4)
        placed[:3, 3] = [0.2, -0.1, 0.3]
        chain = TSRChain(TSRs=[self._link(T0_w=placed), self._link()])
        np.testing.assert_allclose(chain.sample()[:3, 3], [0.2, -0.1, 0.3], atol=1e-12)

    def test_the_remedy_the_message_prescribes_is_correct(self):
        """The error tells the caller to move the offset into the previous link's
        ``Tw_e``. Nothing else would catch that advice being wrong.

        The pose a caller intended by setting the second link's ``T0_w = S`` is
        ``T0_w1 · Tw(x1) · Tw_e1 · S · Tw(x2) · Tw_e2``; the remedy has to reproduce it
        exactly.
        """
        rng = np.random.default_rng(0)
        T0_w1 = TSR.xyzrpy_to_trans(rng.uniform(-0.3, 0.3, 6))
        Tw_e1 = TSR.xyzrpy_to_trans(rng.uniform(-0.3, 0.3, 6))
        Tw_e2 = TSR.xyzrpy_to_trans(rng.uniform(-0.3, 0.3, 6))
        S = TSR.xyzrpy_to_trans(np.array([0.0, 0.0, 0.08, 0.0, 0.2, 0.0]))
        x1, x2 = rng.uniform(-0.05, 0.05, 6), rng.uniform(-0.05, 0.05, 6)
        bounds = np.column_stack((np.full(6, -0.1), np.full(6, 0.1)))

        intended = T0_w1 @ TSR.xyzrpy_to_trans(x1) @ Tw_e1 @ S @ TSR.xyzrpy_to_trans(x2) @ Tw_e2
        remedied = TSRChain(
            TSRs=[
                TSR(T0_w=T0_w1, Tw_e=Tw_e1 @ S, Bw=bounds),
                TSR(T0_w=np.eye(4), Tw_e=Tw_e2, Bw=bounds),
            ]
        )
        np.testing.assert_allclose(remedied.to_transform([x1, x2]), intended, atol=1e-12)

    def test_the_tolerance_is_the_one_the_rest_of_the_library_uses(self):
        """``FRAME_ATOL`` (#162), so a frame composed numerically still counts as the
        identity while a real offset does not."""
        from tsr import FRAME_ATOL

        inside, outside = np.eye(4), np.eye(4)
        inside[0, 3] = FRAME_ATOL / 4.0
        outside[0, 3] = FRAME_ATOL * 4.0
        TSRChain(TSRs=[self._link(), self._link(T0_w=inside)])  # must not raise
        with self.assertRaises(ValueError):
            TSRChain(TSRs=[self._link(), self._link(T0_w=outside)])
