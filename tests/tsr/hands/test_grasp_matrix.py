#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Hypothesis property matrix over every primitive grasp factory (issue #73).

Every public ``grasp_*`` factory -- including the combined entry points and the
named grippers -- is driven through the independent #67 analytic oracle. Example
budgets scale with ``TSR_MATRIX_SCALE`` (default 1; the release gate uses 10); see
``docs/ARCHITECTURE.md`` ("Verification matrix").
"""

import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from tsr.hands import GripperBase

from .._hypothesis_strategies import transforms
from ._grasp_matrix import (
    BOUNDARIES,
    FACTORIES,
    GraspCase,
    GripperSpec,
    _respawn,
    any_cases,
    apply_override,
    boundary_clearances,
    boundary_transition_failures,
    boundary_triplet,
    combined_failures,
    coverage_failures,
    equivariance_failures,
    grasp_cases,
    invalid_overrides_for,
    matrix_settings,
    mirror_family_failures,
    monotonicity_failures,
    oversize_dims,
    public_grasp_factories,
    serialization_failures,
    soundness_failures,
    symmetry_failures,
    uniqueness_failures,
)


class TestFactoryTable(unittest.TestCase):
    def test_every_public_grasp_factory_is_in_the_matrix(self):
        self.assertEqual(set(FACTORIES), public_grasp_factories())

    def test_combined_entry_points_list_existing_parts(self):
        for spec in FACTORIES.values():
            for part in spec.parts:
                self.assertEqual(FACTORIES[part].primitive, spec.primitive)


class TestOracleWitness(unittest.TestCase):
    """(1) Every returned pose has an analytic oracle witness."""

    @matrix_settings(300)
    @given(case=any_cases(), seed=st.integers(0, 2**32 - 1))
    def test_every_pose_has_a_witness(self, case, seed):
        templates = case.run()
        self.assertEqual(soundness_failures(case, templates, np.random.default_rng(seed)), [], case)


class TestEquivariance(unittest.TestCase):
    """(3) Templates are equivariant under an arbitrary reference transform."""

    @matrix_settings(150)
    @given(case=any_cases(), T=transforms())
    def test_reference_transform_maps_poses_and_keeps_the_witness(self, case, T):
        self.assertEqual(equivariance_failures(case, case.run(), T), [], case)


class TestSymmetry(unittest.TestCase):
    """(4) Primitive and opposite-face symmetries."""

    @matrix_settings(150)
    @given(case=any_cases(), theta=st.floats(-np.pi, np.pi))
    def test_symmetries_preserve_the_witness(self, case, theta):
        self.assertEqual(symmetry_failures(case, case.run(), theta), [], case)

    @matrix_settings(150)
    @given(case=any_cases())
    def test_mirror_families_have_equal_depths(self, case):
        self.assertEqual(mirror_family_failures(case, case.run()), [], case)


class TestSerialization(unittest.TestCase):
    """(7) Serialization and instantiation preserve the witness."""

    @matrix_settings(100)
    @given(case=any_cases())
    def test_round_trips_preserve_poses_and_witness(self, case):
        self.assertEqual(serialization_failures(case, case.run()), [], case)


class TestArguments(unittest.TestCase):
    """(2) Invalid arguments raise; physical infeasibility returns []."""

    @matrix_settings(200)
    @given(case=any_cases(), which=st.integers(0, 99))
    def test_invalid_arguments_raise_value_error(self, case, which):
        overrides = invalid_overrides_for(case.spec)
        name = sorted(overrides)[which % len(overrides)]
        kwargs = apply_override(case, overrides[name])
        with self.assertRaises(ValueError, msg=f"{case.factory} {name} {kwargs}"):
            getattr(case.gripper.build(), case.factory)(**kwargs)

    @matrix_settings(150)
    @given(case=grasp_cases())
    def test_oversize_object_returns_empty_without_raising(self, case):
        # No jaw opening within the aperture can straddle these dimensions.
        oversize = case.replace(dims=oversize_dims(case.spec.primitive, case.gripper.max_aperture))
        self.assertEqual(oversize.run(preshape=None), [], oversize)


class TestCoverageAndUniqueness(unittest.TestCase):
    """(6) Declared coverage, structured uniqueness, combined == parts."""

    @matrix_settings(200)
    @given(case=any_cases())
    def test_structured_keys_are_unique(self, case):
        self.assertEqual(uniqueness_failures(case, case.run()), [], case)

    @matrix_settings(200)
    @given(case=any_cases())
    def test_combined_entry_points_equal_their_parts(self, case):
        self.assertEqual(combined_failures(case, case.run()), [], case)

    @matrix_settings(150)
    @given(case=grasp_cases(), k=st.integers(1, 4))
    def test_declared_families_are_all_emitted(self, case, k):
        # A comfortably feasible regime: a cube-ish object well inside reach and aperture.
        A, L = case.gripper.max_aperture, case.gripper.finger_length
        size = min(A / 3.0, L / 2.0)
        clearance = size / 50
        # A linkage hand's fingertips reach further once closed (#193), and a radial grasp
        # keeps them inside the far side: comfortable means a radius beyond that gain. The
        # torus ring is then as small as keeps it a torus, so its span still fits.
        g = case.gripper.build()
        gain = max(g.reach(w) for w in np.linspace(0.0, A, 65)) - L
        r = max(size / 2, gain + 2 * clearance)
        tube = max(size / 4, gain + 2 * clearance)
        dims = {
            "sphere": {"object_radius": r},
            "cylinder": {"cylinder_radius": r, "cylinder_height": 2 * size},
            "box": {"box_x": size, "box_y": size, "box_z": size},
            "torus": {"torus_radius": size if gain == 0.0 else 1.25 * tube, "tube_radius": tube},
        }[case.spec.primitive]
        comfortable = case.replace(dims=dims, options={"k": k, "clearance": clearance})
        self.assertEqual(coverage_failures(comfortable, comfortable.run()), [], comfortable)


def _shift(T, dx=1e-6):
    """A copy of transform ``T`` translated along x (a minimal geometry change)."""
    out = T.copy()
    out[0, 3] += dx
    return out


class TestCombinedEqualityIsExact(unittest.TestCase):
    """(6) The combined == parts check compares the WHOLE template (#127)."""

    CASES = [
        GraspCase(GripperSpec("ParallelJawGripper", 0.08, 0.2), f, dict(dims), {"k": 2, "clearance": 0.004})
        for f, dims in (
            ("grasp_cylinder", {"cylinder_radius": 0.02, "cylinder_height": 0.12}),
            ("grasp_box", {"box_x": 0.05, "box_y": 0.04, "box_z": 0.06}),
            ("grasp_torus", {"torus_radius": 0.03, "tube_radius": 0.01}),
        )
    ]

    def test_combined_entry_points_match_their_parts_exactly(self):
        for case in self.CASES:
            self.assertEqual(combined_failures(case, case.run()), [], case.factory)

    def test_any_altered_field_is_rejected(self):
        # One mutation per public field group; each must be caught.
        mutations = {
            "Tw_e": lambda t: _respawn(t, Tw_e=_shift(t.Tw_e)),
            "T_ref_tsr": lambda t: _respawn(t, T_ref_tsr=_shift(t.T_ref_tsr)),
            "Bw": lambda t: _respawn(t, Bw=t.Bw + 1e-6),
            "preshape": lambda t: _respawn(t, preshape=t.preshape * 1.01),
            "subject": lambda t: _respawn(t, subject=t.subject + "_x"),
            "reference": lambda t: _respawn(t, reference=t.reference + "_x"),
            "name": lambda t: _respawn(t, name=t.name + " (edited)"),
            "description": lambda t: _respawn(t, description=t.description + "!"),
            "variant": lambda t: _respawn(t, variant="mutant"),
            "task": lambda t: _respawn(t, task=t.task + "_x"),
            "stability_margin": lambda t: _respawn(t, stability_margin=0.5),
        }
        for case in self.CASES:
            templates = case.run()
            for field, mutate in mutations.items():
                with self.subTest(factory=case.factory, field=field):
                    mutated = [mutate(t) for t in templates]
                    self.assertNotEqual(combined_failures(case, mutated), [], field)

    def test_provenance_change_is_rejected(self):
        for case in self.CASES:
            templates = case.run()
            mutated = [
                _respawn(t, provenance=replace(t.provenance, depth=t.provenance.depth + 1e-6)) for t in templates
            ]
            self.assertNotEqual(combined_failures(case, mutated), [], case.factory)

    def test_matching_stays_order_independent(self):
        for case in self.CASES:
            self.assertEqual(combined_failures(case, list(reversed(case.run()))), [], case.factory)


SCALES = (1e-3, 1.0, 1e3)  # small / ordinary / large


class TestBoundaryTransitions(unittest.TestCase):
    """(2) Documented feasibility boundaries are exact, active, and behavioural (#126).

    Soundness alone is satisfied vacuously by an empty result, so these assert the
    below/exact/above transition of the family each boundary governs.
    """

    def test_every_boundary_transitions_as_documented(self):
        failures = []
        for b in BOUNDARIES:
            for factory in b.factories:
                for scale in SCALES:
                    failures += boundary_transition_failures(b, factory, scale)
        self.assertEqual(failures, [])

    def test_boundaries_cover_the_short_cylinder_and_straddle_floor(self):
        names = {b.name for b in BOUNDARIES}
        self.assertIn("cap_band_height_half", names)  # #122
        self.assertIn("cylinder_side_height_half", names)  # #124
        self.assertIn("straddle_floor", names)  # #107/#129
        cap = next(b for b in BOUNDARIES if b.name == "cap_band_height_half")
        for factory in ("grasp_cylinder_top", "grasp_cylinder_bottom", "grasp_cylinder"):
            self.assertIn(factory, cap.factories)

    def test_each_boundary_is_active_for_its_factories(self):
        # "Active" = the family really is feasible on one side and empty on the other,
        # so no unrelated constraint masks the boundary under test.
        for b in BOUNDARIES:
            for factory in b.factories:
                triplet = boundary_triplet(b, factory, 1.0)
                self.assertEqual([expect for _, _, expect in triplet].count(True), 2, (b.name, factory))

    def test_random_boundary_clearances_are_factory_relevant(self):
        # A cap formula must not be offered for a side-only factory, and vice versa.
        L, A = 0.08, 0.2
        side = FACTORIES["grasp_cylinder_side"]
        caps = FACTORIES["grasp_cylinder_top"]
        dims = {"cylinder_radius": 0.02, "cylinder_height": 0.03}
        self.assertIn(dims["cylinder_height"] / 2, boundary_clearances(side, dims, L, A))
        self.assertIn(min(L, dims["cylinder_height"]) / 2, boundary_clearances(caps, dims, L, A))
        self.assertNotIn(L - dims["cylinder_radius"], boundary_clearances(caps, dims, L, A))


def _strict_usable_depths(lo, hi, k):
    """`_usable_depths` with the CLOSED interval made open: equality returns empty."""
    if hi <= lo:
        return None
    return np.unique(np.linspace(lo, hi, k))


class TestInclusiveBoundaryMutant(unittest.TestCase):
    """A closed interval turned open must be caught (the #124 bug class)."""

    BAND_BOUNDARIES = (
        "cap_band_height_half",
        "radial_reach",
        "radial_far_surface",
        "box_approach_band",
        "torus_span_reach",
    )

    def test_strict_depth_band_is_detected(self):
        caught = set()
        with mock.patch.object(GripperBase, "_usable_depths", staticmethod(_strict_usable_depths)):
            for b in BOUNDARIES:
                for factory in b.factories:
                    if boundary_transition_failures(b, factory, 1.0):
                        caught.add(b.name)
        for name in self.BAND_BOUNDARIES:
            self.assertIn(name, caught, f"{name} did not detect an exclusive depth band")


class TestMonotonicity(unittest.TestCase):
    """(5) More reach or a wider aperture never removes a feasible family."""

    @matrix_settings(150)
    @given(
        case=grasp_cases(named=False, explicit_clearance=True), attr=st.sampled_from(["finger_length", "max_aperture"])
    )
    def test_growing_the_gripper_keeps_every_family(self, case, attr):
        self.assertEqual(monotonicity_failures(case, attr), [], case)


if __name__ == "__main__":
    unittest.main()
