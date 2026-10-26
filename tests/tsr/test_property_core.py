# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Property-based tests for the pure-math core (TSR, TSRChain, utils).

These encode the core consistency contract from docs/ARCHITECTURE.md:
sample() ⇒ is_valid ⇒ contains ⇒ distance≈0, plus the SE(3) round-trips that
the C1 gimbal-lock fix made accurate everywhere.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tsr import TSR, TSRChain, wrap_to_interval
from tsr.core.utils import CHAIN_TOLERANCE, geodesic_distance, rotation_angle

from ._hypothesis_strategies import angles, any_rotation, near_gimbal_rotations, rotations, transforms, tsrs

# --------------------------------------------------------------------------
# SE(3) conversions
#
# Away from gimbal lock the round-trip is exact to ~1e-9 (the C1 fix). Within
# ~1e-7 rad of pitch=±90° the rotation matrix is treated as exact gimbal lock and
# one roll/yaw DOF is discarded, which costs at most ~cos(pitch)·π of rotation —
# under the ANGLE_ATOL=1e-6 that containment absorbs as round-off (#190).
# --------------------------------------------------------------------------


@given(R=rotations())
def test_rpy_rot_roundtrip_generic(R):
    R2 = TSR.rpy_to_rot(TSR.rot_to_rpy(R))
    np.testing.assert_allclose(R2, R, atol=1e-9)


@given(R=near_gimbal_rotations())
def test_rpy_rot_roundtrip_near_gimbal(R):
    R2 = TSR.rpy_to_rot(TSR.rot_to_rpy(R))
    np.testing.assert_allclose(R2, R, atol=1e-4)


@given(T=transforms())
def test_xyzrpy_trans_roundtrip(T):
    T2 = TSR.xyzrpy_to_trans(TSR.trans_to_xyzrpy(T))
    np.testing.assert_allclose(T2, T, atol=1e-4)


@given(R=any_rotation())
def test_rpy_to_rot_is_a_proper_rotation(R):
    R2 = TSR.rpy_to_rot(TSR.rot_to_rpy(R))
    np.testing.assert_allclose(R2 @ R2.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R2), 1.0, atol=1e-9)


# --------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------


@given(x=st.floats(-100, 100, allow_nan=False), lo=angles)
def test_wrap_to_interval_in_range_and_idempotent(x, lo):
    lower = np.array([lo])
    w = wrap_to_interval(np.array([x]), lower=lower)
    assert lower[0] - 1e-9 <= w[0] < lower[0] + 2 * np.pi + 1e-9
    np.testing.assert_allclose(wrap_to_interval(w, lower=lower), w, atol=1e-9)


@given(x=st.floats(-100, 100, allow_nan=False), lo=angles)
def test_wrap_to_interval_preserves_angle_mod_2pi(x, lo):
    w = wrap_to_interval(np.array([x]), lower=np.array([lo]))
    diff = (x - w[0]) / (2 * np.pi)
    np.testing.assert_allclose(diff, np.round(diff), atol=1e-9)


@given(R=any_rotation())
def test_rotation_angle_in_0_pi(R):
    a = rotation_angle(R)
    assert -1e-9 <= a <= np.pi + 1e-9


@given(A=transforms(), B=transforms())
def test_geodesic_distance_symmetry_and_nonneg(A, B):
    d_ab = geodesic_distance(A, B)
    d_ba = geodesic_distance(B, A)
    assert d_ab >= -1e-12
    np.testing.assert_allclose(d_ab, d_ba, atol=1e-9)


@given(A=transforms())
def test_geodesic_distance_zero_on_identity(A):
    # arccos((trace-1)/2) is ill-conditioned at angle 0 (slope -> inf), so a
    # ~1e-15 orthonormality error in A maps to a ~1e-7 angle. That's expected.
    np.testing.assert_allclose(geodesic_distance(A, A), 0.0, atol=1e-6)


# --------------------------------------------------------------------------
# Core consistency contract
# --------------------------------------------------------------------------


@given(tsr=tsrs(), data=st.data())
def test_sample_is_valid_and_contained(tsr, data):
    # A pose drawn from the TSR must satisfy is_valid, contains, and distance≈0.
    xyzrpy = tsr.sample_xyzrpy()
    assert all(tsr.is_valid(xyzrpy))
    pose = tsr.to_transform(xyzrpy)
    assert tsr.contains(pose)
    dist, _ = tsr.distance(pose)
    assert dist == 0.0


@given(tsr=tsrs(), T=transforms())
def test_contains_iff_distance_zero(tsr, T):
    # contains() and distance() must agree for arbitrary probe poses.
    contained = tsr.contains(T)
    dist, _ = tsr.distance(T)
    if contained:
        assert dist == 0.0
    else:
        assert dist > 0.0


@given(tsr=tsrs())
def test_to_xyzrpy_of_a_contained_pose_is_itself_in_bounds(tsr):
    # A region must not contradict itself: whatever to_xyzrpy names as the coordinates of a
    # pose the region contains has to be coordinates the region accepts, and recomposing them
    # has to give the pose back. RPY double-covers rotations -- (r, p, y) and
    # (r ± π, −π − p, y ± π) are the same rotation -- so "some representative is in bounds" and
    # "the one we return is in bounds" are different claims, and contains() only establishes
    # the first. rot_to_rpy always reports |pitch| <= π/2, so a region whose pitch bounds lie
    # outside that range is named by the other representative (#171).
    #
    # The pose comes from the region rather than from transforms(), because a random transform
    # is essentially never inside a random region.
    pose = tsr.sample()
    xyzrpy = tsr.to_xyzrpy(pose)
    assert all(tsr.is_valid(xyzrpy)), f"to_xyzrpy returned {xyzrpy}, which this TSR's is_valid rejects"
    assert geodesic_distance(tsr.to_transform(xyzrpy), pose) < CHAIN_TOLERANCE


@given(tsr=tsrs())
def test_tsr_dict_json_yaml_roundtrip(tsr):
    for revived in (
        TSR.from_dict(tsr.to_dict()),
        TSR.from_json(tsr.to_json()),
        TSR.from_yaml(tsr.to_yaml()),
    ):
        np.testing.assert_allclose(revived.T0_w, tsr.T0_w)
        np.testing.assert_allclose(revived.Tw_e, tsr.Tw_e)
        np.testing.assert_allclose(revived.Bw, tsr.Bw)


# --------------------------------------------------------------------------
# TSRChain
# --------------------------------------------------------------------------


@given(tsr=tsrs())
def test_single_chain_sample_is_contained(tsr):
    chain = TSRChain(TSRs=[tsr])
    pose = chain.sample()
    assert chain.contains(pose)


# Multi-TSR chains, non-identity frames and mixed fixed/free coords (#57, #85).
# A constructive sample carries a coordinate witness, so membership is decided by
# the EXACT forward/witness contract -- not by a cold inverse solve, which cannot
# prove inverse membership for a rotation-rich chain (#85). We assert the witness
# validates, the warm solve is satisfied, and the warm `contains` fast path holds.
def _compose_chain_independently(parts, coordinates):
    """Serial chain composition (Berenson et al. 2011 §5.1) that does NOT call
    TSRChain.to_transform, so it is an independent oracle for the witness contract
    (only the first component's T0_w participates, matching to_transform)."""
    T = np.array(parts[0].T0_w, dtype=float)
    for tsr, c in zip(parts, coordinates):
        T = T @ TSR.xyzrpy_to_trans(np.asarray(c, dtype=float)) @ tsr.Tw_e
    return T


@settings(max_examples=40)
@given(parts=st.lists(tsrs(), min_size=2, max_size=3))
def test_multi_chain_sample_is_contained(parts):
    # Only the first link carries a T0_w: a later link is positioned by the chain, and
    # passing one it will not read is refused rather than ignored (#166). The strategy
    # draws a frame for every link, so the later ones are reset here.
    parts = [parts[0]] + [TSR(T0_w=np.eye(4), Tw_e=p.Tw_e, Bw=p.Bw) for p in parts[1:]]
    chain = TSRChain(TSRs=parts)
    sample = chain.sample_with_witness()
    assert sample.coordinates.shape == (len(parts), 6)
    # The witness recomposes the pose exactly under an INDEPENDENT oracle (not
    # to_transform), so this also guards to_transform's wrapping-interval
    # canonicalization (#87): the bw() strategy draws wrapping intervals.
    expected = _compose_chain_independently(parts, sample.coordinates)
    np.testing.assert_allclose(sample.pose, expected, atol=1e-9)
    # Exact positive certificate: no optimizer, no SciPy.
    assert chain.validate_witness(sample.pose, sample.coordinates)
    # Warm-started solve takes the fast path and reports a satisfying witness.
    result = chain.solve(sample.pose, initial_guess=sample.coordinates)
    assert result.status == "satisfied"
    assert result.residual < CHAIN_TOLERANCE
    assert chain.contains(sample.pose, initial_guess=sample.coordinates)


@given(tsr=tsrs())
def test_chain_dict_roundtrip(tsr):
    chain = TSRChain(TSRs=[tsr])
    revived = TSRChain.from_dict(chain.to_dict())
    assert len(revived.TSRs) == 1
    np.testing.assert_allclose(revived.TSRs[0].Bw, tsr.Bw)
