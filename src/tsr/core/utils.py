# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

import numpy as np
from numpy import pi

#: Default satisfaction tolerance of a chain's inverse solve: the geodesic residual
#: below which :meth:`TSRChain.solve` reports ``"satisfied"``. A solver tolerance,
#: not part of any region's geometry -- :meth:`TSR.contains` does not use it (#190).
CHAIN_TOLERANCE = 0.001

#: Deprecated alias of :data:`CHAIN_TOLERANCE`. Until #190 it was also the slack
#: :meth:`TSR.contains` granted on every side of every bound; it no longer is.
EPSILON = CHAIN_TOLERANCE

#: Round-off tolerance for an angle [rad], the contract's angular counterpart of
#: :func:`length_atol` (docs/ARCHITECTURE.md). Membership absorbs round-off with it.
ANGLE_ATOL = 1e-6


def length_atol(scale: float) -> float:
    """Scale-aware length tolerance from the geometric contract (docs/ARCHITECTURE.md).

    ``atol = 1e-9 + 1e-6 · scale``, where ``scale`` is the object's characteristic
    dimension. This is the canonical definition: grasp feasibility, placement
    containment and the analytic oracles all measure lengths against it, so a
    millimetre-scale object and a metre-scale one get the same relative slack.
    """
    return 1e-9 + 1e-6 * float(scale)


def wrap_to_interval(angles: np.ndarray, lower: np.ndarray = None) -> np.ndarray:
    """
    Wrap a vector of angles to a continuous interval starting at `lower`.

    Args:
        angles: (N,) array of angles (in radians)
        lower: (N,) array of lower bounds; defaults to -pi if None

    Guarantees the result lies in the half-open interval [lower, lower + 2*pi).

    Returns:
        wrapped: (N,) array of wrapped angles
    """
    if lower is None:
        lower = -pi * np.ones_like(angles)
    two_pi = 2 * pi
    frac = np.asarray((angles - lower) % two_pi, dtype=float)
    # Floating-point guard: for a tiny-negative (angles - lower), numpy's modulo
    # can round up to exactly 2*pi, which would push the result to lower + 2*pi
    # and out of the half-open interval. Map that boundary back to 0 (i.e. lower).
    frac = np.where(frac >= two_pi, 0.0, frac)
    return frac + lower


def rotation_angle(R: np.ndarray) -> float:
    """
    Compute the angle of rotation for a rotation matrix.

    The rotation angle θ satisfies: trace(R) = 1 + 2*cos(θ)
    Therefore: θ = arccos((trace(R) - 1) / 2)

    Args:
        R: 3x3 rotation matrix

    Returns:
        angle: rotation angle in radians [0, π]
    """
    # Compute trace and clamp to valid range for arccos
    trace = np.trace(R)
    # trace = 1 + 2*cos(θ), so cos(θ) = (trace - 1) / 2
    cos_angle = (trace - 1.0) / 2.0
    # Clamp to [-1, 1] to handle numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def geodesic_error(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """
    Compute the geodesic error between two transforms on SE(3).

    The error is computed as:
    - Translation error: the Euclidean distance between positions
    - Rotation error: the angle of the relative rotation R1^T * R2

    Args:
        t1: first transform (4x4)
        t2: second transform (4x4)

    Returns:
        error: 4-vector [dx, dy, dz, rotation_angle]
               where dx, dy, dz are in meters and rotation_angle is in radians
    """
    # Translation error (in world frame)
    trans_error = t2[0:3, 3] - t1[0:3, 3]

    # Rotation error: angle of R1^T * R2
    R1 = t1[0:3, 0:3]
    R2 = t2[0:3, 0:3]
    R_rel = np.dot(R1.T, R2)
    angle_error = rotation_angle(R_rel)

    return np.hstack((trans_error, angle_error))


def geodesic_distance(t1: np.ndarray, t2: np.ndarray, r: float = 1.0) -> float:
    """
    Compute the geodesic distance between two transforms on SE(3).

    The distance combines translation (meters) and rotation (radians) errors.
    The parameter `r` allows weighting rotation relative to translation.

    As noted in Section 4.2 of Berenson et al. 2011:
    "we implicitly weigh rotation in radians and translation in meters equally
    when computing ||Δx||, but the two types of units can be weighed in an
    arbitrary manner"

    Args:
        t1: first transform (4x4)
        t2: second transform (4x4)
        r: weight for rotation in units of meters/radian (default 1.0)
           Higher values penalize rotation errors more.

    Returns:
        distance: weighted geodesic distance
    """
    error = geodesic_error(t1, t2)
    # Weight the rotation component
    error[3] = r * error[3]
    return np.linalg.norm(error)
