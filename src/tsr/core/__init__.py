# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""tsr.core — the pure-math heart of the package.

Layer 0 of the architecture: robot-agnostic SE(3) geometry with no dependencies
beyond NumPy (and SciPy for the optional optimiser paths). Nothing in this
subpackage imports from the convenience layers above it (templates, factories,
io, sampling, viz); that invariant is enforced by
``tests/tsr/test_architecture.py``.

The public symbols are re-exported at the top level (``from tsr import TSR``);
import them from here when you want only the kernel.
"""

from .tsr import FRAME_ATOL, NANBW, TSR
from .tsr_chain import ChainSample, ChainSolveResult, TSRChain
from .utils import (
    ANGLE_ATOL,
    CHAIN_TOLERANCE,
    EPSILON,
    geodesic_distance,
    geodesic_error,
    rotation_angle,
    wrap_to_interval,
)

__all__ = [
    "TSR",
    "TSRChain",
    "ChainSample",
    "ChainSolveResult",
    "NANBW",
    "FRAME_ATOL",
    "ANGLE_ATOL",
    "CHAIN_TOLERANCE",
    "EPSILON",
    "wrap_to_interval",
    "rotation_angle",
    "geodesic_error",
    "geodesic_distance",
]
