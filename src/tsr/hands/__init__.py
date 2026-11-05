# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""tsr.hands — Gripper hand models with TSR template generation.

Usage::

    from tsr.hands import ParallelJawGripper, Robotiq2F140

    gripper   = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)
    templates = gripper.grasp_cylinder(cylinder_radius=0.04,
                                       cylinder_height=0.12,
                                       reference="mug")
    tsr  = templates[0].instantiate(mug_pose)
    pose = tsr.sample()
"""

from .base import GripperBase
from .parallel_jaw import FrankaHand, KinovaJaco2TwoFinger, ParallelJawGripper, Robotiq2F85, Robotiq2F140
from .registry import HandRegistry, default_registry

__all__ = [
    "GripperBase",
    "ParallelJawGripper",
    "Robotiq2F85",
    "Robotiq2F140",
    "FrankaHand",
    "KinovaJaco2TwoFinger",
    "HandRegistry",
    "default_registry",
]
