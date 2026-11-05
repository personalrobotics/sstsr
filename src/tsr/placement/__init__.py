# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""tsr.placement — Stable placement TSR generator.

Usage::

    from tsr.placement import StablePlacer

    placer    = StablePlacer(table_x=0.3, table_y=0.2)
    templates = placer.place_cylinder(cylinder_radius=0.04,
                                      cylinder_height=0.12,
                                      subject="mug")
    tsr  = templates[0].instantiate(surface_pose)
    pose = tsr.sample()

An object described by its parts, in its own frame, is placed whatever they are::

    from tsr.placement import Box, Cylinder, Shape

    shape     = Shape([Cylinder(0.033, 0.123, pose=T_object_can)])
    templates = placer.place(shape)
"""

from .shape import Box, Capsule, Cylinder, Hull, Shape, Sphere
from .stable_placer import StablePlacer

__all__ = ["Box", "Capsule", "Cylinder", "Hull", "Shape", "Sphere", "StablePlacer"]
