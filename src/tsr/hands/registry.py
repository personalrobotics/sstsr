# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""HandRegistry: plain-string keyed registry for gripper template generators."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from tsr.template import TSRTemplate

_Key = Tuple[str, str, str]  # (hand, reference, task)
_Generator = Callable[..., List[TSRTemplate]]


class HandRegistry:
    """Registry mapping (hand, reference, task) string triples to generators.

    A generator is a callable ``(gripper: GripperBase, **geometry) -> List[TSRTemplate]``.

    Usage::

        registry = HandRegistry()

        @registry.register("parallel_jaw", "cylinder", "grasp")
        def _gen(gripper, **kw):
            return gripper.grasp_cylinder(**kw)

        gen      = registry.get("parallel_jaw", "cylinder", "grasp")
        gripper  = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)
        templates = gen(gripper, object_radius=0.04, height_range=(0.02, 0.10))

        registry.list_tasks()
        # [("parallel_jaw", "cylinder", "grasp"), ...]
    """

    def __init__(self) -> None:
        self._registry: Dict[_Key, _Generator] = {}

    def register(self, hand: str, reference: str, task: str):
        """Decorator / direct call to register a generator.

        Can be used as a decorator::

            @registry.register("parallel_jaw", "cylinder", "grasp")
            def _gen(gripper, **kw): return gripper.grasp_cylinder(**kw)

        Or called directly::

            registry.register("parallel_jaw", "cylinder", "grasp")(fn)
        """

        def decorator(fn: _Generator) -> _Generator:
            self._registry[(hand, reference, task)] = fn
            return fn

        return decorator

    def get(self, hand: str, reference: str, task: str) -> _Generator:
        """Return the generator for (hand, reference, task).

        Raises:
            KeyError: with a list of available keys if not registered.
        """
        key = (hand, reference, task)
        if key not in self._registry:
            raise KeyError(f"No generator registered for {key!r}. Available: {sorted(self._registry)}")
        return self._registry[key]

    def list_tasks(self) -> List[_Key]:
        """Return all registered (hand, reference, task) triples, sorted."""
        return sorted(self._registry)

    def __contains__(self, key: _Key) -> bool:
        return key in self._registry


# Module-level registry pre-populated with built-in hands.
default_registry = HandRegistry()

default_registry.register("parallel_jaw", "cylinder", "grasp")(lambda gripper, **kw: gripper.grasp_cylinder(**kw))
default_registry.register("robotiq_2f140", "cylinder", "grasp")(lambda gripper, **kw: gripper.grasp_cylinder(**kw))
default_registry.register("robotiq_2f85", "cylinder", "grasp")(lambda gripper, **kw: gripper.grasp_cylinder(**kw))
# The JACO2 is registered for cylinders only: they are what its grasps were closed on in
# its model (tools/grasp_closure.py). Its box, sphere and torus grasps are unverified.
default_registry.register("kinova_jaco2_two_finger", "cylinder", "grasp")(
    lambda gripper, **kw: gripper.grasp_cylinder(**kw)
)
default_registry.register("parallel_jaw", "box", "grasp")(lambda gripper, **kw: gripper.grasp_box(**kw))
default_registry.register("robotiq_2f140", "box", "grasp")(lambda gripper, **kw: gripper.grasp_box(**kw))
default_registry.register("robotiq_2f85", "box", "grasp")(lambda gripper, **kw: gripper.grasp_box(**kw))
default_registry.register("parallel_jaw", "sphere", "grasp")(lambda gripper, **kw: gripper.grasp_sphere(**kw))
default_registry.register("robotiq_2f140", "sphere", "grasp")(lambda gripper, **kw: gripper.grasp_sphere(**kw))
default_registry.register("robotiq_2f85", "sphere", "grasp")(lambda gripper, **kw: gripper.grasp_sphere(**kw))
default_registry.register("parallel_jaw", "torus", "grasp")(lambda gripper, **kw: gripper.grasp_torus(**kw))
default_registry.register("robotiq_2f140", "torus", "grasp")(lambda gripper, **kw: gripper.grasp_torus(**kw))
default_registry.register("robotiq_2f85", "torus", "grasp")(lambda gripper, **kw: gripper.grasp_torus(**kw))
