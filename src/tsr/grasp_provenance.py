# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Structured, machine-readable provenance for generated grasp templates.

Clause 8 of the geometric grasp contract (docs/ARCHITECTURE.md, issue #66)
requires that every template a factory returns declares its grasp mode
*structurally* — tests and oracles read this record, never parse the
human-readable ``name``.

**Provenance declares intent; the generated pose determines geometric truth**
(issue #84). Three layers are kept separate:

1. :class:`GraspProvenance` — a small, immutable, **extensible** value object
   recording the generator's claim. It validates only *representation*
   invariants (below); it does not know which ``(primitive, mode)`` pairs exist
   or how axes/faces/angles relate. Labels are free strings, so third-party grasp
   generators can use their own vocabulary without modifying sstsr.
2. Native conformance (``tsr.hands._conformance.validate_builtin_provenance``)
   owns the sstsr-specific vocabulary and relational checks; the built-in factory
   tests run it over every emitted template.
3. The analytic oracle (#67) derives contacts/reach/clearance from the concrete
   pose and primitive geometry, and must *not* trust provenance values as evidence
   the pose is correct.

Field frames and meanings (see docs/ARCHITECTURE.md): ``depth`` is the insertion
depth from the approached surface [m]; ``approach`` is the object-relative side
the hand occupies; ``finger_orientation`` is the object-relative direction the
pads are separated along (fingers always close along ``±y_EE``, a library
invariant); ``depth_count`` is the number of **distinct** depth levels emitted for
a *depth family* (at most the requested ``k``; fewer when the feasible band
collapses), and ``depth_index`` in ``0..depth_count-1`` orders them shallow to deep.
``grip`` is how a hand with distinct gripping surfaces holds the object (#189):
``"pad"`` with the contacts on the fingertip pads, ``"encompassing"`` with the object
deep in the hand and the fingers folding around it, ``"enveloping"`` with the palm a
clearance from the object and the fingers folding past its far side (#204); empty for an
ideal parallel jaw.
A depth family is the set of templates **within one factory result** whose non-depth
provenance fields are identical: ``primitive``, ``mode``, ``approach``,
``finger_orientation``, ``symmetry``, ``grip``, and ``metadata`` are held fixed while only
``depth``, ``depth_index``, and ``depth_count`` vary, so a family holds exactly one
template per depth level. The counters are not a global identifier across
independently generated or concatenated collections;
``symmetry`` distinguishes otherwise-equivalent templates; ``metadata`` is
descriptive only and must never be used as geometric evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from ._provenance_common import JSONScalar, freeze_metadata, require_finite_real, require_int


@dataclass(frozen=True)
class GraspProvenance:
    """The generator's structured claim about a grasp template.

    Validates only representation invariants — nonempty string labels, a finite
    nonnegative ``depth``, exact integer depth counters, and JSON-scalar
    ``metadata`` frozen into an immutable mapping — so the record round-trips
    exactly. It intentionally does **not** validate that a ``(primitive, mode)``
    is one sstsr supports or that fields are mutually consistent; that is native
    conformance (``tsr.hands._conformance``), which can evolve with the built-in
    generators without closing this type to external use.
    """

    #: Wire discriminator. Unannotated on purpose: a bare ``KIND: str = "grasp"`` would
    #: become a dataclass field and change ``__init__`` and ``__eq__``. Absent from a
    #: serialized block means "grasp", so every record sstsr has ever written still reads.
    KIND = "grasp"

    primitive: str
    mode: str
    approach: str
    finger_orientation: str
    depth: float
    depth_index: int
    depth_count: int
    symmetry: str = ""
    metadata: Mapping[str, JSONScalar] = field(default_factory=dict)
    grip: str = ""

    def __post_init__(self) -> None:
        for name in ("primitive", "mode", "approach", "finger_orientation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")
        for name in ("symmetry", "grip"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string, got {getattr(self, name)!r}")

        require_int(self.depth_index, "depth_index")
        require_int(self.depth_count, "depth_count")
        if self.depth_count < 1 or not (0 <= self.depth_index < self.depth_count):
            raise ValueError(f"depth_index {self.depth_index} out of range for depth_count {self.depth_count}")

        # canonicalize -> lossless round-trip (see require_finite_real on NumPy scalars)
        object.__setattr__(self, "depth", require_finite_real(self.depth, "depth", minimum=0.0))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict of JSON/YAML scalars (lossless)."""
        d: Dict[str, Any] = {
            "kind": self.KIND,
            "primitive": self.primitive,
            "mode": self.mode,
            "approach": self.approach,
            "finger_orientation": self.finger_orientation,
            "depth": self.depth,
            "depth_index": self.depth_index,
            "depth_count": self.depth_count,
        }
        if self.symmetry:
            d["symmetry"] = self.symmetry
        if self.grip:
            d["grip"] = self.grip
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d

    @staticmethod
    def from_dict(x: Mapping[str, Any]) -> "GraspProvenance":
        """Reconstruct from :meth:`to_dict` output, validating (not narrowing).

        Bracket access on the required keys on purpose: a ``.get()`` with a default is
        the one route by which a placement block could be misread as a grasp block.
        """
        kind = x.get("kind", GraspProvenance.KIND)
        if kind != GraspProvenance.KIND:
            raise ValueError(f"not a grasp provenance block: kind is {kind!r}")
        return GraspProvenance(
            primitive=x["primitive"],
            mode=x["mode"],
            approach=x["approach"],
            finger_orientation=x["finger_orientation"],
            depth=x["depth"],
            depth_index=x["depth_index"],
            depth_count=x["depth_count"],
            symmetry=x.get("symmetry", ""),
            metadata=dict(x.get("metadata", {})),
            grip=x.get("grip", ""),
        )
