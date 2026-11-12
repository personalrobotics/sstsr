# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Native provenance conformance for the built-in parallel-jaw generators.

`GraspProvenance` (tsr.grasp_provenance) is a generic, extensible value object: it
validates representation invariants only, so third-party generators can use their
own vocabulary. This module owns the *sstsr-specific* vocabulary and the
relational checks that a generic value object should not — which
``(primitive, mode)`` pairs the built-ins emit, their allowed approach /
finger-orientation / symmetry labels, the metadata each requires, and the
cross-field consistency the closed value object deliberately does not police
(issue #84).

Built-in factory tests call :func:`validate_builtin_provenance` on every emitted
template. This layer may evolve with the built-in generators without closing the
public provenance type. It never inspects the pose — geometric truth is the
oracle's job (#67); this only checks that a built-in *described* its output with a
self-consistent, in-vocabulary claim.
"""

from __future__ import annotations

from tsr.grasp_provenance import GraspProvenance


def _positive_number(v: object) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and v > 0


def _nonneg_int(v: object) -> bool:
    return not isinstance(v, bool) and isinstance(v, int) and v >= 0


# Native (primitive, mode) vocabulary the built-in generators emit: the allowed
# approach / finger_orientation / symmetry labels and the required metadata keys
# (name -> predicate). Relational checks live in validate_builtin_provenance.
_NATIVE = {
    ("cylinder", "side"): dict(
        approaches={"radial"}, orientations={"tangential"}, symmetries={"roll0", "rollpi"}, metadata={}
    ),
    ("cylinder", "top"): dict(approaches={"+z"}, orientations={"diameter"}, symmetries={""}, metadata={}),
    ("cylinder", "bottom"): dict(approaches={"-z"}, orientations={"diameter"}, symmetries={""}, metadata={}),
    ("box", "top"): dict(
        approaches={"+z"},
        orientations={"x", "y"},
        symmetries={""},
        metadata={"slide_axis": lambda v: v in ("x", "y", "z"), "span": _positive_number},
    ),
    ("box", "bottom"): dict(
        approaches={"-z"},
        orientations={"x", "y"},
        symmetries={""},
        metadata={"slide_axis": lambda v: v in ("x", "y", "z"), "span": _positive_number},
    ),
    ("box", "face"): dict(
        approaches={"+x", "-x", "+y", "-y"},
        orientations={"x", "y", "z"},
        symmetries={""},
        metadata={"slide_axis": lambda v: v in ("x", "y", "z"), "span": _positive_number},
    ),
    ("sphere", "surface"): dict(approaches={"radial"}, orientations={"diameter"}, symmetries={""}, metadata={}),
    ("torus", "side"): dict(
        approaches={"tube"},
        orientations={"tangential"},
        symmetries={"flip0", "flippi"},
        metadata={
            "minor_index": _nonneg_int,
            "minor_count": _nonneg_int,
            "minor_angle": lambda v: not isinstance(v, bool) and isinstance(v, (int, float)),
        },
    ),
    ("torus", "span"): dict(approaches={"+z", "-z"}, orientations={"diameter"}, symmetries={""}, metadata={}),
}

# How a hand with distinct gripping surfaces holds the object (#189). Any built-in grasp
# can be held on the pads; only a curved object deep in the hand can be encompassed, and
# only a cylinder side grasp enveloped (#204).
_GRIPS = {"", "pad"}
_ENCOMPASSABLE = {("cylinder", "side"), ("sphere", "surface")}
_ENVELOPABLE = {("cylinder", "side")}


def validate_builtin_provenance(p: GraspProvenance) -> None:
    """Raise ``ValueError`` if ``p`` is not a self-consistent built-in record.

    Checks native vocabulary, required metadata, and cross-field relationships
    the generic value object intentionally does not own. Does not inspect any
    pose.
    """
    key = (p.primitive, p.mode)
    spec = _NATIVE.get(key)
    if spec is None:
        raise ValueError(f"{key} is not a built-in (primitive, mode); native modes: {sorted(_NATIVE)}")

    for label, allowed in (
        ("approach", spec["approaches"]),
        ("finger_orientation", spec["orientations"]),
        ("symmetry", spec["symmetries"]),
    ):
        value = getattr(p, label)
        if value not in allowed:
            raise ValueError(f"{label}={value!r} is not valid for built-in {key}; allowed {sorted(allowed)}")

    allowed_grips = (
        _GRIPS
        | ({"encompassing"} if key in _ENCOMPASSABLE else set())
        | ({"enveloping"} if key in _ENVELOPABLE else set())
    )
    if p.grip not in allowed_grips:
        raise ValueError(f"grip={p.grip!r} is not valid for built-in {key}; allowed {sorted(allowed_grips)}")

    for name, predicate in spec["metadata"].items():
        if name not in p.metadata:
            raise ValueError(f"built-in {key} requires metadata[{name!r}]")
        if not predicate(p.metadata[name]):
            raise ValueError(f"metadata[{name!r}]={p.metadata[name]!r} fails the requirement for built-in {key}")

    # Relational checks the generic value object does not own.
    if p.primitive == "box":
        if p.finger_orientation == p.metadata["slide_axis"]:
            raise ValueError(
                f"box {p.mode}: finger_orientation must differ from slide_axis (both {p.finger_orientation!r})"
            )
        if p.mode == "face":
            face_normal = p.approach[-1]  # "+x" -> "x"
            if p.finger_orientation == face_normal:
                raise ValueError(
                    f"box face: finger_orientation {p.finger_orientation!r} is the face normal, not in-plane"
                )
    if key == ("torus", "side"):
        idx, count = p.metadata["minor_index"], p.metadata["minor_count"]
        if not (0 <= idx < count):
            raise ValueError(f"torus side: minor_index {idx} out of range for minor_count {count}")
