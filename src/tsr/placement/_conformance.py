# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Native provenance conformance for the built-in placement generators.

:class:`~tsr.placement_provenance.PlacementProvenance` is a generic, extensible value
object: it validates representation invariants only, so a third-party placement
generator can use its own vocabulary. This module owns the *sstsr-specific* vocabulary
and the relational checks a generic value object should not — which primitives the
built-ins emit, that only a sphere may omit a face normal, that ``equilibrium`` agrees
with ``support_margin``, and that the template's ``stability_margin`` really is the angle
its two recorded components imply (issue #160).

It never inspects the pose. Geometric truth is the oracle's job
(``tests/tsr/placement/_placement_oracle.py``); this only checks that a built-in
*described* its output with a self-consistent, in-vocabulary claim.

One rule is deliberately **absent**: ``footprint_radius >= support_margin``. It holds for
all four primitives, but for an arbitrary mesh with an arbitrary frame origin it is not
provable — the footprint is measured from the frame origin while the support margin is
measured from the centre of mass's projection — and a conformance rule that is only
probably true becomes a spurious failure on a user's asset.
"""

from __future__ import annotations

import math

from ..placement_provenance import PlacementProvenance

#: The primitives the built-in factories emit. Spelled as in the grasp vocabulary so a
#: caller can key off ``primitive`` across both contracts.
_PRIMITIVES = frozenset({"cylinder", "box", "sphere", "torus", "mesh", "compound"})

#: ``"neutral"`` is a contact that cannot resist tipping at all but does not fall — a
#: sphere, which rolls. A mesh face with a vanishing margin is a knife-edge equilibrium
#: and is not returned at all, so it never reaches this vocabulary.
_EQUILIBRIA = frozenset({"stable", "neutral"})

#: Only a sphere has no distinguished resting face.
_FACELESS = frozenset({"sphere"})

#: Mesh faces are enumerated; the primitives' variants are geometric labels.
_ENUMERATED = frozenset({"mesh", "compound"})

#: ``variant`` labels that name an axis, and the object-frame normal each promises.
_AXIS_NORMALS = {
    "-x": (-1.0, 0.0, 0.0),
    "+x": (1.0, 0.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-z": (0.0, 0.0, -1.0),
    "+z": (0.0, 0.0, 1.0),
}

_UNIT_ATOL = 1e-9


def validate_builtin_provenance(p: PlacementProvenance) -> None:
    """Check one record against the built-in placement vocabulary. Raises ``ValueError``."""
    if p.primitive not in _PRIMITIVES:
        raise ValueError(f"primitive {p.primitive!r} is not one of {sorted(_PRIMITIVES)}")
    if p.equilibrium not in _EQUILIBRIA:
        raise ValueError(f"equilibrium {p.equilibrium!r} is not one of {sorted(_EQUILIBRIA)}")

    faceless = p.primitive in _FACELESS
    if faceless and p.face_normal is not None:
        raise ValueError(f"a {p.primitive} has no distinguished resting face, but face_normal is {p.face_normal}")
    if not faceless:
        if p.face_normal is None:
            raise ValueError(f"a {p.primitive} rests on a face, so face_normal is required")
        norm = math.sqrt(sum(c * c for c in p.face_normal))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_UNIT_ATOL):
            raise ValueError(f"face_normal must be a unit vector, got norm {norm!r}")

    # Exact, not tolerance-based: the record carries no scale, so there is no principled
    # tolerance to compare a length against. The generator derives this from the same
    # exact comparison, which is what makes the rule checkable at all.
    neutral = p.equilibrium == "neutral"
    if neutral != (p.support_margin == 0.0):
        raise ValueError(
            f"equilibrium {p.equilibrium!r} disagrees with support_margin {p.support_margin!r}: "
            "'neutral' means a zero support margin and nothing else does"
        )

    enumerated = p.primitive in _ENUMERATED
    if enumerated and p.face_index is None:
        raise ValueError(f"a {p.primitive} face is enumerated, so face_index/face_count/facet_count are required")
    if not enumerated and p.face_index is not None:
        raise ValueError(f"a {p.primitive} has no enumerated faces, but face_index is {p.face_index}")

    if p.com_height <= 0.0:
        raise ValueError(f"com_height must be positive for a resting object, got {p.com_height!r}")
    if p.footprint_radius <= 0.0:
        raise ValueError(f"footprint_radius must be positive for a real object, got {p.footprint_radius!r}")


def validate_builtin_placement_template(t) -> None:
    """Check a built-in placement template against its own record. Raises ``ValueError``.

    The record's relational partner is the template: the reported angle has to be the one
    the recorded components imply, and the ``variant`` label has to name the recorded
    face. Neither relation is visible from the record alone.
    """
    if t.task != "place":
        raise ValueError(f"expected a placement template, got task {t.task!r}")
    p = t.provenance
    if not isinstance(p, PlacementProvenance):
        raise ValueError(f"expected a PlacementProvenance, got {type(p).__name__}")
    validate_builtin_provenance(p)

    # Recomputed with math.atan2 rather than the generator's own helper on purpose: if
    # this called _tipping_angle, corrupting that helper would corrupt both sides of the
    # comparison and the check would silently always pass.
    implied = math.atan2(p.support_margin, p.com_height)
    if t.stability_margin is None:
        raise ValueError("a placement template must report a stability_margin")
    if not math.isclose(t.stability_margin, implied, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(
            f"stability_margin {t.stability_margin!r} is not atan2(support_margin, com_height) = {implied!r}"
        )

    if t.variant in _AXIS_NORMALS:
        expected = _AXIS_NORMALS[t.variant]
        if p.face_normal is None or not all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=_UNIT_ATOL) for a, b in zip(p.face_normal, expected)
        ):
            raise ValueError(f"variant {t.variant!r} promises face_normal {expected}, record says {p.face_normal}")
    elif p.primitive in _ENUMERATED:
        # face_index is 0-based, mirroring depth_index; the variant string is 1-based.
        # This is the only thing keeping the two from drifting.
        expected_variant = f"face-{p.face_index + 1}"
        if t.variant != expected_variant:
            raise ValueError(f"variant {t.variant!r} does not match face_index {p.face_index} ({expected_variant!r})")
