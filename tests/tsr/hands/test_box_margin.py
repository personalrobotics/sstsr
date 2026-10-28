# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""A box grasp's default margin scales with the box, and nothing else moves (issue #195).

The inferred default margin of a box top, bottom or face grasp is ``clearance_fraction ×
min(finger_length, approach_extent)``. Two compatibility promises come with it, and they are
checked against templates recorded before the change:

* a box at least as long as the finger along a family's approach gets exactly the templates
  it got before, since there ``min(finger_length, approach_extent)`` is ``finger_length``;
* an explicit ``clearance`` gets exactly the templates it got before, at any size.

"Exactly" means the serialized templates hash the same, ignoring only the two provenance
fields this change adds (``margin``, ``margin_basis``). The hashes are in
``tests/reference/box_margin_baseline.json``, recorded from sstsr 6a5e482 (#193's head,
before this change) with::

    git worktree add <dir> 6a5e482
    PYTHONPATH=<dir>/src uv run python tests/tsr/hands/test_box_margin.py --record

That the scaled margin keeps the fingertips inside the box, open and closed, is the property
in ``test_closed_reach.py``.
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path

from tsr.hands import FrankaHand, Robotiq2F85, Robotiq2F140

BASELINE = Path(__file__).resolve().parents[2] / "reference" / "box_margin_baseline.json"
HANDS = {"Robotiq2F140": Robotiq2F140, "Robotiq2F85": Robotiq2F85, "FrankaHand": FrankaHand}
FAMILIES = ("grasp_box_top", "grasp_box_bottom", "grasp_box_face_x", "grasp_box_face_y")
# Which box dimension each family approaches along.
APPROACH_AXIS = {"grasp_box_top": 2, "grasp_box_bottom": 2, "grasp_box_face_x": 0, "grasp_box_face_y": 1}
NEW_FIELDS = ("margin", "margin_basis")

# At least one dimension of each box reaches the longest finger (114 mm), so every hand has
# families whose approach extent is at least its finger length.
LARGE = ((0.04, 0.05, 0.06), (0.07, 0.07, 0.07), (0.05, 0.06, 0.12), (0.115, 0.115, 0.115), (0.12, 0.03, 0.2))
EXPLICIT = ((0.072, 0.088, 0.028), (0.03, 0.05, 0.015), (0.06, 0.04, 0.1), (0.115, 0.115, 0.115))
CLEARANCES = (0.002, 0.006)


def _cases():
    """``(key, hand, family, dims, clearance)`` for every promise above."""
    for name, make in HANDS.items():
        hand = make()
        for dims in LARGE:
            for family in FAMILIES:
                if dims[APPROACH_AXIS[family]] >= hand.finger_length:
                    yield f"{name}/{family}/{dims}/default", hand, family, dims, None
        for dims in EXPLICIT:
            for family in FAMILIES:
                for c in CLEARANCES:
                    yield f"{name}/{family}/{dims}/clearance={c}", hand, family, dims, c


def _digest(templates) -> str:
    out = []
    for t in templates:
        d = t.to_dict()
        meta = d.get("provenance", {}).get("metadata", {})
        for k in NEW_FIELDS:
            meta.pop(k, None)
        out.append(d)
    return hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()


def _run(hand, family, dims, clearance):
    return getattr(hand, family)(*dims, clearance=clearance)


class TestUnchangedWhereThePromiseHolds(unittest.TestCase):
    def test_large_boxes_and_explicit_clearances_are_byte_identical(self):
        baseline = json.loads(BASELINE.read_text())["cases"]
        keys = set()
        for key, hand, family, dims, c in _cases():
            keys.add(key)
            templates = _run(hand, family, dims, c)
            self.assertEqual(_digest(templates), baseline[key]["sha256"], key)
        self.assertEqual(keys, set(baseline))
        # Not vacuous: most cases produce templates.
        self.assertGreater(sum(1 for v in baseline.values() if v["count"]), len(baseline) // 2)


class TestProvenanceRecordsTheMargin(unittest.TestCase):
    def test_basis_and_margin(self):
        hand = Robotiq2F140()
        for family in FAMILIES:
            for c, basis in ((None, "min(finger_length, approach_extent)"), (0.004, "supplied")):
                templates = _run(hand, family, (0.05, 0.06, 0.07), c)
                self.assertTrue(templates, family)
                for t in templates:
                    meta = t.provenance.metadata
                    self.assertEqual(meta["margin_basis"], basis, family)
                    if c is not None:
                        self.assertEqual(meta["margin"], c, family)


def _record():
    cases = {}
    for key, hand, family, dims, c in _cases():
        templates = _run(hand, family, dims, c)
        cases[key] = {"count": len(templates), "sha256": _digest(templates)}
    import tsr

    doc = {"recorded_with": f"sstsr {tsr.__version__}", "cases": cases}
    BASELINE.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} cases to {BASELINE}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--record"]:
        _record()
    else:
        unittest.main()
