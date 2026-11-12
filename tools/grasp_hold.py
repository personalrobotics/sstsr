# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Close the JACO2 on a free cylinder posed by its side templates, and measure how far the
close pulls it in the hand (#204).

``tools/grasp_closure.py`` closes the fingers on a cylinder fixed in place, so it shows where
the fingers stop, not what they do to an object that can move. The JACO2's fingers pinch a
free object at their tips and then pull it into the hand as they curl, turning it as they do:
the roll-in an enveloping grasp, with the palm close and the fingers folded around the object,
avoids. This trial frees the cylinder. For each cylinder below and each of the JACO2's side
templates, fingertip and enveloping, instantiated at the Bw midpoint and both ends of its
height band, the hand alone (``wrap_hand_spec``):

1. stands the cylinder, free and of ``MASS_KG``, on a table under its base, posed relative
   to the palm by the template alone. Gravity is turned to point down the cylinder's axis,
   so an upright side grasp needs no arm, and the hand stays fixed;
2. lets it settle, then drives the fingers to ``CLOSED`` for ``CLOSE_S``;
3. lowers the table ``DROP_M`` along the axis over ``DROP_S``, as a lift of that height
   would take the support away, so only the fingers carry the cylinder;
4. holds still for ``HOLD_S``.

The simulator's state scores the outcome and nothing else. Each pose records:

* ``roll_in_rad``: how far the close turned the cylinder in the hand, its axis at the end of
  the close against its axis as the template posed it;
* ``pull_in_mm``: how far the close moved its axis across the hand from where it was posed;
* ``slip_mm``: how far it slid down its axis once the table was lowered, through the hold;
* ``held``: at the end of the hold it has slipped at most ``SLIP_TOLERANCE_M`` and both
  fingers touch it.

A hand alone holds still, so ``held`` says little: on ADA's arm fingertip grasps that hold
here are lost after a lift (#204). The roll-in is the property. The report is sorted and
rounded, so two runs give the same bytes. Run it on the assembled bare-hand model, whose
solver options are the hand's::

    uv run --with mujoco==3.14.0 python tools/grasp_hold.py \\
        <ada_assets>/src/ada_assets/models/ada_hand.xml --json grasp_hold.json

It reports; it does not pass or fail.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from finger_reach import JACO2_FINGERS, JACO2_MOUNT, wrap_hand_spec
from grasp_closure import _palm_in_mount, _poses

from tsr.hands import KinovaJaco2TwoFinger

#: (radius, height) [m]: grasp_closure.py's cylinders that side templates reach, and ADA's
#: 4 cm reference cylinder.
CYLINDERS = ((0.0125, 0.10), (0.02, 0.12), (0.025, 0.10), (0.033, 0.10))
MASS_KG = 0.1
#: The fingers' closed command: ada_assets' declared ``closed`` (models/mujoco.toml), what
#: ADA's runtime commands, not the actuators' range end (1.51).
CLOSED = 1.35
GRAVITY = 9.81
SETTLE_S = 0.5
CLOSE_S = 20.0
DROP_M = 0.06
DROP_S = 1.0
HOLD_S = 5.0
SLIP_TOLERANCE_M = 0.01
#: The table: a slab under the cylinder's base, normal to its axis.
TABLE_HALF = (0.3, 0.3, 0.02)
TABLE_FRICTION = (0.8, 0.005, 0.0001)


def _quat(R: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def trial(xml: str, meshdir: str | None, hand, T_cyl_palm: np.ndarray, r: float, h: float) -> dict:
    """One close, lift and hold; see the module."""
    import mujoco

    spec = wrap_hand_spec(xml, meshdir)
    m0 = spec.compile()
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    mount = m0.body(JACO2_MOUNT).id
    T_w_mount = np.eye(4)
    T_w_mount[:3, :3] = d0.xmat[mount].reshape(3, 3)
    T_w_mount[:3, 3] = d0.xpos[mount]
    T_w_cyl = T_w_mount @ _palm_in_mount(hand) @ np.linalg.inv(T_cyl_palm)
    up = T_w_cyl[:3, 2]  # the cylinder's axis: the template frames its base at the origin

    spec.option.gravity = list(-GRAVITY * up)
    cyl = spec.worldbody.add_body(name="cylinder", pos=T_w_cyl[:3, 3], quat=_quat(T_w_cyl[:3, :3]))
    cyl.add_freejoint()
    cyl.add_geom(
        name="cylinder", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, h / 2, 0], pos=[0, 0, h / 2], mass=MASS_KG
    )
    table_pos = T_w_cyl[:3, 3] - up * TABLE_HALF[2]
    table = spec.worldbody.add_body(name="table", mocap=True, pos=table_pos, quat=_quat(T_w_cyl[:3, :3]))
    table.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(TABLE_HALF), friction=list(TABLE_FRICTION))
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cid, gid = m.body("cylinder").id, m.geom("cylinder").id
    mocap = m.body_mocapid[m.body("table").id]
    dt = m.opt.timestep

    def run(seconds: float, lowered=None) -> None:
        for i in range(int(round(seconds / dt))):
            if lowered is not None:
                d.mocap_pos[mocap] = table_pos - up * lowered(i * dt)
            mujoco.mj_step(m, d)

    def axis_height() -> float:
        return float(d.xpos[cid] @ up)

    # The template's pose, before anything moves: the reference for roll-in and pull-in.
    posed_axis, posed_at = d.xmat[cid].reshape(3, 3)[:, 2].copy(), d.xpos[cid].copy()
    run(SETTLE_S)
    d.ctrl[:] = CLOSED
    run(CLOSE_S)
    closed = axis_height()
    roll_in = float(np.arccos(np.clip(abs(d.xmat[cid].reshape(3, 3)[:, 2] @ posed_axis), -1.0, 1.0)))
    moved = d.xpos[cid] - posed_at
    pull_in = float(np.linalg.norm(moved - (moved @ up) * up))
    run(DROP_S, lambda t: DROP_M * min(1.0, t / DROP_S))
    run(HOLD_S, lambda t: DROP_M)

    touching = set()
    for c in d.contact[: d.ncon]:
        if gid in (c.geom1, c.geom2):
            touching.add(m.body(m.geom_bodyid[c.geom2 if c.geom1 == gid else c.geom1]).name)
    both = all(any(b in touching for b in chain) for chain in JACO2_FINGERS)
    slip = closed - axis_height()
    held = bool(both and slip <= SLIP_TOLERANCE_M)
    return {
        "roll_in_rad": round(roll_in, 2),
        "pull_in_mm": round(pull_in * 1e3, 1),
        "slip_mm": round(slip * 1e3, 1),
        "fingers_touching": both,
        "held": held,
    }


def run(xml: str, meshdir: str | None = None) -> dict:
    hand = KinovaJaco2TwoFinger()
    report = {"hand": type(hand).__name__, "cylinders": []}
    for r, h in CYLINDERS:
        entry = {"radius_m": r, "height_m": h, "templates": []}
        for t in hand.grasp_cylinder_side(r, h):
            standoff = round(float(t.Tw_e[0, 3]) * 1e3, 1)
            for label, T in _poses(t):
                res = trial(xml, meshdir, hand, T, r, h)
                entry["templates"].append(
                    {"template": t.name, "grip": t.provenance.grip, "standoff_mm": standoff, "pose": label, **res}
                )
        report["cylinders"].append(entry)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("xml", help="ada_assets src/ada_assets/models/ada_hand.xml (its solver options are the hand's)")
    parser.add_argument("--meshdir", help="the meshes, if not <xml dir>/assets")
    parser.add_argument("--json", help="write the report here")
    args = parser.parse_args()
    import mujoco

    report = run(args.xml, args.meshdir)
    report["mujoco"] = mujoco.__version__
    print(f"mujoco {mujoco.__version__}, {args.xml}")
    for c in report["cylinders"]:
        print(f"r {c['radius_m'] * 1e3:5.1f} mm  h {c['height_m'] * 1e3:5.1f} mm")
        by = {}
        for p in c["templates"]:
            by.setdefault((p["grip"] or "fingertip", p["standoff_mm"]), []).append(p)
        for (grip, standoff), ps in sorted(by.items(), key=lambda kv: -kv[0][1]):
            roll = max(p["roll_in_rad"] for p in ps)
            pull = max(p["pull_in_mm"] for p in ps)
            print(
                f"  {grip:10s} palm {standoff:5.1f} mm from the axis: roll-in at most {roll:.2f} rad, "
                f"pull-in at most {pull:5.1f} mm, {sum(p['held'] for p in ps)}/{len(ps)} held"
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, sort_keys=True)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
