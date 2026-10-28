# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""How far a linkage hand's fingertips reach as it closes, measured from its MuJoCo model (#193).

A four-bar hand such as the Robotiq 2F-85 or 2F-140 swings its pads forward as it closes, so
a hand closed on a narrow object reaches further past the palm than the open hand's
``FINGER_LENGTH``. This closes the model's fingers on nothing, one settled control level at
a time, and reports, fully open and at each level:

* ``width`` -- the gap between the two pads' contact faces (the model's box pad geoms), the
  width of an object the fingers would stop on. It is the quantity ``MAX_APERTURE`` and a
  template's span measure, which is what ``ParallelJawGripper.reach`` is looked up with;
* ``reach`` -- how far the fingers' collision geometry (every follower and pad geom; a mesh
  bounded by its own-frame AABB, so never understated) extends past the TSR palm along the
  approach axis. The palm is ``base_mount + palm_offset`` on ``base_mount``'s +z, as
  ``PALM_OFFSET_FROM_BASE_MOUNT`` documents for each hand.

MuJoCo is not an sstsr dependency, so supply it for the run::

    uv run --with mujoco==3.14.0 python tools/finger_reach.py <2f85.xml> --palm-offset 0.094
    uv run --with mujoco==3.14.0 python tools/finger_reach.py <2f140.xml> --palm-offset 0.100

The table it prints is what ``Robotiq2F85.CLOSED_REACH`` and ``Robotiq2F140.CLOSED_REACH``
record; ``--json`` writes every level.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def measure(xml: str, palm_offset: float, levels: int = 52, settle_steps: int = 2000) -> list:
    import mujoco

    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mount = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_mount")
    side = {}
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        if ("follower" in body or "pad" in body) and m.geom_contype[g]:
            side.setdefault("left" if body.startswith("left_") else "right", []).append(g)
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)

    def corners(g):
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g] + R @ m.geom_aabb[g][:3]
        return [c + R @ (s * m.geom_aabb[g][3:]) for s in signs]

    def row(ctrl):
        R, o = d.xmat[mount].reshape(3, 3), d.xpos[mount]

        def local(geoms):
            return np.array([R.T @ (p - o) for g in geoms for p in corners(g)])

        fingers = {k: local(gs) for k, gs in side.items()}
        pads = {k: local([g for g in gs if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]) for k, gs in side.items()}
        reach = float(max(v[:, 2].max() for v in fingers.values())) - palm_offset
        # The pads face each other along one horizontal axis of base_mount.
        axis = int(np.argmax([abs(pads["left"][:, i].mean() - pads["right"][:, i].mean()) for i in (0, 1)]))
        near, far = sorted((pads["left"][:, axis], pads["right"][:, axis]), key=np.mean)
        return {"ctrl": ctrl, "width_m": float(far.min() - near.max()), "reach_m": reach}

    # Fully open: the model's rest configuration, qpos0, in which the linkage's equality
    # constraints hold exactly. Settling under the drive need not get there: geodude_assets'
    # 2F-140 stops about 18 mm short, its spring links holding against the drive's force limit.
    mujoco.mj_forward(m, d)
    rows = [row(None)]
    lo_ctrl, hi_ctrl = m.actuator_ctrlrange[0]
    for ctrl in np.linspace(lo_ctrl, hi_ctrl, levels):
        d.ctrl[0] = ctrl
        for _ in range(settle_steps):
            mujoco.mj_step(m, d)
        rows.append(row(float(ctrl)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("xml", help="the hand's MJCF, e.g. mujoco_menagerie robotiq_2f85/2f85.xml")
    parser.add_argument("--palm-offset", type=float, required=True, help="TSR palm past base_mount [m]")
    parser.add_argument("--json", help="write every settled level here")
    args = parser.parse_args()
    import mujoco

    rows = measure(args.xml, args.palm_offset)
    print(f"mujoco {mujoco.__version__}, {args.xml}")
    print(" width mm   reach mm")
    for r in rows:
        if r["width_m"] > 0.0:
            print(f"{r['width_m'] * 1e3:9.1f}  {r['reach_m'] * 1e3:9.1f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
