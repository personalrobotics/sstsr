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

``--wrap`` measures an underactuated hand whose fingers wrap around what they close on, the
Kinova JACO2's two fingers on ada_assets ``jaco2.xml``. Its fingers have no pads, and closed on
nothing they stop on each other, so a gap between pads says nothing about the object they
would stop on. Instead each level closes the fingers on a fixed cylinder of that width, and
``reach`` is read once they have stopped on it. How far they reach depends on where the
cylinder is, so each level takes the furthest over two placements: end on, its axis along the
approach and its end at the palm (a top grasp), and broadside, its axis along the palm normal
at every depth 4 mm apart where both fingers stop on it (a side grasp). The fully open row is
the model's rest configuration, and its width is that of the widest end-on cylinder the open
fingers clear. The hand is the arm's ``j2n6s200_link_6`` and the two
finger chains on it; the arm is frozen and kept out of collision. The approach is
``link_6``'s -z, and the palm lies in the fingers' closing plane, ``palm_offset`` along it::

    uv run --with mujoco==3.14.0 python tools/finger_reach.py <ada_assets>/src/ada_assets/models/jaco2.xml \
        --palm-offset 0.125 --wrap
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


#: The JACO2 hand in ada_assets ``jaco2.xml``: the body the fingers mount on, and its fingers.
JACO2_MOUNT = "j2n6s200_link_6"
JACO2_FINGERS = (
    ("j2n6s200_link_finger_1", "j2n6s200_link_finger_tip_1"),
    ("j2n6s200_link_finger_2", "j2n6s200_link_finger_tip_2"),
)


def wrap_hand_spec(xml: str, meshdir: str | None = None):
    """The JACO2 hand alone: arm joints, arm actuators and keyframes removed, the arm out of collision.

    The fingers keep their joints, tendons, springs and actuators. Gravity is off, so the
    fingers settle under their drives alone.
    """
    import os

    import mujoco

    spec = mujoco.MjSpec.from_file(xml)
    # ada_assets' component models leave meshdir to whoever assembles them.
    spec.meshdir = meshdir or os.path.join(os.path.dirname(os.path.abspath(xml)), "assets")
    spec.option.gravity = [0.0, 0.0, 0.0]
    hand = {JACO2_MOUNT, *(b for chain in JACO2_FINGERS for b in chain)}
    fingers = hand - {JACO2_MOUNT}
    # link_6 itself turns on the arm's last joint, which goes with the rest of the arm.
    for j in list(spec.joints):
        if j.parent.name not in fingers:
            spec.delete(j)
    for a in list(spec.actuators):
        if "finger" not in a.target:
            spec.delete(a)
    for k in list(spec.keys):
        spec.delete(k)
    for g in spec.geoms:
        if g.parent.name not in hand:
            g.contype = g.conaffinity = 0
    return spec


def wrap_palm(m, palm_offset: float) -> np.ndarray:
    """The palm in the mount frame: in the fingers' closing plane, ``palm_offset`` along -z."""
    x = np.mean([m.body(chain[0]).pos[0] for chain in JACO2_FINGERS])
    return np.array([x, 0.0, -palm_offset])


def finger_reach_past(m, d, palm_offset: float) -> float:
    """How far the fingers' collision meshes extend past the palm along the approach (-z of the mount).

    A collision mesh is its convex hull, whose extreme along any axis is a vertex, so this is exact.
    """
    mount = m.body(JACO2_MOUNT).id
    R, o = d.xmat[mount].reshape(3, 3), d.xpos[mount]
    fingers = {m.body(b).id for chain in JACO2_FINGERS for b in chain}
    deepest = -np.inf
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in fingers and m.geom_contype[g] and m.geom_dataid[g] >= 0:
            mesh = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
            local = (d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T - o) @ R
            deepest = max(deepest, float(-local[:, 2].min()))
    return deepest - palm_offset


def _probe_spec(xml, meshdir, palm, radius, depth=None, length=0.2):
    """The hand and a fixed cylinder of ``radius``.

    End on (``depth=None``) its axis is the approach and its end is at the palm, as an object
    is held in a top grasp. Broadside its axis is the palm normal (``link_6``'s x) and its
    axis is ``depth`` past the palm, as in a side grasp.
    """
    import mujoco

    spec = wrap_hand_spec(xml, meshdir)
    m0 = spec.compile()
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    R = d0.xmat[m0.body(JACO2_MOUNT).id].reshape(3, 3)
    if depth is None:
        local, axis = palm + np.array([0.0, 0.0, -length / 2]), np.eye(3)
    else:
        local, axis = (
            palm + np.array([0.0, 0.0, -depth]),
            np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        )
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (R @ axis).flatten())
    probe = spec.worldbody.add_body(name="probe", pos=d0.xpos[m0.body(JACO2_MOUNT).id] + R @ local, quat=quat)
    if radius > 0.0:
        probe.add_geom(name="probe", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, length / 2, 0])
    return spec


def _close(xml, meshdir, palm, palm_offset, radius, depth, settle_steps):
    """Close the fingers fully on a probe; ``(reach, both fingers touch it)``."""
    import mujoco

    m = _probe_spec(xml, meshdir, palm, radius, depth).compile()
    d = mujoco.MjData(m)
    d.ctrl[:] = m.actuator_ctrlrange[:, 1]
    for _ in range(settle_steps):
        mujoco.mj_step(m, d)
    touching = set()
    if radius > 0.0:
        probe = m.geom("probe").id
        for c in d.contact[: d.ncon]:
            if probe in (c.geom1, c.geom2):
                touching.add(m.body(m.geom_bodyid[c.geom2 if c.geom1 == probe else c.geom1]).name)
    both = all(any(b in touching for b in chain) for chain in JACO2_FINGERS)
    return finger_reach_past(m, d, palm_offset), both


def measure_wrap(
    xml: str, palm_offset: float, meshdir: str | None = None, step: float = 0.004, settle_steps: int = 3000
):
    """Fully open, then closed on cylinders of each width.

    Each row is ``{"width_m", "reach_m", "end_on_m", "broadside_m", "broadside_depth_m"}``;
    ``reach_m`` is the larger of the end-on reach and the largest broadside reach over every
    depth, in steps of ``step``, from touching the palm to the meeting fingertips, at which
    both fingers stop on the cylinder. The fully open row has only a width and a reach.
    """
    import mujoco

    m = wrap_hand_spec(xml, meshdir).compile()
    palm = wrap_palm(m, palm_offset)

    def clears(radius):
        mp = _probe_spec(xml, meshdir, palm, radius).compile()
        dp = mujoco.MjData(mp)
        mujoco.mj_forward(mp, dp)
        probe = mp.geom("probe").id
        return not any(probe in (c.geom1, c.geom2) and c.dist < 0.0 for c in dp.contact[: dp.ncon])

    # Fully open: qpos0. The widest cylinder the open fingers clear, to 0.1 mm.
    lo, hi = 0.0, 0.1
    while hi - lo > 5e-5:
        lo, hi = ((lo + hi) / 2, hi) if clears((lo + hi) / 2) else (lo, (lo + hi) / 2)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rows = [{"width_m": 2.0 * lo, "reach_m": finger_reach_past(m, d, palm_offset)}]
    meet, _ = _close(xml, meshdir, palm, palm_offset, 0.0, None, settle_steps)
    for width in np.arange(0.0, 2.0 * lo, step):
        r = width / 2
        end_on, _ = _close(xml, meshdir, palm, palm_offset, r, None, settle_steps)
        side, side_depth = -np.inf, None
        for depth in np.arange(r + 0.0005, meet + r, step) if r > 0.0 else ():
            reach, both = _close(xml, meshdir, palm, palm_offset, r, depth, settle_steps)
            if both and reach > side:
                side, side_depth = reach, float(depth)
        rows.append(
            {
                "width_m": float(width),
                "reach_m": float(max(end_on, side)),
                "end_on_m": float(end_on),
                "broadside_m": None if side_depth is None else float(side),
                "broadside_depth_m": side_depth,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("xml", help="the hand's MJCF, e.g. mujoco_menagerie robotiq_2f85/2f85.xml")
    parser.add_argument("--palm-offset", type=float, required=True, help="TSR palm past base_mount [m]")
    parser.add_argument("--json", help="write every settled level here")
    parser.add_argument("--wrap", action="store_true", help="an underactuated hand that wraps: the JACO2")
    parser.add_argument("--meshdir", help="with --wrap: the meshes, if not <xml dir>/assets")
    args = parser.parse_args()
    import mujoco

    print(f"mujoco {mujoco.__version__}, {args.xml}")
    if args.wrap:
        rows = measure_wrap(args.xml, args.palm_offset, args.meshdir)
        print(" width mm   reach mm     end on   broadside  (at depth mm)")
        for r in rows[1:] + rows[:1]:
            if "end_on_m" not in r:
                print(f"{r['width_m'] * 1e3:9.1f}  {r['reach_m'] * 1e3:9.1f}   fully open")
                continue
            side = (
                ""
                if r["broadside_m"] is None
                else f"{r['broadside_m'] * 1e3:9.1f}  ({r['broadside_depth_m'] * 1e3:.1f})"
            )
            print(f"{r['width_m'] * 1e3:9.1f}  {r['reach_m'] * 1e3:9.1f}  {r['end_on_m'] * 1e3:9.1f}  {side}")
    else:
        rows = measure(args.xml, args.palm_offset)
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
