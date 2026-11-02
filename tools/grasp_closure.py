# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Close the JACO2's fingers on cylinders placed by its own grasp templates (#185).

``KinovaJaco2TwoFinger``'s dimensions come from its MuJoCo model, so its templates can be
checked against the same model: a grasp is only as good as what the fingers do when they
close. For each cylinder below, every side, top and bottom template is instantiated with
the cylinder at the origin, and posed at its Bw midpoint and at both ends of its height
band. The cylinder is fixed in place at that pose relative to the hand, as
``tools/finger_reach.py --wrap`` builds it from ada_assets ``jaco2.xml``, and the fingers
close to the end of their range. A pose passes when:

* open, nothing of the hand penetrates the cylinder;
* closed, both fingers touch the cylinder and the palm (``j2n6s200_link_6``'s housing) does
  not; and
* closed, the fingertips are still inside the cylinder's far extent along the approach,
  which is what ``CLOSED_REACH`` promises (#193).

A cylinder wider than the aperture must get no templates at all. The report is written
sorted and rounded to 0.1 mm, so two runs give the same bytes::

    uv run --with mujoco==3.14.0 python tools/grasp_closure.py \\
        <ada_assets>/src/ada_assets/models/jaco2.xml --json grasp_closure.json

It exits nonzero if any pose fails.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from finger_reach import JACO2_FINGERS, JACO2_MOUNT, finger_reach_past, wrap_hand_spec

from tsr.hands import KinovaJaco2TwoFinger

#: (radius, height) [m]: thin to near the aperture, short (the closed reach binds on top
#: grasps) and tall. (0.025, 0.10) is ADA's cylinder.
CYLINDERS = ((0.005, 0.03), (0.005, 0.10), (0.0125, 0.10), (0.025, 0.03), (0.025, 0.10), (0.033, 0.10))
#: Wider than the 73 mm aperture: no templates.
TOO_WIDE = ((0.037, 0.10),)


def _palm_in_mount(hand) -> np.ndarray:
    """The canonical TSR frame in ``j2n6s200_link_6``: the palm, rotated 180° about x."""
    T = np.diag([1.0, -1.0, -1.0, 1.0])
    T[:3, 3] = [hand.PALM_X_IN_LINK_6, 0.0, -hand.PALM_OFFSET_FROM_LINK_6]
    return T


def _poses(template):
    """``(label, T_cylinder_palm)``: the Bw midpoint, then each end of the height band."""
    tsr = template.instantiate(np.eye(4))
    mid = tsr.Bw.mean(axis=1)
    mid[3:] = np.where(np.isfinite(mid[3:]), mid[3:], 0.0)
    out = [("mid", tsr.to_transform(mid))]
    for end, z in (("low", tsr.Bw[2, 0]), ("high", tsr.Bw[2, 1])):
        if z != mid[2]:
            xi = mid.copy()
            xi[2] = z
            out.append((end, tsr.to_transform(xi)))
    return out


def _far_extent(T_cyl_palm, r, h) -> float:
    """How far past the palm, along the approach, the cylinder (axis +z over ``[0, h]``) extends."""
    u, p = T_cyl_palm[:3, 2], T_cyl_palm[:3, 3]
    return max(0.0, u[2] * h) + r * float(np.sqrt(max(0.0, 1.0 - u[2] ** 2))) - float(p @ u)


def close_on(xml, meshdir, hand, T_cyl_palm, r, h, settle_steps=3000) -> dict:
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
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, T_w_cyl[:3, :3].flatten())
    body = spec.worldbody.add_body(name="cylinder", pos=T_w_cyl[:3, 3], quat=quat)
    body.add_geom(name="cylinder", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, h / 2, 0], pos=[0, 0, h / 2])
    m = spec.compile()
    d = mujoco.MjData(m)
    cyl = m.geom("cylinder").id

    def touching():
        out = {}
        for c in d.contact[: d.ncon]:
            if cyl in (c.geom1, c.geom2):
                name = m.body(m.geom_bodyid[c.geom2 if c.geom1 == cyl else c.geom1]).name
                out[name] = min(out.get(name, np.inf), float(c.dist))
        return out

    mujoco.mj_forward(m, d)
    opened = touching()
    d.ctrl[:] = m.actuator_ctrlrange[:, 1]
    for _ in range(settle_steps):
        mujoco.mj_step(m, d)
    closed = touching()
    palm = hand.PALM_OFFSET_FROM_LINK_6
    reach = finger_reach_past(m, d, palm)
    far = _far_extent(T_cyl_palm, r, h)
    fingers = [any(b in closed for b in chain) for chain in JACO2_FINGERS]
    result = {
        "open_penetration_mm": round(min([0.0, *opened.values()]) * -1e3, 1),
        "fingers_touching": fingers,
        "palm_touching": JACO2_MOUNT in closed,
        "closed_reach_mm": round(reach * 1e3, 1),
        "far_extent_mm": round(far * 1e3, 1),
    }
    result["pass"] = bool(not opened and all(fingers) and JACO2_MOUNT not in closed and reach <= far)
    return result


def run(xml: str, meshdir: str | None = None) -> dict:
    hand = KinovaJaco2TwoFinger()
    report = {"hand": type(hand).__name__, "cylinders": [], "too_wide": []}
    for r, h in CYLINDERS:
        entry = {"radius_m": r, "height_m": h, "templates": []}
        for t in hand.grasp_cylinder(r, h):
            for label, T in _poses(t):
                res = close_on(xml, meshdir, hand, T, r, h)
                entry["templates"].append({"template": t.name, "pose": label, **res})
        report["cylinders"].append(entry)
    for r, h in TOO_WIDE:
        report["too_wide"].append({"radius_m": r, "height_m": h, "templates": len(hand.grasp_cylinder(r, h))})
    poses = [p for c in report["cylinders"] for p in c["templates"]]
    report["poses"] = len(poses)
    report["failed"] = sum(not p["pass"] for p in poses) + sum(w["templates"] > 0 for w in report["too_wide"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("xml", help="ada_assets src/ada_assets/models/jaco2.xml")
    parser.add_argument("--meshdir", help="the meshes, if not <xml dir>/assets")
    parser.add_argument("--json", help="write the report here")
    args = parser.parse_args()
    import mujoco

    report = run(args.xml, args.meshdir)
    report["mujoco"] = mujoco.__version__
    print(f"mujoco {mujoco.__version__}, {args.xml}")
    for c in report["cylinders"]:
        n = len(c["templates"])
        ok = sum(p["pass"] for p in c["templates"])
        print(f"r {c['radius_m'] * 1e3:5.1f} mm  h {c['height_m'] * 1e3:5.1f} mm: {ok}/{n} poses close on it")
        for p in c["templates"]:
            if not p["pass"]:
                print(f"  FAIL {p['template']} ({p['pose']}): {p}")
    for w in report["too_wide"]:
        print(
            f"r {w['radius_m'] * 1e3:5.1f} mm  h {w['height_m'] * 1e3:5.1f} mm: "
            f"{w['templates']} templates (wider than the aperture)"
        )
    print(f"{report['poses']} poses, {report['failed']} failed")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, sort_keys=True)
            f.write("\n")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
