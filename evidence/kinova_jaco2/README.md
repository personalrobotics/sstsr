# Kinova JACO2 two-finger hand: measurement and closure evidence (#185, #204)

The evidence behind `KinovaJaco2TwoFinger`'s dimensions, the check that its cylinder grasps close
on the cylinder, and the evidence for its enveloping grasps. Every file is the tools' output,
byte for byte.

| file | sha256 | written by |
|---|---|---|
| `finger_reach.json` | `8cced02bd21ae06b155ec13e70114aedf8cc862a74432508ba60e54903857b1d` | `tools/finger_reach.py --wrap` |
| `grasp_closure.json` | `e547a0cc89f2b041cdea7f55a6fccac29a6116a2f29e6c122ddb4fb641b3b060` | `tools/grasp_closure.py` |
| `envelop.json` | `9cc74ab37f20fc9455b293f66cbb1bb792f0bd28afa49a7c51ff6fb85c4a34b4` | `tools/finger_reach.py --envelop` |
| `grasp_hold.json` | `34e3aa395044671fe77d770f12c64c5ff256bb4a18ae007c5ed1bb73a5c171a3` | `tools/grasp_hold.py` |

## Inputs

- Model, for `finger_reach.json` and `grasp_closure.json`: ada_assets
  `src/ada_assets/models/jaco2.xml` and its meshes at commit `e7798be`.
- Model, for `envelop.json` and `grasp_hold.json`: ada_assets
  `src/ada_assets/models/ada_hand.xml` and its meshes at commit `0703586`, the bare-hand model
  ADA runs. They close the fingers on a free object, which depends on the hand's grip force
  and fingertip friction (ada_assets#47, after `e7798be`) and on the solver options the
  assembled model sets (`impratio`, `implicitfast`, `elliptic`): `jaco2.xml` alone sets none,
  and on MuJoCo's defaults a free object in its closing hand is not stable.
- MuJoCo 3.14.0 (supplied with `uv run --with`; it is not an sstsr dependency).
- Recorded on macOS arm64, Python 3.14.7, NumPy 2.3.2. MuJoCo's stepping is deterministic for
  one build on one platform, so a rerun elsewhere may differ in the last digits of
  `finger_reach.json`. `grasp_closure.json` is rounded to 0.1 mm.

## Reproduce

From the repository root, with ada_assets checked out at `e7798be` in `<ada_assets>`:

```bash
uv run --with mujoco==3.14.0 python tools/finger_reach.py \
    <ada_assets>/src/ada_assets/models/jaco2.xml --palm-offset 0.125 --wrap \
    --json finger_reach.json
uv run --with mujoco==3.14.0 python tools/grasp_closure.py \
    <ada_assets>/src/ada_assets/models/jaco2.xml --json grasp_closure.json
shasum -a 256 finger_reach.json grasp_closure.json
```

and, with ada_assets checked out at `0703586` in `<ada_assets>`:

```bash
uv run --with mujoco==3.14.0 python tools/finger_reach.py \
    <ada_assets>/src/ada_assets/models/ada_hand.xml --palm-offset 0.125 --envelop --closed 1.35 \
    --json envelop.json
uv run --with mujoco==3.14.0 python tools/grasp_hold.py \
    <ada_assets>/src/ada_assets/models/ada_hand.xml --json grasp_hold.json
shasum -a 256 envelop.json grasp_hold.json
```

## What to look for

- `finger_reach.json`. The first row is the fully open hand: `width_m` 0.0735 (`MAX_APERTURE`
  0.073) and `reach_m` 0.0374, the open fingertips. The second row, width 0, is the fingertips
  meeting at 0.0729 (`FINGER_LENGTH` 0.073). The remaining rows are the `CLOSED_REACH` curve.
  Each gives the end-on and broadside reach, and the depth of the furthest broadside placement.
- `grasp_closure.json`. `"failed": 0` of `"poses": 138`. Each pose records the following:
  - whether the open hand penetrates the cylinder;
  - which fingers touch it closed;
  - whether the palm touches it;
  - the closed reach against the cylinder's far extent.

  `too_wide` is the 37 mm-radius cylinder, which gets no templates. `grasp_closure.py` exits
  nonzero if any pose fails. `enveloping` counts the enveloping templates it leaves to the two
  files below: a fixed cylinder cannot show where a free one meets the curling fingers.
- `envelop.json`. One row per width, 4 mm apart, of a free cylinder posed as an enveloping
  side grasp poses it, closed on with the declared 1.35. From 32 mm (`ENVELOPING_DIAMETERS`)
  to 72 mm, both fingers grip it with their tips 38–56 mm past its axis; narrower, the fingers
  miss it. From 56 mm the palm touches it too: narrower enveloped cylinders are wrapped by the
  two fingers.
- `grasp_hold.json`. Each side template, fingertip and enveloping, closed on a free cylinder,
  its support then lowered 60 mm and held 5 s. On ADA's 40 mm cylinder the fingertip grasps
  turn it up to 0.34–0.52 rad in the hand and pull it 12–20 mm across it; the enveloping
  grasp, 0.04 rad and 2.2 mm. `held` is recorded but does not tell them apart: a hand alone,
  held still, keeps most of them (each height band's low end fails at every depth, its
  fingers at the support). On ADA's arm, the fingertip grasps are lost after a lift and the
  enveloping ones held (#204); the hand alone does not reproduce that, and the roll-in is
  the property this file shows.
