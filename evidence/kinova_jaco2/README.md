# Kinova JACO2 two-finger hand: measurement and closure evidence (#185)

The evidence behind `KinovaJaco2TwoFinger`'s dimensions, and the check that its cylinder grasps close
on the cylinder. Both files are the tools' output, byte for byte.

| file | sha256 | written by |
|---|---|---|
| `finger_reach.json` | `8cced02bd21ae06b155ec13e70114aedf8cc862a74432508ba60e54903857b1d` | `tools/finger_reach.py --wrap` |
| `grasp_closure.json` | `cbb37a6757301f33a9497e5f39b9f3e9ded60555f81777c586d29e675394e876` | `tools/grasp_closure.py` |

## Inputs

- Model: ada_assets `src/ada_assets/models/jaco2.xml` and its meshes at commit `e7798be`.
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
  nonzero if any pose fails.
