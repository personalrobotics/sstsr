# Task Space Regions (TSR) Tutorial

A comprehensive guide to understanding and using Task Space Regions for robotic manipulation.

## Table of Contents

1. [Introduction](#introduction)
2. [What is a TSR?](#what-is-a-tsr)
3. [Mathematical Formulation](#mathematical-formulation)
4. [The Bounds Matrix](#the-bounds-matrix)
5. [TSR Chains](#tsr-chains)
6. [TSR Templates](#tsr-templates)
7. [Naming Conventions](#naming-conventions)
8. [Generating Grasp Templates](#generating-grasp-templates)
9. [Generating Placement Templates](#generating-placement-templates)
10. [Visualization](#visualization)
11. [Practical Examples](#practical-examples)

---

## Introduction

When programming robots to manipulate objects, we often need to specify **where** and **how** the robot should approach, grasp, or place something. The naive approach — specifying a single exact pose — is brittle:

- Manufacturing tolerances mean objects are never exactly where we expect
- A mug can be placed anywhere on a table, not just one specific spot
- A drawer handle can be grasped at any point along its length
- A pouring motion must keep the end-effector in a valid tilt range *throughout* the trajectory

**Task Space Regions (TSRs)** solve this by specifying a *region* of valid poses rather than a single pose. A TSR can express:

- **Grasp constraints**: all valid end-effector poses that achieve force closure on an object
- **Placement constraints**: all valid poses where an object can be set down
- **Trajectory-wide constraints**: a region the end-effector must stay inside for every waypoint along a motion (e.g., keep bottle tilted while pouring, keep tray level while carrying)

Because TSRs are continuous regions, motion planners can sample from them, check containment, and compute distance-to-valid — all without enumerating discrete poses.

---

## What is a TSR?

A **Task Space Region** constrains where one thing (the **subject**) can be, relative to another thing (the **reference**).

- A grasp TSR says: "given where the mug (reference) is, where can the gripper (subject) go?"
- A placement TSR says: "given where the table (reference) is, where can the mug (subject) go?"

Same math, different roles.

TSRs are useful for:

- **Grasping**: valid pre-grasp configurations for a given object geometry
- **Placement**: where objects can be set down (on a table, in a bin, on a shelf)
- **Motion constraints**: keeping the end-effector in a valid region during manipulation (pouring, carrying, wiping)
- **Articulated objects**: constraints that move with doors, drawers, rotating handles

---

## Mathematical Formulation

A TSR is defined by three components:

### 1. World-to-TSR Transform: $T_0^w$

This 4×4 homogeneous transformation matrix positions the TSR frame in the world. It represents the reference entity's pose — where the constraint region is located.

$$T_0^w = \begin{bmatrix}
R_{3\times3} & t_{3\times1} \\
0_{1\times3} & 1
\end{bmatrix}$$

**Example**: For grasping a mug, $T_0^w$ is the mug's pose. For placing on a table, it is the table surface's pose.

### 2. TSR-to-End-Effector Transform: $T_w^e$

This 4×4 matrix defines the nominal (center) pose of the subject relative to the TSR frame when all Bw displacements are zero. It encodes the canonical geometry of the constraint: standoff distance, approach orientation, grasp depth, etc.

**Example**: For a side grasp on a cylinder, $T_w^e$ encodes the gripper's standoff distance and approach orientation.

### 3. Bounds Matrix: $B_w$

A 6×2 matrix defining allowed deviations from the nominal pose:

$$B_w = \begin{bmatrix}
x_{min} & x_{max} \\
y_{min} & y_{max} \\
z_{min} & z_{max} \\
roll_{min} & roll_{max} \\
pitch_{min} & pitch_{max} \\
yaw_{min} & yaw_{max}
\end{bmatrix}$$

### Sampling from a TSR

To get a concrete subject pose, sample $s = (\Delta x, \Delta y, \Delta z, \Delta roll, \Delta pitch, \Delta yaw)$ from within $B_w$, then:

```
subject_pose = T0_w  ×  xyzrpy_to_transform(s)  ×  Tw_e
               ─────    ──────────────────────────    ────
               reference    deviation from canonical    canonical
               pose         (within bounds)             subject pose
```

### Checking if a Pose is Valid

1. Compute $T_{rel} = (T_0^w)^{-1} \cdot T_{ee} \cdot (T_w^e)^{-1}$
2. Extract displacement $\Delta$ from $T_{rel}$
3. Check each component against $B_w$

```python
from tsr import TSR
import numpy as np

tsr = TSR(T0_w=object_pose, Tw_e=gripper_offset, Bw=Bw)

pose        = tsr.sample()            # random SE(3) pose in the region
is_valid    = tsr.contains(pose)      # True/False containment check
distance, _ = tsr.distance(pose)      # distance to nearest valid pose (0 if inside)
```

---

## The Bounds Matrix

The 6×2 bounds matrix $B_w$ is the heart of a TSR. Each row constrains one degree of freedom:

| Row | DOF | Units | Description |
|-----|-----|-------|-------------|
| 0 | X | meters | Translation along local X axis |
| 1 | Y | meters | Translation along local Y axis |
| 2 | Z | meters | Translation along local Z axis |
| 3 | Roll | radians | Rotation about local X axis |
| 4 | Pitch | radians | Rotation about local Y axis |
| 5 | Yaw | radians | Rotation about local Z axis |

### Special Values

- **Fixed DOF**: set min = max (e.g., `[0, 0]` pins the axis)
- **Free rotation**: `[-π, π]` for full 360° freedom
- **Symmetric range**: `[-r, r]` centers the range at zero

### Common $B_w$ Patterns

**Object on a table** (placement):
```python
Bw = np.array([
    [-0.15, 0.15],    # x: 30 cm range on surface
    [-0.15, 0.15],    # y: 30 cm range on surface
    [0.0,   0.0 ],    # z: exactly on surface
    [0.0,   0.0 ],    # roll: upright
    [0.0,   0.0 ],    # pitch: upright
    [-np.pi, np.pi],  # yaw: any orientation
])
```

**Side grasp on a cylinder** (any angle around, limited height band):
```python
Bw = np.array([
    [0.0,  0.0 ],         # x: no radial freedom (baked into Tw_e)
    [0.0,  0.0 ],         # y: no tangential freedom
    [-0.03, 0.03],        # z: 6 cm height band
    [0.0,  0.0 ],         # roll: fixed by Tw_e
    [0.0,  0.0 ],         # pitch: fixed
    [-np.pi, np.pi],      # yaw: approach from any angle
])
```

**Pouring constraint** (trajectory-wide):
```python
Bw = np.array([
    [-0.02, 0.02],          # x: small tolerance over glass
    [-0.02, 0.02],          # y: small tolerance over glass
    [0.15,  0.25],          # z: 10 cm height band above glass
    [-np.pi/4, -np.pi/6],   # roll: tilted 30–45° for pouring
    [0.0,   0.0 ],          # pitch
    [-np.pi, np.pi],        # yaw: any rotation around glass axis
])
```

---

## TSR Chains

A **TSR Chain** connects multiple TSRs in sequence, where each TSR's frame is relative to the previous one. This is essential for **articulated objects** and **multi-constraint trajectories**.

### Example: Door Handle Grasp

The handle's reachable pose depends on the door's current angle:

```
World → Door Frame (hinge) → Handle Frame (on door) → Grasp Frame
```

```python
from tsr import TSR, TSRChain

# TSR 1: door rotation about hinge
door_tsr = TSR(
    T0_w=door_hinge_pose,
    Tw_e=np.eye(4),
    Bw=np.array([
        [0, 0], [0, 0], [0, 0],          # no translation
        [0, 0], [0, 0], [-np.pi/2, 0],   # door angle: closed to 90° open
    ])
)

# TSR 2: handle fixed on door
handle_tsr = TSR(
    T0_w=np.eye(4),         # relative to previous TSR's frame
    Tw_e=handle_offset,
    Bw=np.zeros((6, 2))
)

# TSR 3: grasp tolerance on handle
grasp_tsr = TSR(
    T0_w=np.eye(4),
    Tw_e=np.eye(4),
    Bw=np.array([
        [-0.02, 0.02], [0, 0], [0, 0],    # small position tolerance
        [0, 0], [0, 0], [-np.pi, np.pi],  # any grasp rotation
    ])
)

chain = TSRChain(TSRs=[door_tsr, handle_tsr, grasp_tsr])
pose  = chain.sample()
```

### How Chains Work

1. Sample/evaluate the first TSR to get $T_1$
2. Use $T_1$ as the world frame for TSR 2, producing $T_2$
3. Continue until the final pose $T_n$

The result is valid for any door angle within the allowed range — the planner doesn't need to enumerate door positions.

---

## TSR Templates

A **TSRTemplate** is a serializable TSR that is not yet bound to a specific scene object. It stores the constraint relative to a reference object's local frame. At runtime you bind it by supplying the reference object's world pose.

```python
from tsr import TSRTemplate, save_template, load_template
import numpy as np

# Build a template (reference object frame as origin)
template = TSRTemplate(
    T_ref_tsr=np.eye(4),    # TSR frame relative to reference object
    Tw_e=Tw_e,
    Bw=Bw,
    task="place",
    subject="mug",
    reference="table",
    name="Mug on Table",
    description="Place mug upright anywhere on table surface",
)

# Save to disk
save_template(template, "templates/mug_on_table.yaml")

# Load later
template = load_template("templates/mug_on_table.yaml")

# Bind to the actual table pose at runtime and sample
table_pose = perception.get_table_pose()
pose = template.sample(table_pose)

# Or get the TSR first if you need distance/contains
tsr  = template.instantiate(table_pose)
pose = tsr.sample()
```

Templates separate **constraint design** (done offline, per object type) from **constraint binding** (done at runtime, per scene instance). A library of templates can be reused across any scene where those objects appear.

### Serialized Format

Templates are stored as YAML with raw TSR matrices plus semantic metadata.

**Required fields:** `name`, `description`, `task`, `subject`, `reference`, `T_ref_tsr`, `Tw_e`, `Bw`

**Optional fields:** `variant`, `preshape`, `stability_margin`

```yaml
name: Mug Side Grasp
description: Side grasp avoiding handle
task: grasp
subject: gripper
reference: mug

T_ref_tsr:           # 4×4 reference-to-TSR-frame transform (usually identity)
- [1.0, 0.0, 0.0, 0.0]
- [0.0, 1.0, 0.0, 0.0]
- [0.0, 0.0, 1.0, 0.0]
- [0.0, 0.0, 0.0, 1.0]
Tw_e:                # 4×4 canonical subject pose in TSR frame (at Bw = 0)
- [0.0,  0.0, -1.0,  0.09]   # EE x-axis = cylinder axis; EE origin 9 cm out
- [0.0,  1.0,  0.0,  0.0 ]   # EE y-axis = tangential (finger opening)
- [1.0,  0.0,  0.0,  0.0 ]   # EE z-axis = -radial (approach toward mug)
- [0.0,  0.0,  0.0,  1.0 ]
Bw:                  # (6,2) bounds over [x, y, z, roll, pitch, yaw]
- [0.0,  0.0 ]       # x: fixed (radius is in Tw_e)
- [0.0,  0.0 ]       # y: fixed
- [0.02, 0.08]       # z: height on mug body
- [0.0,  0.0 ]       # roll: fixed
- [0.0,  0.0 ]       # pitch: fixed
- [0.79, 5.50]       # yaw: 45°–315° (avoids handle at 0°)

# optional
preshape: [0.08]     # gripper aperture in meters
```

`T_ref_tsr` is usually identity (TSR frame = reference frame). It is non-identity when the canonical grasp point is offset from the object origin — for example, a mug handle grasp where the TSR frame is at the handle center, not the mug center.

`task`, `subject`, and `reference` are plain strings with no enum validation. Use any values meaningful to your application.

### Loading and saving

```python
from tsr.io import load_template, save_template, load_templates_from_directory

template = load_template("templates/grasps/mug_side_grasp.yaml")
tsr = template.instantiate(mug_pose_in_world)

templates = load_templates_from_directory("templates/grasps/")
```

---

## Naming Conventions

The original TSR paper uses `T0_w` (world-to-TSR), `Tw_e` (TSR-to-end-effector), and `Bw` (bounds in TSR frame). The "end-effector" language comes from the grasp case where the subject is always a gripper. In placement TSRs, the "end-effector" is actually the object being placed.

In this library:

- **reference** — the entity whose pose defines `T0_w` (mug for grasps, table for placements)
- **subject** — the entity whose pose is being constrained (gripper for grasps, mug for placements)
- `task`, `subject`, and `reference` are plain strings in the template format — no enum enforcement

---

## Generating Grasp Templates

For common geometries, grasp templates can be generated programmatically from object dimensions rather than hand-authored. The `ParallelJawGripper` class in `tsr.hands` demonstrates the pattern.

### Gripper Frame Convention

All templates in this library use a canonical EE frame:

```
z = approach direction  (toward object surface)
y = finger opening direction
x = palm normal         (right-hand rule: x = y × z)
```

AnyGrasp / GraspNet uses `x = approach` — convert with:
```python
R_convert = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
```

### Why Radius Goes into Tw_e

The composition formula:

```
subject_pose = T0_w  ×  xyzrpy_to_transform(bw)  ×  Tw_e
```

`xyzrpy_to_transform` builds `[R | t]` where `R` comes from the angular components and `t` from the translational components. `R` rotates `Tw_e`'s translation — but `t` is **not** rotated by `R`.

- A translation in **Tw_e** sweeps a circle/sphere as angles in Bw vary ✓
- A translation in **Bw** stays fixed regardless of angles ✗

So radius (and standoff) go in `Tw_e`, not `Bw`:

```
Ring grasp (radius = 0.08, yaw varies):
  Tw_e translation = [0.08, 0, 0]     ← radius here

  yaw=0°   → R_z(0°)  @ [0.08,0,0] = [0.08,  0,    0]  ✓
  yaw=90°  → R_z(90°) @ [0.08,0,0] = [0,     0.08, 0]  ✓
  yaw=180° → R_z(180°)@ [0.08,0,0] = [-0.08, 0,    0]  ✓
```

### Side Grasp on a Cylinder

The fundamental challenge for cylinder side grasps: **the radial approach direction couples with yaw** in $B_w$ and cannot be encoded as a free DOF directly. Instead, `grasp_cylinder` generates `2*k` templates — `k` discrete approach depths × 2 roll orientations — each with the radial standoff baked into $T_w^e$.

```python
from tsr.hands import ParallelJawGripper

# A generic parallel-jaw gripper: 55 mm fingers, 140 mm max aperture.
# (For real hardware use the preconfigured classes, e.g. Robotiq2F140().)
gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

# Returns TSRTemplates: depth levels × 2 roll orientations
templates = gripper.grasp_cylinder(
    cylinder_radius=0.040,        # 4 cm mug radius
    cylinder_height=0.120,        # total height of cylinder
    reference="mug",
)

# Bind to detected mug pose and sample
mug_pose = perception.get_object_pose("mug")
grasp_poses = [t.sample(mug_pose) for t in templates]
```

`ParallelJawGripper` also supports `grasp_box`, `grasp_sphere`, and `grasp_torus`.

---

## Generating Placement Templates

`StablePlacer` generates one `TSRTemplate` per stable resting pose for objects placed on a flat surface. It uses the same TSR math as grasp templates — `Tw_e` encodes the stable orientation, `T_ref_tsr` the height at which the object frame origin rests, and `Bw` slides the object over the surface within its own footprint.

```python
from tsr.placement import StablePlacer

placer = StablePlacer(table_x=0.60, table_y=0.40)
```

### Analytic Primitives

For standard shapes, stable poses are computed analytically:

```python
# Cylinder: 2 templates (top face down, bottom face down)
templates = placer.place_cylinder(cylinder_radius=0.040, cylinder_height=0.120, subject="mug")

# Box: 6 templates (one per face — opposite faces are semantically distinct)
templates = placer.place_box(lx=0.08, ly=0.06, lz=0.18, subject="box")

# Sphere: 1 template; roll and pitch also free since all orientations are stable
templates = placer.place_sphere(radius=0.040, subject="ball")

# Torus: 2 templates (flat on each face, axis = z)
templates = placer.place_torus(major_radius=0.035, minor_radius=0.015, subject="ring")
```

Bw structure for all placement templates:
- `[±(table_x − footprint), ±(table_y − footprint)]` — the object slides anywhere it
  stays fully on the table. Yaw is free, so the inset is the object's *circumscribed*
  footprint radius (the half-diagonal of a box's resting face), not its half-extent.
- `z = [0, 0]` — exactly on the surface. The resting height itself lives in
  `T_ref_tsr`, where no rotation `Bw` admits can move it.
- `roll = pitch = [0, 0]` — fixed by the stable orientation (`Tw_e`), except for the
  sphere, which frees both because every orientation rests identically.
- `yaw = [-π, π]` — free rotation about the vertical axis

An object too large for the table is a valid request with no feasible pose, so the
factory returns `[]` and logs `exceeds_surface` at `DEBUG` — the same contract the
grasp factories follow. Non-finite or non-positive dimensions raise `ValueError`.

### Arbitrary Mesh

`place_mesh` handles any shape by computing the convex hull and finding which faces produce stable resting poses using the COM-projection criterion.

```python
import numpy as np
from tsr.placement import StablePlacer

placer = StablePlacer(table_x=0.60, table_y=0.40)

vertices = np.array([...])   # (N, 3) vertex cloud in object frame
com      = np.array([cx, cy, cz])

templates = placer.place_mesh(
    vertices, com,
    subject="widget",
    min_margin_deg=5.0,   # discard unstable faces (margin < 5°)
)
```

Results are sorted by descending stability margin (most stable face first). Each template carries `stability_margin` (in radians) — the minimum tipping angle before the object falls:

```python
table_pose = np.eye(4)
table_pose[2, 3] = 0.75   # table surface at z = 0.75 m

for t in templates:
    pose = t.sample(table_pose)
    print(t, "→ origin z =", round(pose[2, 3], 4))
    # TSRTemplate(task='place', subject='widget', variant='face-1', margin=31.9°)
```

**Stability criterion**: a face is stable if the COM projects strictly inside the support polygon — a COM exactly on a support edge is a tipping case, not a rest. The margin is the tipping angle `arctan(d_min / h_com)`, with `d_min` measured *in the face's own plane*, so it is a property of the object and the pose: the same rigid object re-expressed in a rotated frame, or in different units, reports the same angles.

Every factory takes `min_margin_deg`, not only `place_mesh`. It means the same thing everywhere, so a needle-thin box standing on its end (1.15°) is rejected by `min_margin_deg=5` whether you ask through `place_box` or `place_mesh`.

**Reading a template's semantics.** Every placement template carries a
`PlacementProvenance` (`t.provenance`): the primitive, the resting face's outward normal
in the object frame, `support_margin` and `com_height` — so
`stability_margin == atan2(support_margin, com_height)` can be re-derived rather than
taken on trust — `footprint_radius`, `equilibrium` (`'stable'`, or `'neutral'` for a
sphere, which rolls rather than tipping), and for a mesh `face_index` / `face_count` /
`facet_count`. `facet_count` is how many co-planar hull facets were merged into the
face: more than two means you handed over a finely tessellated surface.

**Tessellated curved surfaces.** `place_mesh` describes the polyhedron you give it, so a cylinder tessellated into `n` flat sides really does have `n` resting poses on those sides — it cannot know you meant a cylinder. Their margin is exactly `180/n` degrees, which goes to zero as the tessellation refines, matching the ideal cylinder that contacts along a line and rolls rather than tipping. Pass `min_margin_deg` above `180/n` and you get back exactly what `place_cylinder` returns: the two caps.

```python
verts = tessellated_cylinder(r=0.04, h=0.12, n=48)   # 48 side facets
placer.place_mesh(verts, com)                        # 50 poses: 2 caps + 48 sides at 3.75°
placer.place_mesh(verts, com, min_margin_deg=4.2)    # 2 poses: the caps, as place_cylinder
```

### Releasing into a Container

A container is entered through its opening, not placed on. `OpeningPlacer` takes the
opening as a rectangle's full extents (`size`) or a circle's `radius`, with its `depth`
from the rim to the floor, framed on the rim plane with z pointing out of the container.
Each template releases the object in one of the resting orientations `StablePlacer.place`
finds, so it falls as it rests:

```python
from tsr.placement import OpeningPlacer

placer = OpeningPlacer(size=(0.238, 0.238), depth=0.757)
templates = placer.place(
    shape,
    lowest=(0.02, 0.10),   # the object's lowest point, 2 to 10 cm above the rim
    clearance=0.026,       # how far inside the opening the whole object stays
)
```

Bw structure for release templates:
- `[±hx, ±hy]` — the object origin slides where the footprint, swept through every
  yaw, stays `clearance` inside the opening: `size/2 − clearance − footprint` for a
  rectangle, and for a circle the square inscribed in the circle of radius
  `radius − clearance − footprint`.
- `z = lowest` — the object's lowest point, from below the rim (negative, no lower than
  the floor) to above it. As for placements, the resting height of the origin is in
  `T_ref_tsr`.
- roll, pitch and yaw as the rest's: fixed by `Tw_e`, yaw free.

The container is taken to be the opening's section straight down to its floor. Where
the object comes to rest once let go is not part of the contract: check that in the
application.

---

## Visualization

The recommended viewer is `tsr.viser` — an interactive browser view, with sliders over a
template's free `Bw` coordinates so a continuous freedom can be scrubbed rather than
sampled:

```python
from tsr.viser import explore_templates, show_templates

server = show_templates(templates, cylinder=(0.04, 0.12), gripper=gripper, seed=0)
server = explore_templates(templates, cylinder=(0.04, 0.12), gripper=gripper)
```

`pip install "sstsr[viser]"`, then open <http://localhost:8080>. See
[VISER.md](VISER.md).

Install it with `pip install "sstsr[viz]"` (or the identical `sstsr[viser]`).

For debugging generation itself — change a primitive's dimensions, the gripper, `k` or
the clearance and watch the TSRs change, with the reason shown when a request yields
nothing — use the studio:

```python
from tsr.viser import studio

studio().sleep_forever()
```

## Practical Examples

### Example 1: Grasp a Mug with a Parallel Jaw Gripper

```python
from tsr.hands import ParallelJawGripper
import numpy as np

gripper   = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)
templates = gripper.grasp_cylinder(cylinder_radius=0.040, cylinder_height=0.120, reference="mug")

mug_pose   = perception.get_object_pose("mug")
candidates = [t.sample(mug_pose) for t in templates]
```

### Example 2: Generate All Stable Placements for a Mug

```python
from tsr.placement import StablePlacer
import numpy as np

placer    = StablePlacer(table_x=0.60, table_y=0.40)
templates = placer.place_cylinder(cylinder_radius=0.040, cylinder_height=0.120, subject="mug")

table_pose = perception.get_table_pose()
for t in templates:
    pose = t.sample(table_pose)
    print(t, "→ origin z =", round(pose[2, 3], 4))
```

### Example 3: Stable Placements for Arbitrary Shape (Mesh)

```python
from tsr.placement import StablePlacer
import numpy as np

# L-shaped object: 12 vertices
vertices = np.array([
    [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.05, 0.0], [0.05, 0.05, 0.0],
    [0.05, 0.1, 0.0], [0.0, 0.1, 0.0],
    [0.0, 0.0, 0.02], [0.1, 0.0, 0.02], [0.1, 0.05, 0.02], [0.05, 0.05, 0.02],
    [0.05, 0.1, 0.02], [0.0, 0.1, 0.02],
])
com = vertices.mean(axis=0)

placer    = StablePlacer(table_x=0.5, table_y=0.5)
templates = placer.place_mesh(vertices, com, subject="L-shape", min_margin_deg=5.0)

table_pose = np.eye(4)
table_pose[2, 3] = 0.75

for t in templates:
    pose = t.sample(table_pose)
    print(t)
```

### Example 4: Place Mug on Table (Direct TSR)

```python
from tsr import TSR
import numpy as np

table_pose = perception.get_table_pose()

place_tsr = TSR(
    T0_w=table_pose,
    Tw_e=np.eye(4),
    Bw=np.array([
        [-0.20, 0.20],    # x: 40 cm range on table
        [-0.15, 0.15],    # y: 30 cm range on table
        [0.0,   0.0 ],    # z: exactly on surface
        [0.0,   0.0 ],    # roll: upright
        [0.0,   0.0 ],    # pitch: upright
        [-np.pi, np.pi],  # yaw: any orientation
    ])
)

placement    = place_tsr.sample()
distance, _  = place_tsr.distance(proposed_pose)   # 0 if valid
```

### Example 5: Pour from Bottle (Trajectory-Wide Constraint)

A TSR used as a trajectory constraint keeps every waypoint in the valid region:

```python
from tsr import TSR

glass_pose = perception.get_object_pose("glass")

pour_tsr = TSR(
    T0_w=glass_pose,
    Tw_e=np.eye(4),
    Bw=np.array([
        [-0.02, 0.02],          # x: over glass
        [-0.02, 0.02],          # y: over glass
        [0.15,  0.25],          # z: 10 cm band above rim
        [-np.pi/4, -np.pi/6],   # roll: 30–45° tilt for pouring
        [0.0,   0.0 ],          # pitch
        [-np.pi, np.pi],        # yaw: any rotation around glass axis
    ])
)

# Pass to planner — every waypoint must satisfy pour_tsr
plan = cbirrt.plan(start=q_start, goal=q_goal, constraints=[pour_tsr])
```

---

## Summary

| Concept | Purpose |
|---------|---------|
| **$T_0^w$** | Positions the constraint in the world (reference entity's pose) |
| **$T_w^e$** | Nominal subject offset from TSR origin (encode radius/standoff here) |
| **$B_w$** | Allowed deviations — the actual constraint region |
| **TSR Chains** | Constraints on articulated objects (doors, drawers) |
| **TSRTemplate** | Serializable, reusable constraint bound at runtime |
| **ParallelJawGripper** | Programmatic grasp templates from object geometry |
| **StablePlacer** | Programmatic placement templates; one per stable pose |
| **OpeningPlacer** | Release templates into a container's opening; one per resting orientation that fits |
| **tsr.viser** | Interactive 3D inspection in a browser (`show_templates`, `explore_templates`, `studio`) |

TSRs are a single abstraction that covers grasping, placement, and trajectory constraints — the difference lies only in what $T_0^w$ refers to and how $B_w$ shapes the valid region.

For runnable scripts, see the [examples](../examples/) directory.

---

## References

1. Berenson, D., Srinivasa, S., Ferguson, D., & Kuffner, J. (2009). *Manipulation planning on constraint manifolds*. IEEE International Conference on Robotics and Automation.

2. Berenson, D., Srinivasa, S., & Kuffner, J. (2011). *Task Space Regions: A framework for pose-constrained manipulation planning*. International Journal of Robotics Research.
