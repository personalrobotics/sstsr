# Architecture

`tsr` is organised in four layers. Lower layers never import upper ones; the
boundary is enforced by `tests/tsr/test_architecture.py`.

| Layer | Module(s) | Role | Dependencies |
|-------|-----------|------|--------------|
| **0 — core** | `tsr.core` (`TSR`, `TSRChain`, math helpers) | The pure-math heart: SE(3) geometry, bounds, sampling, distance, serialisation. Robot-agnostic. | NumPy (SciPy only for `TSR.distance_optimize`, since #174 moved the chain's inverse to NumPy) |
| **1 — recipe** | `tsr.template` (`TSRTemplate`) | A reusable, serialisable, scene-agnostic recipe that instantiates to a concrete `TSR` at a reference pose. | Layer 0 |
| **2 — factories** | `tsr.hands` (`ParallelJawGripper`, …), `tsr.placement` (`StablePlacer`) | Domain knowledge that *produces* recipes from a described object. | Layers 0–1 |
| **3 — services** | `tsr.io`, `tsr.sampling`, `tsr.viser` (optional `viz`/`viser` extra) | Persist, select among, and inspect recipes/TSRs. | Layers 0–2 |

## The spine

> **A factory produces recipes; a recipe instantiates to math.**

```
gripper.grasp_box(...)  ->  [TSRTemplate]          # factory  (Layer 2)
template.instantiate(T_world)  ->  TSR             # recipe   (Layer 1 -> 0)
tsr.sample() / contains() / distance()             # math     (Layer 0)
```

If a piece of code isn't (a) the math, (b) a recipe, (c) a factory that emits
recipes, or (d) a service that consumes them, it's misplaced.

## Core consistency contract (Layer 0)

The heart's correctness *is* its mutual consistency. For any `TSR` and pose:

- `sample()` returns a pose that `is_valid()` accepts and `contains()` confirms.
- **Membership is exact** (#190). `contains(T)` and `is_valid` hold iff the coordinates lie
  within `Bw`, up to round-off only: `length_atol(scale)` for translation, with `scale` the
  summed lengths of `T0_w`, `Tw_e` and the coordinates (the lengths composed into the pose,
  not the region's width), and `ANGLE_ATOL = 1e-6` rad for rotation. There is no further
  slack. Before #190 every bound carried a fixed `1e-3` — as wide as a millimetre-scale region
  — and `distance` jumped from 0 to about `1e-3` at its edge.
- `contains(T)` is true ⇔ `distance(T)[0] == 0`. Outside, `distance` grows continuously from
  the boundary: a pose `2e-4` past a bound is not contained and measures `2e-4`.
- **Acceptance tolerances belong to the caller.** A planner that accepts a pose "close
  enough" compares `distance` against its own tolerance; the region does not widen itself
  for it. The one tolerance the library owns beyond round-off is `CHAIN_TOLERANCE = 1e-3`,
  the default residual below which a chain's inverse solve reports `"satisfied"` — a solver
  tolerance, never applied to a single region's membership.
- For a pose the region contains, `to_xyzrpy(T)` returns coordinates that `is_valid()`
  accepts, and recomposing them returns `T`. This is a real constraint rather than a
  restatement: RPY double-covers rotations — `(r, p, y)` and `(r ± π, −π − p, y ± π)` are
  the same rotation — so "some representative is in bounds", which `contains` establishes,
  and "the representative we return is in bounds" are different claims. Violating it made
  `to_xyzrpy` contradict the same region's `is_valid` for a quarter of contained poses
  (#171).
- For a pose the region does **not** contain, `distance(T)[0]` is the *minimum*
  displacement over every equivalent RPY representative, not merely over one wrapping of
  them. RPY is periodic and double-covering, so each angle has infinitely many
  representatives and a rotation has nine equivalent triples; the measured one must be the
  nearest to the bound interval. Wrapping each angle into `[lower, lower + 2π)` instead
  sent an angle just *below* a bound narrower than a full turn almost a full turn above the
  upper bound, which overstated 41% of the conformance corpus's recorded distances — by up
  to 10× — and reported a pose 0.006 rad outside a zero-width bound as about π away (#186).
  Wrap relative to the interval **centre**: the distance to `[lo, hi]` is
  `max(0, |x − c| − w/2)`, monotone in `|x − c|`, so the representative nearest the centre
  is the one nearest the interval.
- The RPY ↔ rotation ↔ transform conversions round-trip (including near gimbal
  lock — see `docs/REVIEW.md` C1).

These are exercised exhaustively by the Hypothesis property tests.

**These rules have a second implementation.** `cpp/` is a C++ port of Layer 0's region and
chain, for a planner that cannot afford a Python call per edge sample. The Python stays the
source of truth: the port is held to it by a checked-in corpus of the Python's own answers,
and neither implementation reads the other. So a change to a rule here is a change to two
implementations and to the corpus between them — in that order. See
[CPP.md](CPP.md).

### Construction rejects a malformed region (#162)

`TSR.__init__` raises `ValueError` when a frame is not a finite 4×4 whose last row is
`[0, 0, 0, 1]` and whose rotation block satisfies `R Rᵀ = I` and `det R = +1`, or when
`Bw` is not a finite 6×2, or when a translation row has `lo > hi`. A **rotational** row
with `hi < lo` is valid: it is an outer interval wrapping through ±π, which the
continuous-bounds construction expands correctly.

The tolerance is `tsr.FRAME_ATOL = 1e-6`, absolute, and it is public because the
contract is shared. pycbirrt's native TSR runtime is checked differentially against
this implementation and applies the same rules at the same tolerance, so what one
accepts the other must accept; a second implementation is why the rules are stated as
conditions on the inputs rather than as properties of this code.

Rejecting here is what keeps a malformed region diagnosable. Accepted, it surfaces much
later and in a misleading place — a NaN distance, a sample nothing can use, or a
projection that never converges.

### Chain composition, and the frame a chain will not read (#166)

A `TSRChain` composes serially (Berenson et al. 2011 §5.1). The **first** link is placed
by its own `T0_w`. Every **later** link is placed *by the chain*: its frame **is** the
previous link's end frame, `C_{i-1}.T0_w · Tw(ξ_{i-1}) · C_{i-1}.Tw_e`. So a later
link's own `T0_w` never participates, and `TSRChain.append` — the single path every
constructor and `from_dict` funnels through — raises `ValueError` when one is not the
identity within `FRAME_ATOL`.

Ignoring it instead is the worst of the options: the offset disappears and the chain
still produces a plausible pose, wrong by exactly that offset. The reported case was a
door-handle grasp whose template `T_ref_tsr` became the second link's `T0_w`; the grasp
landed 8 cm low, nothing was logged, and planning failed somewhere unrelated.

**Where an offset belongs.** To sit a later link's region a fixed distance `S` from the
previous link's end frame, post-multiply it into the **previous** link's `Tw_e`
(`Tw_e ← Tw_e · S`) — that is the transform which moves the end frame the next link
hangs off, and it reproduces `… · Tw_e · S · Tw(ξ) · …` exactly. The order matters, so
it is tested rather than only documented.

## Factory contract (Layer 2)

Every `grasp_*` / `place_*` factory method obeys one uniform contract:

> **An exception reports an invalid request. An empty list reports an empty
> feasible set.**

- **`raise ValueError`** only when an argument is nonsensical on its own — a
  non-positive size, `k < 1`, a reversed `angle_range`. These indicate a bug in
  the caller and must never be silently swallowed.
- **Return `[]`** whenever valid arguments simply admit no grasp — the object is
  wider than the jaws, or a positive-but-large clearance leaves no band. A 6 mm
  cylinder and a 0.5 m clearance are both *valid*; they just yield an empty
  feasible set. Treating one physical infeasibility differently from another
  would recreate the inconsistency this contract removes. Callers can sweep many
  objects without `try`/`except`.
- **Observability:** infeasibility carries a machine-readable reason
  (`exceeds_aperture`, `cannot_straddle`, `insufficient_clearance_band`,
  `finger_too_short`, `closed_reach_too_long`). The *public* factory method emits a
  single `logger.debug` at its boundary (not at each internal early return), so an
  empty result is diagnosable without turning it into an exception. If debug logs prove
  insufficient, add an explicit diagnostic-result API — exceptions do not carry
  that responsibility.
- Templates carry `task` / `subject` / `reference` metadata and the canonical
  gripper frame convention (`z` = approach, `y` = finger opening, `x = y × z`).

## Geometric grasp contract (primitive parallel-jaw grasps)

The factory contract above governs *when* a method returns templates. This
section governs *what a returned grasp template geometrically means*. It is the
executable specification the `#67` analytic oracle and the `#73` Hypothesis
matrix test against (issue `#66`).

### Three assurance layers

A generated grasp template can be validated at increasing strength. Each layer
is a strictly stronger claim than the one above it, and each is tested
separately — a template may attain one without the next:

1. **Representation validity** — the recipe and instantiated `TSR` are
   well-formed: finite `T_ref_tsr`/`Tw_e`/`Bw`, orthonormal rotation, ordered
   bounds, `sample() ⇔ contains() ⇔ distance≈0`. (Established in 2.1.0.)
2. **Geometric soundness** — *every* pose admitted by a template returned by a
   primitive grasp factory is a kinematically viable parallel-jaw pre-grasp
   (clauses below).
3. **Coverage** — every approach family the API advertises is represented,
   subject to explicitly documented discretization by `k` (depths) and `n_minor`
   (torus minor angles). A statement about *which modes exist*, not about any one
   template.

These factory guarantees do not extend automatically to hand-authored
`TSRTemplate` recipes. The YAML examples packaged with sstsr are checked for
representation validity, but omit the primitive dimensions and gripper model needed
by the analytic oracle. Their authors and users remain responsible for geometric
feasibility in the intended scene.

**Coverage is not soundness, and neither is a sampling distribution.** In
particular the sphere-grasp TSR *covers* SO(3) as a set, but sampling independent
uniform roll/pitch/yaw (`TSR.sample`) is **not** Haar-uniform on SO(3). Haar-uniform
sampling over a TSR's rotation box is a separate API, `tsr.sampling.sample_haar`,
which draws `sin(pitch)` uniformly (the ZYX Haar density is `∝ cos(pitch)`) (#72).
Distribution claims require their own proof and are never implied by set coverage.

### Soundness clauses

For a primitive solid `S`, finger length `L`, max aperture `A`, requested
preshape `a`, clearance `c`, and **every** pose admitted by a returned template:

1. All inputs, transforms, and bounds are finite and well formed.
2. `0 < a ≤ A`, and `S`'s relevant span along the closing line is strictly
   smaller than `a` (the object fits *between* the inner pad faces).
3. The palm lies outside `S` with at least the promised clearance `c`.
4. The usable finger-depth interval is nonempty (a positive `c` that leaves no
   band yields `[]`, never a reversed depth sequence).
5. Closing along the canonical `±y_EE` directions meets two opposing surfaces of
   `S` within the usable depth interval.
6. The contact normals oppose the closing directions within tolerance.
7. The whole continuous TSR region (`Bw` extrema, midpoint, and interior)
   preserves clauses 2–6.
8. The template declares its primitive, mode, hand-occupied approach, object-frame
   finger orientation, insertion depth (index/value), and symmetry **structurally**,
   via `TSRTemplate.provenance` (`GraspProvenance`). Tests and oracles read that
   record; they never infer semantics by parsing `name`.

This describes a viable *pre-grasp*. It is **not** a claim of dynamic stability
or force closure under arbitrary friction — an explicit non-goal.

**Closing must not undo it** (#193). A linkage hand's pads swing forward as it closes,
so closed on a span `w` its fingertips reach `ParallelJawGripper.reach(w) ≥ L` past the
palm. Wherever a depth limit faces the far side of what the fingers span — the box top,
bottom and face bands, the cylinder caps and side, the sphere, the torus side — the
fingertips stay the promised margin inside that far side closed as well as open: the
limit is `extent − m − (reach(w) − L)`. A family that this alone empties reports
`closed_reach_too_long`. The oracle above models the open hand only, which every closed
template also satisfies; `tests/tsr/hands/test_closed_reach.py` checks the closed one.

**A default margin scales with the object** (#195). When the caller passes no `clearance`,
it is `clearance_fraction` of the depth the fingers can use: `min(L, r)` for the cylinder
side, sphere and torus side, `min(L, extent)` for a box family (the box's extent along that
family's approach), and `L` for the cylinder caps and the torus span. A supplied
`clearance` is used exactly. Box templates record the margin they used, and its basis, in
`provenance.metadata["margin"]` and `["margin_basis"]`.

### Tolerances and conventions (defined once)

- Length tolerance is scale-aware: `atol = 1e-9 + 1e-6 · scale`, where `scale`
  is the primitive's characteristic dimension. Angles use `1e-6 rad`.
- Contact-normal opposition holds when the angle between the inward surface
  normal and the closing direction is `≤ 1e-6 rad`.
- **The palm and closing axes are defined at the *concrete* end-effector pose,
  not only at `Bw = 0`.** For a TSR coordinate `ξ ∈ Bw`, the end-effector frame in
  the reference frame is `E(ξ) = T_ref_tsr · xyzrpy_to_trans(ξ) · Tw_e` (and
  `T_ref_world · E(ξ)` in the world frame after `instantiate`). The palm is the
  translation of `E(ξ)`; the approach and closing axes are its `z`/`y` columns.
  The oracle evaluates soundness at a concrete `ξ` this way — clause 7 quantifies
  over all `ξ` in the region, so the palm is a function of `ξ`, not the fixed
  `Tw_e` translation. "Outside `S` with clearance `c`" means signed distance
  `≥ c − atol`.
- Frames follow the library convention: `z_EE` = approach, `y_EE` = finger
  opening (fingers always close along `±y_EE`), `x_EE = y_EE × z_EE`.
- **Straddle feasibility is decided on the margin, with slack** (#107, #129). The
  object fits between the open jaws when the margin `preshape − span` is at least
  `4·atol`, i.e. each pad clears the object by `2·atol` — twice the `atol` at which the
  oracle's clause 2 admits a pose, leaving one tolerance of slack per side. Two points matter. First the
  test is on the *margin*, which is exact by Sterbenz, not on `preshape` versus
  `span + 4·atol`: when the margin is orders of magnitude smaller than the span, that
  addition loses it and a margin under the floor is accepted. Second the floor is
  *twice* the oracle's tolerance, because at a margin of exactly `2·atol` the oracle's
  clause-2 bound evaluates to the contact coordinate itself and certification is
  decided by rounding — measured at ~59% of sampled poses failing on a sphere. The
  realized margin is quantized by `ulp(span)`, so that, not `ulp(margin)`, is the
  resolution at which the boundary can be probed. A **default** preshape is therefore
  built by `_default_preshape`, which advances `span + c` until the realized margin is
  at least the requested `c`: a public clearance request is met by the nearest
  representable geometry rather than weakened by rounding (#131). An explicitly
  supplied preshape is never altered — it is judged by its own realized margin.
- **Edge-facing band limits use `m = max(c, 2·atol)`** (#121). A limit that faces an
  edge of the primitive — the approached face, the opposite face or cap, a lateral
  slide edge, a cylinder rim — keeps contacts strictly off that edge, where the
  surface normal is ambiguous and the insertion would be degenerate. `c = 0` is a
  valid request (#104), and a clearance below `atol` is indistinguishable from it, so
  the generator floors these limits. The floor is *twice* `atol` because the pose's
  realized insertion is recovered by a cancelling subtraction, so exactly `atol`
  loses to rounding. This mirrors the scale-aware straddle margin, under which the
  object must clear each pad by `2·atol`. Palm clearance and the default preshape still
  use the requested `c`.

### Structured provenance — three separate layers (#84)

**Provenance declares intent; the pose determines geometric truth.** Three
responsibilities are kept apart so the public record does not become a second
grasp DSL:

1. **`GraspProvenance`** (value object) — a small, immutable, **extensible**
   record of the generator's claim. It validates only *representation*
   invariants (nonempty string labels; finite `depth ≥ 0` canonicalized to float;
   exact integer `depth_index`/`depth_count` with `0 ≤ index < count`; JSON-scalar
   `metadata` frozen into a read-only mapping; exact dict/JSON/YAML round-trip).
   Labels are free strings — it does **not** know which `(primitive, mode)` pairs
   exist or how fields relate, so third-party generators can use their own
   vocabulary. Fields:

   | field | frame | meaning |
   |---|---|---|
   | `primitive`, `mode` | — | free labels selecting the analytic model |
   | `depth` | approach axis | insertion depth from the approached surface [m], `≥ 0` |
   | `approach` | object | side/family the **hand occupies** (not the sign of `z_EE`) |
   | `finger_orientation` | object | direction the pads are separated along — object axis (box) or yaw-free family `tangential`/`diameter` (fingers always close along `±y_EE`) |
   | `depth_index`/`depth_count` | — | index (shallow → deep) and number of **distinct** depth levels in the template's *depth family* (`≤ k`; a one-point band gives `depth_count = 1`, an empty band emits nothing) (#81). A depth family is the templates **within one factory result** whose non-depth fields — `primitive`, `mode`, `approach`, `finger_orientation`, `symmetry`, `metadata` — are identical; it holds one template per depth level. The counters do not identify templates across independently generated or concatenated collections. |
   | `symmetry` | — | distinguishes otherwise-equivalent templates (e.g. a roll flip) |
   | `metadata` | object | descriptive extras only — **never** geometric evidence (torus `minor_*`, box `slide_axis`/`span`) |

2. **Native conformance** (`tsr.hands._conformance.validate_builtin_provenance`)
   owns the sstsr vocabulary and cross-field relational checks the value object
   does not: the built-in `(primitive, mode)` set, allowed `approach`/
   `finger_orientation`/`symmetry` labels, required `metadata`, and relations
   (box `finger_orientation ≠ slide_axis`; box-face orientation lies in the face
   plane; torus `minor_index` in range). Built-in factory tests run it over every
   emitted template; it may evolve without closing the public type.

3. **Geometric truth** — the analytic oracle (`#67`) derives contacts, reach,
   clearance, and the realized depth/minor-angle **from the concrete pose and
   primitive geometry**, and must **not** use `provenance.depth`, a recorded
   minor angle, or other metadata as inputs to the calculation that certifies
   those same values. `primitive`/`mode` select the analytic model; everything
   else is checked against the pose, not trusted. A provenance failure means the
   generator described its output inconsistently; an oracle failure means the
   emitted pose is geometrically wrong.

Built-in modes emitted today (native conformance vocabulary):

| primitive | modes | finger_orientation | symmetry / metadata |
|---|---|---|---|
| cylinder | `side`, `top`, `bottom` | `tangential` (side), `diameter` (top/bottom) | side: `roll0`/`rollpi` |
| box | `top`, `bottom`, `face` | object axis `x`/`y`/`z` | `metadata.slide_axis`, `metadata.span` |
| sphere | `surface` (full SO(3)) | `diameter` | — |
| torus | `side`, `span` | `tangential` (side), `diameter` (span) | side: `flip0`/`flippi`, `metadata.minor_*` |

### Worked examples

Each snippet is numerically complete: the gripper fixes `L` (finger length) and
`A` (max aperture); the default `preshape a = span + c` and `clearance c = 0.1 ·
graspable_depth` are shown where they matter, so soundness/infeasibility follows
from the numbers alone. `g = ParallelJawGripper(finger_length=L, max_aperture=A)`.

- **Cylinder** — *sound:* `L=0.08, A=0.14`; `g.grasp_cylinder_side(0.03, 0.12)`.
  Diameter `2r = 0.06 < a ≈ 0.06 + c`; `L=0.08 > r=0.03` so the palm at radial
  standoff `ro = r + L − depth` stays outside the surface. *Infeasible:*
  `g.grasp_cylinder_side(0.10, 0.12)` → `[]` with reason `exceeds_aperture`
  (`2r = 0.20 > A`). (Separately, `L < r` must not place the palm inside — `#69`.)
- **Box** — *sound:* `L=0.055, A=0.14`; `g.grasp_box_top(0.05, 0.06, 0.07)` — for
  span-x the pads straddle `box_x=0.05 < A` and slide along y. *Infeasible
  orientation:* `g.grasp_box_top(0.20, 0.06, 0.07)` drops the span-x orientation
  (`box_x=0.20 > A`) but keeps span-y (per-face, per-orientation; `#70`).
- **Sphere** — *sound:* `L=0.08, A=0.14`; `g.grasp_sphere(0.03)` (mode `surface`,
  full SO(3); `2r=0.06 < a`). *Infeasible:* `g.grasp_sphere(0.10)` → `[]`,
  `exceeds_aperture` (`2r=0.20 > A`).
- **Torus** — *sound:* `L=0.08, A=0.30`; `g.grasp_torus_side(0.06, 0.02)`
  (tube diameter `2r=0.04 < a`). *Infeasible span:* `g.grasp_torus_span(0.06, 0.02)`
  on `A=0.14` → `[]`, because `2(R+r)+c = 0.16+ > A` (outer diameter exceeds the
  jaw). Torus coverage semantics (`n_minor`, inner/outer half) are refined in `#71`.

## Geometric placement contract (stable placements on a flat surface)

The counterpart of the grasp contract for `StablePlacer`. It governs what a returned
*placement* template geometrically means, and it is the specification the independent
placement oracle (`tests/tsr/placement/_placement_oracle.py`) and the placement matrix
(`_placement_matrix.py`) test against (issue `#148`).

### Frames and what a pose places

- The **surface frame** has `z` up and its origin at the centre of the surface, so the
  surface is the plane `z = 0` and spans `±table_x`, `±table_y`.
- The **object frame** is the caller's own. A returned pose places *its origin*: the
  geometric centre for the primitives, and an arbitrary point for `place_mesh`, which
  takes the centre of mass separately and does not assume the two coincide. Poses are
  not centre-of-mass poses; `#148` records the documentation that once said they were.
- `table_x` / `table_y` bound the **object**, not just its origin.

### Soundness clauses

For an object `O`, its centre of mass `com`, and **every** pose admitted by a returned
template — not merely the midpoint of `Bw`:

1. **Resting** — `O` touches the surface and does not penetrate it: the lowest point of
   the posed body is `z = 0` within `atol`.
2. **Supported** — `com` projects inside the convex hull of the contact patch (the
   points of `O` at the lowest `z`). A sphere's point contact lies directly under its
   centre and is the neutral boundary case.
3. **On the surface** — the whole of `O` lies within the surface footprint, at every
   yaw the region admits.
4. **Described** — a reported `stability_margin` is the physical tipping angle: the
   angle through which `O` must rotate about the nearest support-polygon edge before
   `com` passes over it. It is a property of the object and the pose, so it is
   invariant under rigid motion of the object and under a change of the object frame,
   and scale-free.
5. **Declared** — the template declares its primitive, the resting face's outward
   object-frame normal, the `support_margin` and `com_height` the tipping angle is built
   from, the `footprint_radius` that set the `Bw` inset, whether the equilibrium is
   stable or neutral, and for a mesh its face and merged-facet counters —
   **structurally**, via `TSRTemplate.provenance` (`PlacementProvenance`). Tests and
   oracles read that record; they never infer semantics by parsing `name`.

This describes a *statically stable* resting pose. It is **not** a claim about
dynamics, friction, or that the object will survive being let go from above it.

### Construction rules that make the clauses structural

- **The resting height lives in `T_ref_tsr`, and `Tw_e` is a pure rotation** (`#149`).
  A pose is `T_ref_tsr · xyzrpy_to_trans(ξ) · Tw_e`, so a height carried by `Tw_e` is
  rotated by `ξ`'s roll and pitch: freeing either sank the object (the sphere's centre
  followed `r·cos(roll)·cos(pitch)`, so every admitted pose but a measure-zero set was
  below the surface). With the height ahead of `ξ`, the resting height is invariant
  under every rotation `Bw` admits, and a primitive that frees roll or pitch — a tilt
  tolerance, a sphere, a cone on its side — cannot silently violate clause 1.
- **The `xy` bounds are inset by the object's footprint radius** (`#150`): the largest
  horizontal distance from the object-frame origin to any point of the resting object.
  Yaw is free, so that footprint sweeps a *disc*; the inset is the circumscribed
  radius, not an axis-aligned half-extent. For a box this is the half-diagonal of the
  resting face — with a per-axis inset a `0.20 × 0.08 m` box still overhung by 0.112 m
  at `yaw = 0.464`. Equality is feasible and leaves an exact zero-width interval, as
  elsewhere in the library.
- **An object that cannot fit is an empty feasible set**, not an unusable region: the
  factory returns `[]` and logs `exceeds_surface` once at its boundary, per the factory
  contract above. `place_mesh` also reports `no_stable_face` and `below_min_margin`.
- **A `variant` that names a face is a claim about that face**: for the named face's
  outward object-frame normal `n`, every admitted pose satisfies `R(ξ) · n = −z`.
  Opposite faces are emitted separately (the box's six, the cylinder's two caps, the
  torus's two sides) because which face is down is caller-visible and semantically
  distinct for a labelled object — a mug's opening, a cereal box's front — even where
  the bare solid is invariant under the flip that separates them (`#155`).
- **The tipping angle is measured in the face's own plane** (`#151`). `d_min` is the
  distance from the projected centre of mass to the nearest support edge, in an
  *orthonormal* basis of that plane. Projecting by dropping the normal's dominant axis
  foreshortens it by `|n_max| ≥ 1/√3`, which made the reported angle depend on how the
  object frame happened to be oriented — under-reported by up to 40%, and different for
  the same rigid object in a rotated frame. Every primitive's margin is the same
  quantity in closed form: `arctan(min(a,b)/l)` for a box face, `arctan(r/(h/2))` for a
  cylinder cap, `arctan(R/r)` for a torus lying flat.
- **A sphere reports `0`, and it means neutral, not critical.** Point contact directly
  beneath the centre never tips; it rolls. A mesh face reporting `0` would instead be a
  knife-edge equilibrium, which is why those are excluded rather than reported.
- Length tolerance is the same scale-aware `atol = 1e-9 + 1e-6 · scale` as the grasp
  contract, defined once in `tsr.core.utils.length_atol`. **Containment is decided on a
  length and fails closed** (`#153`): the old test compared a cross product — an *area* —
  against an absolute `1e-10`, so below ~1e-5 m every edge test was skipped and the
  helper fell through to "inside", inventing stable poses at small scales. A face is
  stable only when the projected centre of mass is inside by more than `atol`; a centre
  of mass exactly on a support edge is a critical equilibrium, not a rest.

### Assurance

Two layers, as on the grasp side. `_placement_oracle.py` certifies a *concrete* posed
object using only support functions of the geometry the caller supplied — the lowest
point is `t_z − h(−z)`, the reach in a horizontal direction is `t·d + h(d)` — so it
never reuses a construction formula and a template cannot certify itself.
`test_placement_soundness.py` drives every public `place_*` factory through it at the
`Bw` midpoint, every free dimension's extrema, all corners and interior points, under
an arbitrary surface pose, and a guard test fails if a new factory is added without
joining the matrix.

The margin is checked the same way: `_placement_oracle.tipping_angle` reads the contact
patch and the centre of mass off the *pose*, so it never sees the generator's
arithmetic, and the primitives are additionally checked against closed forms and against
routing the same solid through `place_mesh`.

### Structured placement provenance — the same three layers (#160)

The placement counterpart of clause 8's `GraspProvenance`, and the same separation:
**provenance declares intent; the pose determines geometric truth.**

1. `PlacementProvenance` (`tsr/placement_provenance.py`) validates *representation*
   only — non-empty labels, finite non-negative lengths, a length-3 non-zero
   `face_normal`, integer mesh counters present together. Labels stay free strings, so a
   third-party placement generator can use its own vocabulary.
2. `tsr.placement._conformance` owns the sstsr vocabulary and the relational checks:
   only a sphere may omit a face normal; `equilibrium` is `"neutral"` **iff**
   `support_margin` is exactly zero; and a template's `stability_margin` really is
   `atan2(support_margin, com_height)` — recomputed with `math.atan2`, never with the
   generator's own helper, so corrupting that helper cannot corrupt both sides.
3. The oracle derives resting, support and containment from the pose and never treats
   the record as evidence.

Two fields exist to resolve things that were previously indistinguishable. `com_height`
is the **lever arm**, not the origin height; they coincide only when the frame origin is
the centre of mass, which is why an off-centre mesh is what exposes a confusion between
them. `equilibrium` separates a sphere's *neutral* rest — point contact, it rolls rather
than tips — from a knife-edge equilibrium about to fall, both of which otherwise read as
`stability_margin ≈ 0`.

**Wire format.** A serialized `provenance` block carries a `kind` (`"grasp"` or
`"placement"`); **absent means `"grasp"`**, so every record sstsr wrote before #160
still reads. An unrecognised `kind` raises `ProvenanceKindError`, which is deliberately
*not* a `ValueError`/`KeyError`/`TypeError` — `tsr.io.load_templates_from_directory`
skips those with a warning, which is right for a truncated file and wrong for a record
from a newer version, since it would turn a version-skewed directory into a silently
empty list. Both `from_dict`s use bracket access on their required keys, so neither
record type can be misread as the other. This is a cross-implementation contract now
that pycbirrt's native runtime is checked differentially against this implementation.

**Scalars only.** No support polygon and no support area: area is 0 for a torus's line
contact and a sphere's point contact although the torus is the most stable pose the
library emits, so it is not comparable across factories, and the polygon is
variable-width, chart-dependent, and reconstructible by a caller from their own vertices
plus the pose.

### What `place_mesh` takes a face to be (#152)

Two rules that pull against each other, and the tolerance between them is where the
tessellation question lives.

- **A face is a face however it was triangulated.** Hull facets lying in one plane are
  one resting face, grouped by *proximity* — normals within `1e-6` and offsets within
  `atol` — because the facets of a single face agree only to floating point. Grouping on
  a rounded normal instead, as this replaces, fragments a face whenever its facets
  straddle a rounding boundary: a rotated box whose vertices came from float32 split into
  twelve single-triangle faces, none able to support the centre of mass. That is the
  ordinary case, not a corner one — STL, OBJ, glTF and MuJoCo all store float32.
- **Nearly parallel is not the same plane.** A tessellated curved surface has genuinely
  distinct facets, and merging them would invent a flat face the caller never supplied.
  A cylinder cut into `n` sides separates neighbouring facets by `2π/n`, which stays
  ~600× above the tolerance even at `n = 10000`.

So `place_mesh` describes **the polyhedron it was given**, and a 48-gon prism is a
48-gon prism: the library cannot know it was meant to be a cylinder. That is why it
returns resting poses on a tessellated curved side while `place_cylinder` documents that
sideways is not stable — and the two are reconciled by the margin, not by a special
case. A cylinder tessellated into `n` sides rests on each at exactly `180/n` degrees,
which tends to 0 as the tessellation refines: the ideal cylinder's true answer, since it
contacts along a line and rolls rather than tipping. A caller says "this is a curved
surface" with `min_margin_deg` above `180/n`, and then the two entry points agree
exactly, which is a test rather than a claim.

## Verification matrix and release gate (#73)

The per-primitive soundness suites (`test_reach_soundness.py`, `test_box_soundness.py`,
`test_torus_soundness.py`, …) are the deterministic regressions. On top of them,
`tests/tsr/hands/_grasp_matrix.py` drives **every** public `grasp_*` factory — the
combined entry points and the named grippers included — through the independent
oracle. A guard test fails if a new factory is added without joining the matrix.

**Cases.** Primitive dimensions are drawn relative to the hand's reach and aperture
across scale regimes, together with `k`, `n_minor`, restricted yaw and minor-angle
ranges, and explicit or default preshape and clearance. `boundary_cases` additionally
solves one clearance that places a boundary **active for that factory** exactly, then
offsets it by ±1 ulp. Solving for the clearance rather than the gripper keeps the
named hardware in the matrix.

**Boundaries are behavioural, not just biased** (#126). A random case sitting near a
boundary proves little: an empty result satisfies soundness, equivariance, symmetry,
serialization and uniqueness *vacuously*, so a closed interval silently becoming open
(the #124 bug class) would escape. `BOUNDARIES` therefore pins each documented
boundary with dimensions that leave every unrelated constraint slack, and asserts the
**transition** of the family it governs across below / exact / above — empty on the
infeasible side, non-empty and oracle-sound on the feasible side and, where the
interval is closed, at the boundary itself. All dimensions are dyadic multiples of a
scale (small `1e-3`, ordinary `1`, large `1e3`), so identities like `h − h/2 == h/2`
hold exactly in binary and the `nextafter` neighbours really do straddle the
comparison the generator makes:

| boundary | active for | closes at |
|---|---|---|
| cap band (#122) | cylinder top/bottom | `c = h/2` (short cylinder, `h < L`) |
| side height band (#124) | cylinder side | `c = h/2` |
| radial reach | sphere, cylinder side, torus side | `c = L − r` (`L < 2r`) |
| radial far surface | sphere, cylinder side, torus side | `c = r` (`2r ≤ L`) |
| box approach band | each box face | `c = min(L, extent)/2` |
| box slide band (#110) | box top | `c = dim/2` |
| torus span reach | torus span | `c = L − r` |
| aperture limit | box top | `c = A − span` |
| straddle floor (#107, #129) | explicit preshape | margin `= 4·atol(scale)` |

The straddle floor is pinned on the **explicit-preshape** path, where the realized
margin `preshape − span` is ulp-exact, and its boundary value is certified like any
other. Through the *default* preshape the margin is built by `span + c`, so realized
margins are quantized by `ulp(span)` — far coarser than `ulp(c)` — and the meaningful
neighbour is one step of that lattice, not one ulp of the clearance (#131).
`tests/tsr/hands/test_straddle_margin.py` exercises that public path across scales and
every factory's own span and scale (#133).

**Invariants.** Each is a pure check returning failures, shared with the gate:

| # | invariant |
|---|---|
| 1 | every pose (Bw midpoint, per-axis extrema, corners, interior points) has an oracle witness, and its provenance passes native conformance |
| 2 | invalid arguments raise `ValueError`; physical infeasibility returns `[]` |
| 3 | equivariance: `instantiate(T)` maps every pose by `T` and preserves the witness |
| 4 | symmetries: rotation about the object axis, and the x/y/z mirrors, preserve the witness under the mapped side labels; declared opposite-side pairs have equal depth multisets |
| 5 | monotonicity: more reach or a wider aperture never removes a family |
| 6 | declared families are all emitted; structured keys are unique; a combined entry point equals the union of its parts — matched by structured key (emission order is not contractual), then compared as whole templates via `to_dict()`, so semantic and display fields (`subject`, `reference`, `name`, …) are covered too (#127) |
| 7 | dict/JSON/YAML round-trips preserve poses and the witness |

**Mutation gate** (`test_grasp_mutation_gate.py`). A suite that never fails proves
nothing, so each bug class named in #73 is injected and the same checks must reject
it: approach sign, face origin, axis mapping, object extent, clearance term and reach
as black-box mutants over the factory output, plus code mutants that patch the real
generator (`_resolve_clearance`, `_usable_depths`, `_infeasibility_reason`).
Detection is required per (mutant, factory), aggregated over a deterministic corpus —
one case suffices, since a mutation can be harmless in one geometry and fatal in
another. Two exclusions are explicit, and both are *correctness* statements:

- rotating a **sphere's** TSR frame is a symmetry, so the axis-mapping mutant stays
  sound (`KNOWN_SYMMETRIC`, re-certified rather than skipped);
- **inflating** an object is not by itself unsound — a grasp planned for a larger
  object can execute soundly on the real, smaller one — so the extent mutant shrinks.

The corpus includes a deliberately short-fingered hand (`L < 2r`), because the radial
deep limit `min(L, 2r) − c` only *binds* there; with long fingers it is conservative
and exceeding it is still a sound grasp that no check may reject.

**Runtime.** Budgets scale with `TSR_MATRIX_SCALE` (default 1): the matrix and gate
add ~13 s to a ~41 s suite. A separate `.github/workflows/release-gate.yml` runs them
at `TSR_MATRIX_SCALE=10` weekly and on demand (`workflow_dispatch`). It is its own
workflow, with its own concurrency group, so a scheduled run does not also launch the
five-version suite and a push to `main` can neither cancel a running gate nor be
cancelled by one (#128). Run it locally with

```bash
TSR_MATRIX_SCALE=10 uv run pytest tests/tsr/hands/test_grasp_matrix.py tests/tsr/hands/test_grasp_mutation_gate.py
```

Hypothesis runs with `deadline=None` and prints a reproduction blob on failure, so
results are stable across Python 3.10–3.14. A confirmed counterexample is shrunk and
promoted to a deterministic regression next to the primitive it belongs to.
