# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The Robotiq hands' fingertips reach further as they close, and their grasp templates now
allow for it (#193). Both are four-bar hands whose pads swing forward on closing: the 2F-140
closed on a 72 mm object reaches about 132 mm past the palm, not its open `FINGER_LENGTH` of
114 mm. A template that kept the *open* fingertips a clearance inside an object's far face
could drive the *closed* ones through it — into the table, for a top grasp of a short box.
That is what Geodude's lifts were starting in collision with: a top grasp of `gelatin_box`
(72 × 88 × 28 mm) closed across 72 mm puts the fingertips about 18 mm further down, into
the worktop.

Minor rather than patch: there is a new API, and **the 2F-140's and 2F-85's templates
change**. `FrankaHand` and every `ParallelJawGripper` built from dimensions are unchanged:
they declare no curve, so their reach is `finger_length` at every width, as before. Pin
`sstsr>=4.1` if you rely on Robotiq templates keeping their closed fingertips inside the
object.

**What changes for the Robotiq hands.** Only the deep end of a depth band, and only where it
faces the far side of what the fingers span: it moves shallower by `reach(span) −
finger_length`. Shallow depths, palm limits, slide bands, preshapes and frames do not move.
For `Robotiq2F140()` at its default clearance (11.4 mm for a box, 3.3 mm for this cylinder):

| call | before | after |
|---|---|---|
| `grasp_box_top(0.072, 0.088, 0.028)` | 6 templates, depths 11.4–16.6 mm | none: `closed_reach_too_long` |
| `grasp_box_top(0.158, 0.028, 0.084)` | 3, depths 11.4 / 42.0 / 72.6 mm | 3, depths 11.4 / 30.3 / 49.1 mm |
| `grasp_cylinder_side(0.033, 0.123)` | 6, depths 33.0 / 47.9 / 62.7 mm | 6, depths 33.0 / 38.5 / 44.0 mm |

The gelatin box still has top grasps at a smaller clearance — up to 5.2 mm across its 72 mm,
7.0 mm across its 88 mm — which is the consumer's choice to make. Radial grasps need a radius beyond the closed-reach gain: with
default clearance the 2F-140 side-grasps cylinders and spheres from 24 mm radius, the 2F-85
from 15 mm, where both used to accept any radius their aperture allowed.

### Added
- **`ParallelJawGripper.reach(width)`** (#193): how far the fingertips reach past the palm
  along the approach, closed to `width` — the gap between the pads' contact faces, the same
  quantity as `max_aperture` and a template's span. It is `finger_length` unless the hand
  declares a measured curve as **`CLOSED_REACH`**, which it interpolates linearly, holds at
  its ends outside the measured widths, and never lets fall below `finger_length`.
- **`Robotiq2F140.CLOSED_REACH` and `Robotiq2F85.CLOSED_REACH`**, measured from geodude_assets
  `2f140.xml` (395991d) and mujoco_menagerie `robotiq_2f85/2f85.xml` (feadf76) under MuJoCo
  3.14.0. Fully open each reads its own `MAX_APERTURE` and `FINGER_LENGTH` back, to within
  0.4 mm. The curves are the models'; neither has been checked against Robotiq's drawings.
- **`tools/finger_reach.py`**, which measures such a curve from a hand's MJCF, so the numbers
  above can be regenerated rather than trusted.
- **A new empty-set reason, `closed_reach_too_long`**: the open hand fits, and only the
  closed fingertips would pass the far face. `insufficient_clearance_band` keeps its meaning,
  an empty band for the open hand.

### Changed
- **Every depth limit that faces the far side of what the fingers span holds for the closed
  fingertips too** (#193): it is `extent − m − (reach(span) − finger_length)` rather than
  `extent − m`. That covers box top, bottom and both face families (`span` is the box
  dimension the fingers close across), cylinder top, bottom and side (`span` is the
  diameter), and the sphere and torus side, whose radial band has the same far-surface limit
  as the cylinder side. The torus span grasp is unchanged: its fingers pass outside the
  ring, so no far face bounds its depth.
- **A box family emptied in different ways per orientation reports the one nearest to
  feasible**: `closed_reach_too_long`, then `cannot_straddle`, then
  `insufficient_clearance_band`. Before, any orientation with a slide band read as
  `cannot_straddle`.
- **Tests.** A property test checks, from the poses alone and for each named hand over a grid
  of boxes, cylinders and spheres, that the palm stays a clearance outside the near face and
  the fingertips a clearance inside the far face, open and closed
  (`tests/tsr/hands/test_closed_reach.py`). Three existing tests were sized for a prismatic
  jaw: `test_outputs_canonical_frames` compares the 2F-140 with a plain jaw that now carries
  the same curve, the comfortable regime in `test_declared_families_are_all_emitted` grows a
  radial object's radius past a linkage hand's closed-reach gain, and the mutation-gate
  corpus's torus is 22 mm / 18 mm instead of 30 mm / 10 mm, so the 2F-85 can still side-grasp
  its tube.

## [4.0.0] — 2026-10

`TSR.contains` no longer accepts poses up to 1 mm (and 1 mrad) outside a region, and
`TSR.distance` no longer jumps from 0 to about `1e-3` at the edge of that band (#190).

Major rather than minor: `contains` verdicts change, not only numbers. A pose that was
accepted inside the old band is now refused, and a downstream conformance corpus generated
from 3.x no longer matches. Upgrade deliberately — `sstsr>=4,<5` — and regenerate any such
corpus when you do.

**Migration: set your own tolerance where you decide acceptance.** If you relied on
`contains` accepting a pose just outside `Bw` — a planned waypoint, an executed pose, a
projection that converged to within `1e-3` — compare `distance(T)[0]` against a tolerance
you choose instead. sscbirrt's `membership_tolerance` is already that tolerance.

### Changed
- **Membership is exact, up to round-off** (#190). `contains` and `is_valid` used to grant a
  fixed `EPSILON = 1e-3` on both sides of every bound, a 2015 round-off guard that was
  absolute rather than scaled: as wide as Geodude's ±2 mm approach line. Planned waypoints
  then read distance 0 while up to 1 mm outside `Bw`, and a few-µm joint change moved them
  across the band's edge to a distance just over sscbirrt's `1e-3` tolerance
  (personalrobotics/sscbirrt#269). Now the only allowance is round-off: the contract's
  scale-aware `length_atol` for translation (scaled by the lengths composed into the pose)
  and `ANGLE_ATOL = 1e-6` rad for rotation. A pose `2e-4` past a bound is not contained and
  its distance is `2e-4`; `distance` is continuous through the old band, and
  `contains ⇔ distance == 0` still holds.
- **The gimbal-lock band of RPY extraction is narrowed from `|R[2,0]|` within `1e-9` of 1 to
  within `1e-14`.** Inside the old band pitch was snapped to ±π/2, discarding up to
  `4.5e-5` rad — hidden by the `1e-3` slack, and exposed by removing it: every pose of a region
  with pitch fixed `1e-5` under π/2 read as outside it. Snapping now costs at most about
  `1.4e-7·π` rad, under `ANGLE_ATOL`. `to_xyzrpy` and `distance` change in the old band
  accordingly.
- **The chain's solve tolerance has its own name**: `CHAIN_TOLERANCE` (C++
  `kChainTolerance`), still `1e-3`, is the default residual below which `TSRChain.solve`
  reports `"satisfied"`. `EPSILON` and `kEpsilon` remain as deprecated aliases of it (the C++
  one marked `[[deprecated]]`); neither is a containment slack any more.

### Added
- `tsr.ANGLE_ATOL` and `tsr.CHAIN_TOLERANCE`; C++ `sstsr::kAngleAtol`,
  `sstsr::length_atol` and `sstsr::kChainTolerance`.
- Conformance corpus: a sweep at `0`, `2e-4`, `9e-4` and `1.1e-3` past an upper bound in
  every region, an `approach_line` region, and a `pitch_in_old_gimbal_band` region (11
  regions, 282 probes).

## [3.5.0] — 2026-10

`TSR.distance` was measuring from the wrong rotation representative, and reporting poses just
outside a narrow rotation bound as about π away instead of millimetres.

Pin `sstsr>=3.5` if you use `distance` or `closest_transform` for membership, projection or
validation. The numbers change: they are smaller and correct, never larger, and a `contains`
verdict never moves — but 41% of the conformance corpus's recorded distances were overstated,
some by 10×, so a threshold tuned against the old values will behave differently. That is why
this is a minor release rather than a patch.

Found twice in the field, from opposite directions: measuring where Geodude's hand ended after
executing a planned TSR goal (`ssrobot#129`), and a CI run where ADA's articutool joints, which
sit outside the planned arm group, settled after execution and pushed the fork tip just under
its roll bound (`ssrobot#131`). Both are now regression tests.

### Fixed
- **`TSR.distance` no longer overstates how far a pose just under a rotation bound is** (#186).
  The displacement formula wrapped each angle into `[lower, lower + 2π)` before measuring it
  against the bounds. An angle slightly *below* a bound narrower than a full turn is a full
  turn away from `lower`, so it landed at almost `lower + 2π` — far above the upper bound —
  and none of the nine equivalent-RPY candidates could recover the true small displacement. A
  pose 3 mm and 0.006 rad outside a region measured **4.441** (about `π√2`) instead of 0.0077.

  Angles are now wrapped relative to the interval **centre**, which is the representative
  nearest the bounds: the distance to `[lo, hi]` is `max(0, |x − c| − w/2)`, monotone in
  `|x − c|`, so nearest-to-centre is nearest-to-interval. Verified to never *increase* a
  reported distance — over 20 000 random angle/interval draws, 72% improved and none
  worsened.

  This was not an edge case. It overstated **41% of the conformance corpus's recorded
  distances**, by up to 10× (`5.661` → `0.549`). `contains` is unaffected and its verdicts are
  unchanged everywhere: it wraps with its own `EPSILON`-shifted origin, and `distance`
  short-circuits to exactly `0.0` for a contained pose, so the `contains ⇔ distance == 0`
  invariant still holds.

  Consumers using `distance` for membership, projection or validation saw a pose just outside
  a narrow rotation bound as about π away instead of millimetres — in a planner that makes
  projection fail where it should succeed, and in a checker it refuses a valid near-boundary
  state. Found downstream in `personalrobotics/ssrobot#129`, measuring where Geodude's hand
  ended after executing a planned TSR goal.

  The rule is now stated in `docs/ARCHITECTURE.md`'s Layer 0 consistency contract, the C++
  core follows it, and the corpus was regenerated — 66 of 162 probes moved.

## [3.4.1] — 2026-10

Metadata only; no code changes.

The repository is renamed `personalrobotics/tsr` → `personalrobotics/sstsr`, matching the
distribution name it has published under since 2.0. The import name is still `tsr` and the
distribution is still `sstsr`; nothing a caller writes changes.

This release exists because a project's PyPI page is rendered from its latest release's
metadata, and PyPI does not allow editing a published release. So the links and the rendered
README on <https://pypi.org/project/sstsr/> kept pointing at the old repository path until a
new version shipped. GitHub redirects the old path, so nothing was broken — but the front
door named the wrong place.

### Changed
- `Homepage`, `Repository` and `Issues` in the package metadata, the README's clone command,
  and a release-tag link in this file now name `personalrobotics/sstsr` (#183).

## [3.4.0] — 2026-10

Two changes to how a chain's inverse behaves and how the viewer reads a template, neither of
which moves a rule.

The chain's cold inverse is now the **same projected Levenberg–Marquardt** the C++ core runs,
replacing L-BFGS-B over a finite-difference gradient. The objective is unchanged — it was
always `‖r‖²` for a 12-vector in disguise — so the switch buys recall and cost rather than new
semantics: 98.6% against 98.2% at 7.8× fewer objective evaluations and 3.1× faster, measured by
`tools/chain_solver_comparison.py`, which is checked in so the claim stays re-runnable.
`ChainSolveResult` values move, and its documented contract does not: `residual` is still an
upper bound, `"not_found"` still no proof of non-membership (#85). Every exact path is
untouched, so the conformance corpus is byte-identical.

Both implementations sharing one algorithm still does **not** make the cold inverse recordable
in that corpus — a redundant chain's solution set is a continuum, so two solvers that both
succeed agree on the coordinates about 1% of the time, and both are right. `docs/CPP.md` says
so plainly, because the opposite is the natural assumption.

And the viewer no longer raises on placement templates, which it had advertised as supported
input since the placement factories arrived.

### Fixed
- **The viewer accepts placement templates** (#167). `tsr.viser` advertised them as supported
  input while every site that read provenance assumed a `GraspProvenance`, so
  `explore_templates(placer.place_box(...))` raised
  `AttributeError: 'PlacementProvenance' object has no attribute 'mode'`. Five read sites now
  dispatch on the `KIND` wire discriminator #160 introduced, rather than on the record's class,
  so a third-party record is described generically instead of refused.

  A placement is labelled from its own vocabulary — the resting face, the tipping angle, and
  whether the equilibrium is stable or neutral — not a depth family it does not have. `mode_of`
  groups it by **face normal**: an analytic box carries no `face_index`, so grouping on the
  index would collapse all six faces into one entry. `filter_templates(..., depth_index=...)`
  is a grasp idea, so no placement matches it, and it returns nothing rather than raising.

  A **mixed** grasp/placement collection is refused, naming what it holds. The controls differ
  per kind — a depth slider means nothing for a placement, a tipping angle nothing for a grasp
  — so drawing one with half the controls inert would be the quietly-wrong behaviour this
  library rejects elsewhere (#166). Grasp labels and grouping are unchanged.

  The bug predates #160, which only changed the shape of the failure. `docs/VISER.md` gains a
  placement section.

- **A chain test no longer pins SciPy's trajectory.**
  `test_deterministic_counterexample_witness_certifies_membership` asserted that the cold solve
  **fails** on a pose that is provably a member — true of L-BFGS-B, and the reason that test
  could never cross over to the C++ port. The new solver finds it, so the test now asserts the
  property it was always protecting: a retained witness decides membership with no optimiser
  involved. Its companion no longer patches `scipy.optimize.fmin_l_bfgs_b` to inspect the
  optimiser's start, which tested SciPy's call signature rather than this library; it reads the
  canonicalised first start off the public result instead.

### Changed
- **The chain's cold inverse is projected Levenberg–Marquardt with an analytic Jacobian**
  (#174), the same solver the C++ core uses, replacing L-BFGS-B over a finite-difference
  gradient. The objective is unchanged, and the least-squares form is exact rather than an
  approximation of it: since `3 − tr(AᵀB) = ‖A − B‖²_F / 2` on SO(3), the scalar
  `Δt·Δt + (3 − tr(RᵀR′))` *is* `‖r‖²` for the 12-vector `r = [Δt, (R − R′)/√2]`. One O(n)
  prefix/suffix sweep builds all `6n` Jacobian columns, so a gradient costs no extra chain
  composition where it used to cost `6n`.

  Measured over 560 cold solves on 70 random chains at equal budget: **98.6% recall against
  98.2%, with 7.8× fewer objective evaluations and 3.1× faster wall-clock**.
  `tools/chain_solver_comparison.py --against-scipy` reproduces it and fails if recall ever
  regresses.

  **`ChainSolveResult` values move.** `status`, `coordinates`, `residual`, `nfev` and `starts`
  all come from a different search, and `nfev` now counts this solver's residual evaluations
  rather than SciPy's objective calls — the same unit (one forward composition) but not
  comparable numbers. The documented contract is unchanged and still holds: `residual` is an
  upper bound rather than a certified distance, `"not_found"` is no proof of non-membership
  (#85), `nfev <= max_nfev` strictly and `starts <= max_starts`. Every exact path is untouched,
  so the conformance corpus is byte-identical and `max_starts == 0` still reports the
  un-optimised midpoint with `nfev == 0`.

  Sharing the algorithm with the C++ deliberately does **not** make the cold inverse
  recordable in the conformance corpus: two solvers that both succeed recover the same
  coordinates about 1% of the time, because a redundant chain's solution set is a continuum
  and both answers are correct. `docs/CPP.md` records why.

  SciPy leaves `tsr.core.tsr_chain` entirely. It remains a package dependency — the placement
  factories need `scipy.spatial.ConvexHull`, and `TSR.distance_optimize` still uses it.

## [3.3.0] — 2026-10

TSR chains exist in C++. `sstsr` now ships a second implementation of both the pose region and
the chain, so a planner that evaluates a constraint on every edge sample no longer pays a Python
call to do it. Measured end to end on `sscbirrt`'s door-opening example, which plans the same set
twice: 18.52 s as a `TSRChain` against 1.01 s as the equivalent single native TSR.

The chain is specified in two halves, because only one of them can be. The forward paths and
every exact `solve` — an empty chain, a validating witness, a single link, an all-fixed chain,
`max_starts == 0` — are held to the Python's recorded answers by a checked-in corpus. The cold
inverse is held to **properties** instead: it reports the best point a bounded search found, and
a different optimiser evaluates a different set of points, so `"not_found"` remains no proof of
non-membership in either implementation. `docs/CPP.md` is the whole contract.

The Python stays the source of truth, and writing the second implementation found a defect in
the first: `TSR.to_xyzrpy` was checking rotation candidates against the *translation* bounds, so
it returned coordinates the same region's `is_valid` rejected for a quarter of contained poses.

Pin `sstsr>=3.3` for the C++ chain, and consume it from the installed wheel with
`find_package(sstsr_cpp CONFIG REQUIRED)` — only the wheel states a version.

### Added
- **`TSRChain` in the C++ core: the forward half and every exact solve path** (#165, stage 2a).
  `sstsr/tsr_chain.hpp` adds `TSRChain`, `ChainSample`, `ChainSolveResult` and `SolveOptions`
  beside the pose region from stage 1, so a planner can compose regions, sample a chain, and
  check membership natively. Sampling returns the coordinates that built the pose, and those are
  a *constructive witness*: `validate_witness` certifies membership with one bounds check and one
  composition, no optimiser. A `solve` handed a validating witness takes the exact warm path —
  `nfev == 0`, `starts == 0` — which is the case a planner actually hits, since a chain used as a
  path constraint along an edge has the neighbouring state's coordinates to hand. The empty,
  single-link, all-fixed and `max_starts == 0` paths are exact too. `geodesic_distance` joins
  `sstsr/transform.hpp`.
- **The conformance corpus covers chains** (#165). `tests/reference/tsr_conformance.json` gains 8
  chains with 72 coordinate probes, 48 witness probes and 18 exact solves, and
  `tools/tsr_conformance.py --check` now walks them — it previously walked only the regions, so a
  chain regression would have passed silently. Residual comparisons use a `1e-7` floor rather
  than `1e-9`, because the geodesic's rotation term is `acos` of a value near 1 and a 1-ulp trace
  error there is reported as `2.1e-08` of angle; the old floor would have failed on a different
  libm while nothing was wrong.
- **The corpus also records `to_xyzrpy` for every region probe.** `to_xyzrpy` picks between the
  two RPY representatives of the same rotation, and that choice is invisible in `contains`,
  `distance` and `closest_transform` — which is exactly how #171 stayed hidden behind 162
  agreeing probes. Recording it, with whether the result satisfies the region's own `is_valid`,
  closes that blind spot for both implementations.

- **The C++ chain's cold inverse** (#165, stage 2b), so a planner no longer needs a witness to
  hand. It is **projected Levenberg–Marquardt with an analytic Jacobian**, and the least-squares
  form is exact rather than an approximation of the Python's objective: since
  `3 − tr(AᵀB) = ‖A − B‖²_F / 2` on SO(3), the Python's scalar `Δt·Δt + (3 − tr(RᵀR′))` *is*
  `‖r‖²` for the 12-vector `r = [Δt, (R − R′)/√2]`. One O(n) prefix/suffix sweep builds all `6n`
  Jacobian columns, so a gradient costs no extra composition — where the Python pays `6n` of them
  to finite-difference one.

  Two details are load-bearing, each shown by ablation rather than asserted: the Jacobian must be
  analytic, and the step needs gradient projection (a coordinate pinned at a bound whose gradient
  pushes further out is held fixed). Removing the projection takes the SO(3)-ball fixture from
  40/40 to 17/40 and quadruples the evaluations elsewhere. The solver finds the deterministic
  counterexample of issue #85 — a provable member the Python's cold solve reports as
  `not_found` — on its first start, at a residual of `1.4e-17`.

  The cold path stays specified by **properties**, not by recorded numbers, and that does not
  change just because this solver is better: `not_found` is not a proof of non-membership in
  either implementation.

- **Every placement template carries a structured `PlacementProvenance`** (#160). Grasp
  templates have declared their semantics structurally since 2.2.0, and clause 8 of the
  geometric contract forbids inferring semantics by parsing `name` — but placement left
  a caller no alternative, since the resting face's normal, the lever arm behind the
  reported tipping angle, and the footprint radius that set the `Bw` inset lived only
  inside a free-text name or nowhere at all. The record carries the primitive, the
  object-frame face normal, `support_margin` and `com_height` (so
  `stability_margin == atan2(support_margin, com_height)` is auditable rather than merely
  asserted), `footprint_radius`, `equilibrium`, and for a mesh its face and merged-facet
  counters. Same three layers as the grasp side: a representation-only value object that
  stays open to third-party vocabularies, `tsr.placement._conformance` for the sstsr
  vocabulary and the relational rules, and an oracle that never trusts either.
- **`equilibrium` separates neutral from knife-edge.** A sphere contacts at a point and
  *rolls*; a mesh face with a vanishing margin is about to tip. Both read as
  `stability_margin ≈ 0`, and nothing distinguished them before.
- **`tsr.ProvenanceKindError`.** A serialized `provenance` block now carries a `kind`;
  absent means `"grasp"`, so every record sstsr has written still reads. An unrecognised
  kind raises this rather than a `ValueError`, because `load_templates_from_directory`
  skips those with a warning — right for a truncated file, wrong for a record from a
  newer version, which would otherwise make a directory load silently come back empty.

- **A C++ core with a CMake package** (#165). `cpp/` is a second implementation of the
  pose region, for a planner that needs TSRs in C++ without a Python call per edge
  sample. It moved here from `sscbirrt`, which had been carrying it: a second
  implementation of sstsr's contract was living outside sstsr, so every rule change here
  had to be followed there — 3.1 → 3.2's construction validation was one.

  Standard library only, C++20; a transform is `std::array<double, 16>` and the generator
  is `std::mt19937_64`, so a consumer inherits no linear-algebra dependency and the
  exported config needs no `find_dependency`. The wheel stays `py3-none-any` and carries
  the sources and headers rather than a binary, so the release pipeline and install story
  are unchanged. `tsr.get_cmake_dir()` and `tsr.get_include()` are how a C++ consumer
  finds it; `docs/CPP.md` has the `find_package` recipe, and `cpp/examples/` has working
  consumers for both the installed package and the one inside a wheel.

  The Python stays the source of truth for the rules. The C++ is held to them by a
  checked-in corpus of the Python's answers — 162 probes across 9 regions, `contains`
  exactly, lengths to 1e-9 — and the constants it restates are checked against the
  Python's, since a drifting tolerance fails much later than it is introduced.

  Chains arrived in the two entries above. `docs/CPP.md` records what a port can and cannot
  promise, which is what split them: the forward and witness paths are exactly specifiable, the
  cold inverse is not, because "best found" is taken over an optimiser's trajectory including its
  finite-difference probes.
- **`TSR.volume`** and **`TSR.continuous_bounds`**, which make the measure and the
  continuous chart part of the region's public surface rather than a private attribute the
  conformance generator reached into. `tsr.sampling._interval_sum` now delegates to
  `TSR.volume`, so the measure has one definition for both implementations.

### Fixed
- **Only the wheel states an `sstsr_cpp` version now** (#175). The two packagings rendered the
  CMake package version independently — a source install from a literal in
  `cpp/CMakeLists.txt`, the wheel from the release version — so they drifted: 3.2.0 against
  3.2.1. A consumer pinning `find_package(sstsr_cpp 3.3)` would have got different answers from
  a checkout and from the wheel it installs, which passes in development and fails on install.
  `project()` now declares no `VERSION` and a source install ships no `ConfigVersion` file, so
  that call succeeds from the wheel and fails with `version: unknown` from a checkout rather
  than answering staleley. This also removes the single exception to `docs/RELEASING.md`'s
  invariant that there is no version number in the tree to edit.

- **`TSR.to_xyzrpy` no longer returns coordinates its own `TSR` rejects** (#171). It passed the
  full six-row `_Bw_cont` to `rot_within_rpy_bounds`, but that helper reads rows 0–2 of
  whatever bounds it is handed — so the roll/pitch/yaw candidates were being checked against
  the **x/y/z translation** bounds. The check then almost always failed and the method fell
  back to the raw `rot_to_rpy` extraction. RPY double-covers rotations, and `rot_to_rpy`
  always reports `|pitch| <= π/2`, so for any region whose pitch bounds lie outside that range
  the returned triple was the *other* representative of the same rotation — out of bounds.
  `to_xyzrpy` therefore contradicted `contains` and `is_valid` for **25.9%** of contained poses
  (measured over 4000 random regions). It propagated to `TSRChain.solve`'s single-link path and
  from there to `TSRChain.distance`, `to_xyzrpy` and `closest_transform`. The two other call
  sites, `is_valid` and `contains`, always sliced correctly; now all three do.

  It hid because the conformance corpus records `contains`, `distance` and `closest_transform`
  but never `to_xyzrpy` directly, so #170's 162 probes could not see it. It surfaced while
  building the chain corpus for #165, when the C++ port — which reads the rotation rows, as
  `contains` does — disagreed on a single-link exact solve.
- **A `TSRChain` no longer accepts a frame it will not read** (#166). A chain places
  every link after the first on the previous link's end frame, so a later link's own
  `T0_w` never participates — but one could be passed, and was silently dropped. The
  offset then disappeared while the chain still produced a plausible pose, wrong by
  exactly that amount: the reported case was a door-handle grasp whose template
  `T_ref_tsr` became the second link's `T0_w`, landing the grasp 8 cm low with nothing
  logged and planning failing somewhere unrelated. `append` — which every constructor
  and `from_dict` funnels through — now raises `ValueError` naming the offset it will not
  use and where it belongs: post-multiplied into the **previous** link's `Tw_e`, which is
  what moves the end frame the next link hangs off. The first link is unaffected; it is
  placed by its own `T0_w`.

### Documentation
- **The README covers the C++ core**, which until now appeared nowhere a user would look:
  `docs/CPP.md` was not linked from anything, and the README never mentioned `sstsr_cpp`,
  `get_include()`, `get_cmake_dir()`, or that a chain has a C++ form. There is now a **From
  C++** section with the `find_package` recipe and a compiling example, and the documentation
  index lists the architecture, C++, viewer and releasing guides rather than only the tutorial.
- **The chain section explains membership**, which is where a chain differs from a TSR: the
  constructive witness (`sample_with_witness`, `validate_witness`), the composition rule that
  makes a later link's `T0_w` unreadable, and the standing caveat that a bounded solve's
  `False` is not a proof of non-membership and `distance` is an upper bound (#85).
- **`docs/ARCHITECTURE.md` states the `to_xyzrpy` invariant** alongside the rest of the Layer 0
  consistency contract, and records that those rules now have a second implementation in `cpp/`
  which is held to them by the corpus — so changing a rule means changing two implementations
  and the corpus between them, in that order.

### Changed
- `tsr.placement._stable_poses.stable_poses_mesh` yields a `StableFace` record instead
  of a 3-tuple (internal; it is not exported).

## [3.2.0] — 2026-09

A TSR now rejects a malformed region at construction instead of letting it surface
later as a NaN distance or a projection that never converges.

The reason it matters beyond this package: pycbirrt is building a native C++ TSR
runtime checked *differentially* against this implementation, so sstsr is now a
reference implementation and its construction rules are a shared contract rather than
an internal detail. That is why the tolerance is exported and why the rules are stated
in `docs/ARCHITECTURE.md` as conditions on the inputs — which argument, which
condition, which tolerance — so a second implementation can reproduce them without
reading the Python. Pin `sstsr>=3.2` to rely on them.

### Added
- **`TSR.__init__` rejects a malformed region** (#162). A frame must be a finite 4×4
  whose last row is `[0, 0, 0, 1]` and whose rotation block satisfies `R Rᵀ = I` and
  `det R = +1`; `Bw` must be a finite 6×2; translation rows must be ordered. Previously
  only the last of these was checked, so a non-finite frame, a scaled or reflected
  rotation block, or a NaN in `Bw` was accepted and surfaced much later as a NaN
  distance, an unusable sample, or a projection that never converged. A **rotational**
  row with `hi < lo` remains valid — it is an outer interval wrapping through ±π.
- **`tsr.FRAME_ATOL`**, the absolute tolerance those frame checks use (`1e-6`). It is
  public because the contract is shared: pycbirrt's native TSR runtime is checked
  differentially against this implementation and applies the same rules at the same
  tolerance (personalrobotics/pycbirrt#87).

- **A checked-in placement mutation gate** (#159). 3.1.0's claim that corrupting the
  placement generator fails the suite was a measurement taken once, from a throwaway
  plugin; nothing in the repository held it. Eleven mutants now run against a
  deterministic corpus through the same check functions the property matrix uses, and
  the weekly release gate covers the placement suites at `TSR_MATRIX_SCALE=10`
  alongside the grasp ones. Writing it down caught an error in the original
  verification: `TSRTemplate` is frozen, so the variant-relabelling mutant had been
  *raising* rather than being detected.

Additive validation on inputs that were already unusable, so no working caller changes.

## [3.1.0] — 2026-09

Placement templates now mean what they say. 2.2.0 did this for grasps (#65); this does
it for placement (#148), and it found the same class of defect — a region admitting
poses that violate the property it claims — in every factory.

The visible consequences: a sphere no longer sinks through the table, no admitted pose
hangs off the surface, `stability_margin` is the physical tipping angle rather than a
number that changed when you rotated the object, every factory reports that margin and
takes `min_margin_deg`, and a mesh loaded from a file keeps its stable poses instead of
losing them to float32 rounding.

Two behaviour changes worth reading before upgrading: `Bw`'s xy extents are now
narrower than `table_x`/`table_y` by the object's footprint, and factories return `[]`
where they previously returned unusable templates. Both are listed under Fixed.

### Fixed
- **Freeing roll or pitch no longer sinks the object** (#149). `StablePlacer` carried
  the resting height in `Tw_e`'s translation, which a pose's own roll and pitch rotate:
  `place_sphere` — the one factory that frees them, and advertises it — put the sphere's
  centre at `r·cos(roll)·cos(pitch)`, so every admitted pose but a measure-zero set
  penetrated the surface, at `roll = π` entirely below it. The height now lives in
  `T_ref_tsr`, ahead of the region's rotation, and `Tw_e` is a pure rotation, so the
  resting height is invariant under every rotation `Bw` admits. Poses are unchanged for
  the templates that fix roll and pitch.
- **Placement regions no longer admit poses hanging off the surface** (#150). The `xy`
  bounds slid the object's *origin* over the full surface extent, so at the region's
  edge half a box hung off, and 61% of uniformly sampled placements overhung. Bounds
  are now inset by the object's footprint radius — the circumscribed radius, because
  yaw is free, so a per-axis inset would still overhang by the half-diagonal. An object
  too large for the surface now returns `[]` and logs `exceeds_surface`, per the
  factory contract, instead of a region no pose of which is valid.
- **`stability_margin` is the tipping angle it is documented to be** (#151). The
  COM-to-edge distance was measured in a projection that dropped the dominant axis of
  the face normal, foreshortening it by up to `1/√3`: margins were under-reported by up
  to 40%, and the *same rigid object* re-expressed in a rotated frame reported different
  angles — a physical quantity that depended on an arbitrary frame choice. Measured in
  an orthonormal basis of the face's own plane, it now reproduces the analytic angle
  exactly: a regular octahedron reports 35.264° rather than 22.21°, and a box reports
  the same three angles in every one of 3000 random object frames. This also corrects
  the "most stable first" ordering and the `min_margin_deg` threshold, both of which
  were computed from the wrong number.
- **Stability containment is scale-invariant and fails closed** (#153). The test
  compared a cross product — an *area* — against an absolute `1e-10`, so below ~1e-5 m
  every edge test was skipped and the helper fell through to "inside": an obtuse
  tetrahedron whose centroid provably projects outside its bottom facet was reported as
  resting on it, with a fabricated 86° margin that ranked it the most stable face.
  Containment is now decided on an in-plane *length* against the scale-aware `atol`, and
  an indeterminate test rejects the pose. A centre of mass exactly on a support edge is
  a critical equilibrium and is no longer returned as a rest.
- **A mesh with float32 vertices no longer loses its stable poses** (#152). Co-planar
  hull facets were grouped by an 8-decimal rounding of the face normal, which splits a
  face whenever its facets straddle a rounding boundary. A rotated 0.20 × 0.10 × 0.30 box
  whose vertices came from float32 fragmented into twelve single-triangle "faces", none
  able to support the centre of mass: 3.0.0 answered with twelve templates all claiming
  a 0° margin — each admitted by the default filter — and with containment now failing
  closed it would answer with none at all. This is the ordinary path for a mesh loaded
  from a file, since STL, OBJ, glTF and MuJoCo all store float32. Facets are now grouped
  by plane *proximity*, and the float32 box reports the same six angles as its float64
  twin. The tolerance is deliberately tight: a tessellated curved surface keeps its
  facets, because `place_mesh` describes the polyhedron it was given and cannot know a
  48-gon prism was meant to be a cylinder.
- **Non-finite placement input is rejected at ingress** (#154). `nan` and `inf`
  dimensions passed `x <= 0` and produced entirely finite templates, leaving no trace
  of the invalid request; `place_mesh` reported a non-finite vertex as a degenerate
  hull, and `min_margin_deg=nan` as an empty feasible set. Dimensions, surface extents,
  `vertices`, `com` and `min_margin_deg` now raise `ValueError` naming the argument.

### Added
- **An independent placement oracle and soundness matrix** (#148). The placement
  counterpart of the #67/#73 grasp work: `tests/tsr/placement/_placement_oracle.py`
  certifies a concrete posed object from support functions of the caller's own
  geometry, never from a construction formula, and `test_placement_soundness.py` drives
  every public `place_*` factory through it at the `Bw` midpoint, every free
  dimension's extrema, all corners and interior points, under an arbitrary surface
  pose. A guard test fails if a new factory skips the matrix.
- **The geometric placement contract** in `docs/ARCHITECTURE.md`: frames, what a pose
  places, surface-extent semantics, four soundness clauses, the construction rules that
  make them structural, and the open deviations still tracked under #148.
- **Every placement template reports a `stability_margin`, and every factory takes
  `min_margin_deg`** (#156). The primitives were opaque where `place_mesh` was
  self-describing, so a caller could not tell a 1.15° knife-edge rest from a 45° one, or
  reject it. The values are analytic — `arctan(min(a,b)/l)` for a box face,
  `arctan(r/(h/2))` for a cylinder cap, `arctan(R/r)` for a torus lying flat — and
  routing the same box through `place_box` and `place_mesh` now gives the same margins.
  A sphere reports `0`: it is neutrally stable, so it rolls rather than tipping, and any
  positive threshold rejects it.

### Changed
- **`place_mesh` and `place_cylinder` no longer appear to contradict each other** on a
  tessellated cylinder, and the relationship is now documented and tested rather than
  left to the reader. A cylinder cut into `n` sides rests on each at exactly `180/n`
  degrees — a margin that tends to 0 as the tessellation refines, which is the ideal
  cylinder's own answer, since it contacts along a line and rolls rather than tipping.
  Passing `min_margin_deg` above `180/n` returns exactly what `place_cylinder` returns.
- `place_torus`'s docstring said it returned one template while it returned two
  (#155). Both sides are returned, as for the box's faces and the cylinder's caps,
  because which side is down is caller-visible in `variant` and distinct for a labelled
  object; the documented count now matches, and the rule is stated once in the contract.
- Placement documentation called a placed pose a "COM z" throughout, a leftover from
  the C2 fix that was off by half an object for a mesh whose origin is not its centre.
  It is the object frame origin. `README.md` also claimed `place_box` returns "up to 3
  poses"; it returns six.

## [3.0.0] — 2026-09

**Breaking.** The PyVista backend is removed. `sstsr[viz]` now installs Viser, so an
existing install keeps working and gets the supported backend; code importing
`tsr.viz` must move to `tsr.viser` — see `docs/MIGRATION-VISER.md`, which maps every
removed workflow. Pin `sstsr<3` if you need browser-free PNG rendering, which has no
replacement.

Visualization is now one thing: an interactive viewer you interrogate, rather than a
picture of samples. `studio()` closes the loop for development — change a primitive,
the gripper or the clearance and watch the TSRs change, with the factory's own reason
shown when a request yields nothing.

Nothing about grasp geometry changed in this release: the 2.2.0 correctness work and
its oracle-backed matrix are untouched, and `pip install sstsr` remains
visualization-free.

### Removed
- **The PyVista backend** (#140), after shipping deprecated in 2.3: `tsr.viz`,
  `TSRVisualizer`, every `*_renderer`, and `GripperBase.renderer()` / the
  `ParallelJawGripper` override that reached into it. Visualization is `tsr.viser`.
- **PyVista, Matplotlib and Pillow as visualization dependencies.** `sstsr[viz]` and
  `sstsr[viser]` now install the same thing — Viser — so an existing `sstsr[viz]`
  install keeps working and gets the supported backend. `pip install sstsr` remains
  visualization-free, and importing sstsr still loads neither backend nor starts a
  server. Pillow moved to the dev group, where the figure script uses it.
- **Browser-free PNG rendering.** Viser renders through a connected browser; there is
  no replacement. Pin `sstsr<3` if you need it. `docs/MIGRATION-VISER.md` maps each
  removed workflow to its replacement.
- The three PyVista examples keep their printed API demonstrations and lose their
  rendering: `scripts/render_readme_figures.py` builds those same scenes in Viser, and
  `tsr.viser.studio()` is the interactive path, so re-implementing them would have
  duplicated both.

### Added
- **`tsr.viser.studio()`** — an interactive bench for debugging grasp generation.
  Choose a primitive, drag its dimensions and the gripper's, pick a factory and change
  `k`, clearance or `n_minor`; the templates regenerate and redraw on every change.
  When a request yields nothing the factory's own diagnostic is surfaced
  (`exceeds_aperture`, `finger_too_short`, `insufficient_clearance_band`), and an
  invalid request is reported as invalid rather than as an empty feasible set — the
  two failures the factory contract deliberately separates. Sliders and the factory
  list are derived from `PRIMITIVE_SPECS`, so they cannot drift from the real
  signatures. A `Show` folder isolates one depth level or one **mode**
  (`filter_templates` / `mode_of`, available without the GUI); the options are rebuilt
  from whatever the request produced. A *mode* is `mode/approach/finger_orientation`,
  deliberately coarser than the *depth family* of `docs/ARCHITECTURE.md` (which also
  fixes `symmetry` and `metadata`): a torus side grasp is one mode covering ten depth
  families. The viewer wants the coarse grouping, so it does not reuse the word
  *family* for it.
- The viewer draws jaws at the **template's own preshape** (`span + clearance`), so the
  opening differs per template — a torus side grasp closes on the tube, a torus span
  grasp swallows the ring. A template carrying no preshape is drawn as axes only rather
  than substituting the gripper's `max_aperture`, which would show a far coarser grasp
  than the template claims.

### Fixed
- **The release build pins the version with the generic setuptools-scm variable.**
  `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SSTSR` is silently ignored — hatch-vcs does not
  pass the distribution name through — so a tag build still fell back to `git describe`
  and, with an rc and its final tag on one commit, resolved to the rc. The `v2.3.0`
  build produced `2.3.0rc1` and the tag/artifact guard stopped it before publishing.
  Verified by building with each variable set to `9.9.9`: the per-distribution form
  produced `2.3.0` (from describe), the generic form produced `9.9.9`.

## [2.3.0] — 2026-09

Interactive visualization. `tsr.viser` replaces static rendering as the recommended way
to inspect grasp templates: it serves a scene you interrogate rather than a picture of
samples, with a slider per free `Bw` coordinate so a continuous freedom is scrubbed
instead of sampled. The PyVista backend still works and is now deprecated, with a
migration guide and one published release of warning before it is removed in 3.0.

Visualization remains explanatory and diagnostic. It is never evidence that a template
is correct — that is the analytic oracle's job.

### Added
- **Interactive Viser viewer** (`tsr.viser`, `pip install "sstsr[viser]"`; #77).
  `show_templates` draws a reproducibly sampled pose cloud — reference object, canonical
  end-effector axes, and jaw geometry at each template's own preshape — and
  `explore_templates` drives a single grasp from one slider per **free** `Bw` coordinate
  (from `free_coordinates`), so a continuous freedom is scrubbed rather than sampled.
  Viser stays optional: `import tsr` pulls in neither backend and never starts a server,
  and the default test suite does not require the extra.

### Changed
- **The README figures are rendered with Viser** (`scripts/render_readme_figures.py`),
  so the documentation images no longer depend on PyVista ahead of its removal. Grasp
  panels are captured per primitive and composed; the placement figures are single
  shared-table scenes with each box face in a fixed colour, which is what shows that
  the placer finds all six faces, both cylinder caps and both torus sides.
  `tsr.viser.add_primitive` and `torus_mesh` were added to support this — Viser has no
  native torus.

### Deprecated
- **The PyVista backend (`tsr.viz`) is deprecated and will be removed in sstsr 3.0**
  (#139). Viser is now the recommended viewer in the README, examples and docs.
  Constructing `TSRVisualizer` emits a `DeprecationWarning` naming the replacement, the
  removal version and the migration guide; **importing `tsr.viz` does not warn**, so a
  library that imports it cannot warn on its users' behalf. Nothing breaks in 2.x: the
  backend keeps working and the `viz` extra still installs PyVista — repointing it at
  Viser in a minor release would silently change what an install provides.
  `docs/MIGRATION-VISER.md` states the replacement for every public PyVista workflow.
- **Browser-free PNG rendering ends with PyVista.** Viser's pixels come from a browser's
  WebGL context: `get_render` is a method on a *connected* `ClientHandle`, viser 1.1.1
  exposes no offscreen or screenshot API, and the scene serializer produces a `.viser`
  file or a standalone interactive HTML page rather than an image. This is an accepted
  consequence of the migration, documented rather than discovered: through 2.x use
  `tsr.viz`; from 3.0 images are captured from the viewer.
- **The release version comes from the git tag** (hatch-vcs). `pyproject.toml` no
  longer carries a static `version`; `tsr.__version__` reads the build-time
  `_version.py`, falling back to installed metadata. This makes the TestPyPI dry run
  real: an rc tag now necessarily builds an rc artifact, whereas through 2.2.0 the
  static version meant `v2.2.0rc1` published `2.2.0` to TestPyPI and the rc was never
  exercised. The release workflow additionally refuses to publish when the built
  artifact's version does not equal the tag, or when an rc tag builds a final version
  (or vice versa). In CI the version is pinned from the ref rather than inferred:
  `git describe` must choose when several tags point at one commit — the normal
  rc → final flow — and it resolved to the rc in CI while resolving to the final tag
  locally. The guard caught exactly that before anything was published. See
  `docs/RELEASING.md`.

## [2.2.0] — 2026-09

Establishes the *geometric* correctness of primitive parallel-jaw grasp templates
(epic #65). 2.1.0 guaranteed that a template was well **formed**; 2.2.0 guarantees
that every pose it admits is a viable parallel-jaw pre-grasp, certified against an
independent analytic oracle rather than against the generator's own formulas.

**Highlights**

- An executable **geometric contract** (`docs/ARCHITECTURE.md`) and an independent
  **analytic oracle** for boxes, cylinders, spheres and tori (#66, #67).
- **Seven generator defects fixed**, each with a deterministic regression: palm
  inside the object beyond finger reach (#69), whole-face rejection from one thin
  box dimension (#70), torus minor-angle and coverage semantics (#71), cylinder caps
  ignoring the height (#122), contacts on edges at near-zero clearance (#121),
  straddle feasibility decided on sums without slack (#129), and a default preshape
  that could refuse the clearance it was asked for (#131).
- A **Hypothesis property matrix** over every factory, combined entry point and
  named gripper, plus a **mutation gate** that fails if the generator is corrupted
  (#73) — it found #121, #122 and #129.
- **Structured provenance** on every template (#66, #81), and set coverage separated
  from sampling distribution, with a Haar-uniform sampler (#72).

Templates that were previously emitted but are not geometrically sound are now
withheld: short-cylinder cap grasps beyond the far cap, near-zero-clearance box and
cylinder poses on edges, and grasps whose jaw margin sits below the straddle floor.

### Changed
- **Packaged YAML templates have an explicit assurance boundary** (#75). The seven
  hand-authored examples are now smoke-tested through the public loading and
  instantiation APIs and documented as representation-valid recipes, not as
  analytically certified primitive grasps. This avoids a parallel validation manifest
  for templates that intentionally omit primitive dimensions and gripper geometry.

### Fixed
- **Straddle feasibility is decided on the margin, with slack** (#129). `_infeasibility_reason`
  compared *sums* — `preshape` against `object_span + 2·atol` — so when the margin was
  orders of magnitude smaller than the span the addition lost it and a margin below the
  floor was accepted (e.g. `Robotiq2F85().grasp_sphere(0.02125, clearance=c)` with `c`
  one ulp under the floor). It now tests `preshape − object_span`, which is exact by
  Sterbenz, and requires `4·atol(scale)` — each pad clearing the object by `2·atol`,
  twice the oracle's tolerance, for the same reason #121 doubles the edge margin: at
  exactly `2·atol` the oracle's clause-2 bound lands on the contact coordinate and ~59%
  of sampled poses were uncertifiable, while any larger margin left none. A **default**
  preshape is now built by `GripperBase._default_preshape`, which advances `span + c`
  until the realized margin is at least the requested clearance, so asking for the
  documented floor is honoured instead of being rejected by the rounding of that sum
  (#131); an explicitly supplied preshape is never altered. Ordinary clearances and
  preshapes are unaffected. Found by the #73 property matrix.
- **Box and cylinder contacts stay off the edges at zero or sub-tolerance clearance**
  (#121). `clearance = 0` is a valid request (#104), but the closed bands used the
  edges themselves as limits, so contacts landed exactly on an edge: zero insertion
  (`grasp_box_top(0.05, 0.05, 0.05, clearance=0)` emitted a depth-0 template whose
  fingertips only touch the face), a contact on the far face, a lateral slide edge, or
  a cylinder rim, where the surface normal is ambiguous. Every edge-facing band limit
  now uses `m = max(clearance, 2 · _length_atol(scale))`, mirroring the scale-aware
  straddle margin (#107): the box approach and slide bands, the cylinder cap bands,
  and the cylinder-side height band. Clearances above the tolerance are unchanged, and
  palm clearance and the default preshape still use the requested clearance. Found by
  the #73 property matrix.
- **Cylinder cap insertion depth is bounded by the height** (#122). `grasp_cylinder_top`
  and `grasp_cylinder_bottom` built depths over `[clearance, finger_length − clearance]`
  and ignored the cylinder, so on a short cylinder the deeper templates pushed the
  fingertips past the opposite cap (#67 oracle clause 4) — including at the default
  clearance, e.g. `grasp_cylinder_top(0.02, 0.03, clearance=0.006)` with 80 mm fingers
  emitted depths 0.04 and 0.074 into a 30 mm cylinder. The band is now
  `d ∈ [m, min(finger_length, cylinder_height) − m]` with the edge margin
  `m = max(clearance, 2 · atol(scale))` (#121), the same rule the box approach bands
  use, returning `[]` with `insufficient_clearance_band` when empty.
  Found by the #73 property matrix.
- **Grasp provenance depth-count semantics** (#81). `GraspProvenance.depth_count` is
  now documented and guaranteed to be the number of **distinct** depth levels emitted
  for a family (`≤ k`), with `depth_index` ordering them shallow to deep. The shared
  depth helper already collapsed an exact one-point band to a single depth (#68);
  it now also drops duplicates when a band only a few ulps wide makes `np.linspace`
  round samples onto the same float, which previously produced repeated depths that
  all reported `depth_count = k`. A Hypothesis property checks the invariant for every
  grasp factory, including the combined entry points, with one-point-band tests
  per family and ulp-wide-band regressions.
- **Sphere grasp coverage vs. sampling distribution** (#72). `grasp_sphere` claimed
  its roll/pitch/yaw box makes sampling produce "uniformly distributed approach
  directions"; independent uniform RPY (`TSR.sample`) is **not** Haar-uniform on
  SO(3) and concentrates approach directions near the poles. The docs now separate
  the represented set (all of SO(3) by default) from the sampling distribution, and
  define a restricted `angle_range` precisely: approach directions in the **lune**
  `azimuth ∈ angle_range` (not a spherical cap), any spin about the approach axis.
  Template descriptions no longer say "full SO(3)" when yaw is restricted.
- **Torus minor-angle, reach, clearance, and coverage semantics** (#71). Torus-side
  grasps applied no clearance to the deep depth endpoint (`min(2r, L)`), so the palm
  could sit inside the tube surface (oracle clause 3); the band now follows the same
  reach/clearance rule as cylinder-side and sphere,
  `d ∈ [tube_radius, min(2·tube_radius, finger_length) − clearance]`, returning `[]`
  with `finger_too_short` when `finger_length < tube_radius + clearance`, or with
  `insufficient_clearance_band` when reach is ample but `clearance > tube_radius`
  empties the far-surface limit (#114). Torus-span
  grasps used a `[clearance, L − clearance]` depth band whose shallow depths did not
  reach the equatorial plane (the widest outer diameter), so the fingertip stopped
  short (oracle clause 5); the band now starts at `tube_radius` (the fingertip
  reaches the equator) up to `finger_length − clearance`. `n_minor == 1` now samples
  the **centre** of the minor-angle range (the equator `α = 0` by default) instead of
  `linspace(...)[0] = −π/2` (a lower approach). A new `minor_angle_range` parameter
  (keyword-only, default `[−π/2, +π/2]`) makes the represented set explicit and is
  accepted by `GripperBase.grasp_torus_side`, the combined `grasp_torus`, and the
  registry route, which forwards it to side grasps only; existing positional calls
  are unchanged (#112). It must lie within the externally accessible **outer half**,
  validated as an exact closed interval with no tolerance: endpoints exactly on
  `±π/2` are valid, anything outside raises `ValueError`; inner-hole approaches are
  out of scope (#113). Torus-side straddle feasibility uses the
  scale-aware tolerance with the torus scale `R + r` (#107). A new oracle-backed
  property (`tests/tsr/hands/test_torus_soundness.py`) certifies every side and span
  pose against the #67 analytic oracle, with rotational equivariance about the axis,
  equatorial mirror balance, reach boundaries, and the minor-angle/coverage semantics,
  including generated custom minor-angle subintervals through the direct and combined APIs.
- **Per-orientation box grasp feasibility** (#70). Each box face offers two
  finger-opening orientations (open along one in-face axis, slide along the other).
  The factories used to reject the **whole face** when either slide band was empty,
  so a thin dimension discarded a still-valid perpendicular orientation
  (`grasp_box_top(0.003, 0.06, 0.05)` returned `[]` although span-x straddles the
  thin x and slides along y). Feasibility is now evaluated **per orientation** — a
  negative slide band or an over-wide span removes only that orientation, and the
  public method emits one diagnostic only when the whole requested family is empty.
  A slide dimension exactly `2·clearance` yields a *zero-width* band, kept as one
  fixed centered pose (only a negative band is empty), per the exact-interval
  convention (#105, #110).
  The insertion-depth band is now capped by the box's extent **along the approach
  axis** (`min(finger_length, extent) − clearance`), so the fingertip clears the far
  face by a clearance rather than penetrating a short box, and box straddle
  feasibility uses the contract's scale-aware tolerance (#107). A new oracle-backed
  property (`tests/tsr/hands/test_box_soundness.py`) certifies every emitted pose
  across all six faces and both orientations against the #67 analytic oracle, with
  axis-swap invariance, opposite-face mirror consistency, thin-dimension removal,
  and aperture/slide boundary regressions — all reading structural provenance, never
  parsing display names.
- **Cylinder-side and sphere reach soundness** (#69, #107, #108). The radial depth
  band `[radius, min(finger_length, 2·radius) − clearance]` can be empty for two
  distinct reasons — the fingers are too short to keep the palm outside with
  clearance (`finger_length < radius + clearance`), or reach is ample but an
  excessive clearance relative to the radius empties the far-surface limit — and the
  factories now classify the empty set accordingly (`finger_too_short`, precedence,
  else `insufficient_clearance_band`), so the short-finger case that used to emit a
  palm-inside template is `[]` with an accurate reason (#108). Factory straddle
  feasibility now uses the contract's **scale-aware tolerance** (`_length_atol`,
  `1e-9 + 1e-6·scale`, matching the oracle): a preshape whose contact-to-jaw margin
  is below tolerance yields `[]` with `cannot_straddle` rather than a template the
  oracle rejects (#107). A new oracle-backed property
  (`tests/tsr/hands/test_reach_soundness.py`) certifies every emitted cylinder-side
  and sphere pose — at its `Bw` extrema and midpoint — against the independent #67
  analytic oracle, across radius/finger-length/aperture/clearance/`k`, the exact
  reach and aperture boundaries and their `nextafter` neighbours at several scales,
  SO(3) equivariance for spheres, and the named grippers at their reach boundaries.
  No previously-sound templates were removed.
- **Gripper parameter hardening and empty-depth semantics** (#68). An excessive
  positive `clearance` (more than half the finger length) used to make the
  cylinder-cap, box, and torus-span factories build depths with
  `linspace(clearance, finger_length - clearance, k)` over a *reversed* interval,
  silently emitting templates with a backwards shallow-to-deep ordering. A shared
  `_usable_depths` helper now returns `[]` with reason `insufficient_clearance_band`
  when the band is empty, under an **exact, scale-independent** boundary policy
  (`hi < lo` → empty; `hi == lo` → one deduplicated slot; `hi > lo` → `k` depths) —
  a positive-width interval never collapses through a default `np.isclose` tolerance,
  and a single-slot template gets a single-slot human label (#105). Input validation
  is centralized and applied uniformly across every factory: finite/positive
  `finger_length`, `max_aperture`, primitive dimensions and `preshape`; finite,
  nonnegative `clearance_fraction` **and every explicit per-call `clearance`**
  (NaN/inf/negative/Boolean raise a `ValueError` naming `clearance`, before it
  derives any preshape/bounds/depths; `clearance=0` stays valid) (#104); integer
  `k >= 1` and `n_minor >= 1`; finite, ordered `angle_range`; and positive
  `cylinder_height` for the side/top/bottom entry points alike. Nonsensical arguments
  raise `ValueError`; valid arguments with an empty feasible set return `[]` with one
  coherent diagnostic log. No generated template contains NaN or infinity.
  Property-tested with non-finite/zero/negative scalars, `nextafter` boundary
  neighbours at small/ordinary/large scale, excessive clearances, and
  integral/non-integral counts.

### Added
- **Hypothesis property matrix and mutation-sensitive release gate** (#73).
  `tests/tsr/hands/_grasp_matrix.py` drives every public `grasp_*` factory — combined
  entry points and named grippers included — through the independent #67 oracle, with
  cases across scale and aspect regimes and `boundary_cases` that sit exactly on, and
  one ulp either side of, every documented feasibility boundary. Seven invariants are
  checked: oracle witness at Bw extrema/corners/interior points, argument validation
  vs. physical infeasibility, rigid-transform equivariance, primitive and
  opposite-face symmetries, feasibility monotonicity, declared coverage and structured
  uniqueness (including combined == parts, compared as whole templates so semantic and
  display fields count), and witness preservation through dict/JSON/YAML round-trips.
  Documented feasibility boundaries are additionally pinned per factory family and
  asserted as below/exact/above *transitions*, since an empty result satisfies the
  other checks vacuously. `tests/tsr/hands/test_grasp_mutation_gate.py` injects
  each bug class from #73 — approach sign, face origin, axis mapping, object extent,
  clearance term, reach, plus code mutants patching `_resolve_clearance`,
  `_usable_depths` and `_infeasibility_reason` — and requires the same checks to
  reject it, with two documented, re-certified exclusions where the "mutation" is a
  true symmetry. Budgets scale with `TSR_MATRIX_SCALE` (default 1, ~11 s); the new
  weekly/on-demand `release-gate` workflow runs at 10x in its own concurrency group. The matrix found #121 and #122
  on its first run. See docs/ARCHITECTURE.md, "Verification matrix and release gate".
- **`sample_haar` / `sample_haar_xyzrpy`** (`tsr.sampling`, exported from `tsr`;
  #72). Samples a TSR with rotations Haar-uniform over its Bw box by drawing roll and
  yaw uniformly and `sin(pitch)` uniformly (the ZYX Haar density is `∝ cos(pitch)`);
  translation is uniform. Requires pitch bounds within `[-π/2, π/2]` (exact, else
  `ValueError`); a nonzero pitch interval may touch `±π/2`, but a pitch fixed exactly
  at the ZYX gimbal lock is rejected, since roll and yaw are coupled there (#116).
  Reproducible under an explicit `rng`, and kept separate from
  `TSR.sample`, which is unchanged. For sphere grasps it gives approach directions
  uniform by area over the sphere or lune. Deterministic KS tests
  (`tests/tsr/test_haar_sampling.py`, `tests/tsr/hands/test_sphere_sampling.py`)
  check the rotation-angle law, uniform axes, lune semantics, and left-invariance
  under reference rotations; a Hypothesis property checks set membership, and every
  sphere-grasp Haar sample keeps a #67 oracle witness.
- **Independent analytic grasp oracle** (#67, #93–#102, test-side) in
  `tests/tsr/hands/_grasp_oracle.py`: certifies the geometric-soundness clauses of
  the #66 contract for box / cylinder / sphere / torus. Contact geometry is
  **exact per `(primitive, mode)`** — closed-form line/surface intersection derives
  jaw contacts, span, analytic normals, realized insertion depth, torus minor
  angle, and clearance-aware usable bands from the concrete pose (no sampling grid,
  so results are resolution-independent and scale-equivariant; ~0.1 ms/cert). An
  independent signed-distance representation is used only to *verify* those
  analytic contacts, never to search (#93). `mode` (and optional `approach` /
  `finger_orientation`) select and are *checked against* the pose — a mislabeled
  mode fails — and mode-specific clearance bands (cylinder ends, box edges) are
  enforced (#94). All inputs are validated up front: a malformed model raises
  `ValueError`, a malformed pose returns a structured clause-1 failure, and a
  reflection (`det(R) = -1`) or non-finite clearance is rejected (#95). Clearance
  bands are enforced along the full insertion interval for cap/face grasps, not
  just laterally (#96); aperture fit is **pose-relative** (both contacts must lie
  between the open jaws about the palm, so an object translated outside the finite
  opening no longer certifies on span alone, #97); declared `approach` labels use
  the hand-occupied-side convention with a built-in vocabulary that raises on
  unknown labels and fails clause 8 on a geometric mismatch (#98); and the
  torus-side model handles the full approach-minor-angle range with tube-centre
  ray verification, reporting the pose-derived approach minor angle (#99).
  Axis-aligned modes use a true angular tolerance and every accepted contact must
  lie in the concrete posed finger sweep `p + t·z_EE + s·y_EE` (#100); every
  successful witness has finite geometry and a contact plane strictly on the
  forward finger segment `0 < t ≤ L`, with `radial`/`tube`/occupied-side predicates
  checked against a shared pose-derived local frame (#101); and a deterministic
  acceptance matrix covers each mode's boundaries, axis permutations, azimuths,
  scale regimes, and real-factory provenance across all families (#102). Returns a
  structured `GraspWitness` naming the violated clause and its geometry. Its unit
  tests use hand-derived poses only. NumPy-only, no shipped-package change; it
  backs the generator repairs in #68–#73.
- **Executable geometric grasp contract** (#66) in `docs/ARCHITECTURE.md`: four
  assurance layers (representation / soundness / coverage / embodied), eight
  soundness clauses, and the soundness-vs-coverage-vs-distribution distinctions,
  with defined tolerances and one sound + one infeasible worked example per
  primitive.
- **`GraspProvenance`** (#66, #78–#84) — a small, immutable, **extensible** value
  object on `TSRTemplate.provenance` recording each grasp's primitive, mode,
  hand-occupied `approach`, object-frame `finger_orientation`, insertion `depth`,
  `symmetry`, and descriptive `metadata`. It validates only representation
  invariants and round-trips losslessly; it is *not* a closed schema, so external
  grasp generators can use their own labels. The sstsr native vocabulary and
  cross-field relational checks live separately in
  `tsr.hands._conformance.validate_builtin_provenance`, which the built-in factory
  tests run over every emitted template. The design principle — **provenance
  declares intent; the pose determines geometric truth** — keeps these two layers
  and the analytic oracle (#67) distinct. Every `ParallelJawGripper` factory emits
  a conformant record, so tests read grasp modes structurally instead of parsing
  `name`. The sphere grasp is mode `surface` (full SO(3)), not `equatorial`.
  Representation-only; grasp geometry is unchanged.
- **Witness-aware, bounded `TSRChain` inverse membership** (#85). A chain is an
  exact parameterized set, but deciding membership from a pose alone is a bounded
  nonconvex inverse problem: a local solve can produce a positive witness, but
  failing to find one is **not** a proof of nonmembership. New API makes that
  distinction explicit:
  - `TSRChain.sample_with_witness(rng=None) -> ChainSample` returns a sampled
    `pose` and the `(n, 6)` `coordinates` that constructed it — a positive
    membership certificate needing no optimizer.
  - `TSRChain.validate_witness(pose, coordinates, tolerance=EPSILON) -> bool`
    certifies a witness with one bounds check and one forward composition — no
    SciPy, no solve.
  - `TSRChain.solve(pose, initial_guess=None, max_starts=11, max_nfev=2200) ->
    ChainSolveResult` runs a bounded, deterministic multi-start inverse solve that
    warm-starts from `initial_guess`, exits on the first tolerance-satisfying
    witness, and returns `status` (`"satisfied"`/`"not_found"`, never
    `"infeasible"`), `coordinates`, `residual`, `nfev`, and `starts`. `max_starts`
    is a **strict total** cap on optimizer starts and `max_nfev` a **strict total**
    cap on objective evaluations, enforced by an internal global counter rather
    than SciPy's soft `maxfun` (#89, #91); the valid-witness fast path takes one
    forward composition and does not import SciPy (#88); numerical controls are
    validated before SciPy is imported.
  - `ChainSample` and `ChainSolveResult` are exported from `tsr`.
- **`TSRChain.to_transform` canonicalizes wrapping rotational coordinates** before
  clamping (#87): a valid RPY coordinate expressed in `[-pi, pi]` for a wrapping
  interval (e.g. `-3.0` for `[3π/4, -3π/4]`) is wrapped into the component's
  continuous interval instead of being clipped to an unrelated boundary rotation,
  which had silently changed the pose and broken the witness contract. A single
  shared coordinate→continuous-chart helper is used by both `to_transform` and the
  inverse solver's start construction, so a wrapping neighboring-state
  `initial_guess` is canonicalized, not clipped, before it warm-starts the
  optimizer (#90). `validate_witness` on an empty chain now returns `False` instead
  of raising.

### Changed
- **`TSRChain.contains`, `distance`, and `closest_transform` are documented as
  numerical, non-certifying for multi-TSR chains** (#85). Multi-TSR `contains`
  now delegates to `solve` and accepts an optional `initial_guess` for the exact
  warm-start fast path; `False` means no witness was found within the numerical
  budget, not certified nonmembership. `distance` returns `0` with the witness
  when one is found, else a best-found residual (an upper bound on the true
  minimum). Single-TSR `contains` remains the exact closed-form check.

## [2.1.0] — 2026-09

### Added
- **`TSR.closest_transform(trans)`** (#53) and **`TSRChain.closest_transform(trans)`**
  (#63) — return the distance *and* the closest in-bounds pose in the **world
  frame** (`T0_w @ xyzrpy_to_trans(bwopt) @ Tw_e` for a TSR; the composed
  transform for a chain), so planner projection code no longer reconstructs the
  frame composition by hand (a recurring source of frame bugs).
- `Robotiq2F140.PALM_OFFSET_FROM_BASE_MOUNT = 0.100` (base_mount → grasp_site along
  approach), for symmetry with `Robotiq2F85`.

### Fixed
- **Multi-TSR `TSRChain` could still reject a pose from its own `sample()`** (#57).
  The 2.0.1 fix removed the zero-width-gradient stall, but a single midpoint-start
  L-BFGS-B could converge to an ordinary nonzero local minimum (reporting success,
  no warning) and reject a constructively-valid sample — especially with
  non-identity component frames. `TSRChain.distance` now minimises a **smooth
  squared pose residual** (the geodesic's `arccos` is non-smooth at 0 and breeds
  local minima) from **deterministic multi-start** points, and reports the true
  geodesic at the recovered coordinates. Verified across the deterministic
  non-identity-frame stress reproduction and a multi-TSR Hypothesis property.

### Changed
- **`Robotiq2F140.MAX_APERTURE` corrected `0.140 → 0.128 m`** (#60) — the usable
  inner-face-to-inner-face gap at full open, measured from `2f140.xml`. The old
  0.140 was the manufacturer's nominal/outer figure and overstated the gap by
  ~12 mm, claiming feasible grasps of objects that don't fit between the pads
  (same distinction as the `Robotiq2F85`'s 0.085 m). **Behavior change:** some
  wide grasps that previously returned templates now return `[]`.

### Documentation
- Reconciled the preconfigured grippers' `finger_length` documentation (#58). All
  three (`Robotiq2F140` 0.114 m, `Robotiq2F85` 0.059 m, `FrankaHand` 0.037 m) use
  one convention — **palm → pad tip** (finger reach along approach) — now stated
  consistently in the class docstrings, the README table, and the tests. Fixed the
  `Robotiq2F140` docstring (it claimed the wrong derivation and a stray "82 mm"),
  the stale README values (55 mm / 44.5 mm), the "pad-contact midpoint" misnomer,
  and a tutorial example that mislabeled a generic 55 mm gripper as a "2F-140".
- Finished the pad-tip terminology reconciliation (#64): the `FrankaHand.FINGER_LENGTH`
  inline comment and the historical 2.0.0 changelog line no longer say "pad-mid" /
  "pad-contact midpoint"; all named-gripper docs now say **palm → pad tip**.

## [2.0.1] — 2026-09

Patch release: correctness and reproducibility fixes from the post-merge audit
of #54. No API removals; adds an optional `rng` parameter.

### Fixed
- **Multi-TSR `TSRChain` could reject a pose from its own `sample()`** (#57).
  `TSRChain.distance` optimised over all `6·n` coordinates, so a point-bound
  coordinate (e.g. a fixed `[0, 0]`) gave a zero-width finite-difference step,
  a NaN gradient, and a stalled optimiser. It now optimises only the free
  coordinates, holding fixed ones at their value.
- **Documentation described `TSRChain` as "AND"/intersection semantics** (#55).
  A chain is the serial composition of its component TSRs, not the intersection
  of their world-frame pose sets (two serial `x∈[0,1]` TSRs reach `x=1.5`).
  Reworded the changelog, `contains()` docstring, and the semantics tests.

### Added
- **`rng` parameter** on `TSR.sample`/`sample_xyzrpy`, `TSRChain.sample`/
  `sample_xyzrpy`, and `TSRTemplate.sample`; `sample_from_tsrs` /
  `sample_from_templates` forward it (#52). Passing a `numpy.random.Generator`
  now makes the sampled *pose* reproducible, not just the TSR choice. Defaults to
  the global RNG when omitted.

### CI
- The downstream-dispatch job no longer reddens an otherwise-green run when
  `ROBOT_CODE_DISPATCH_TOKEN` is absent; it skips with a visible notice (#56).

## [2.0.0] — 2026-09

First PyPI release. The distribution is named **`sstsr`** (the name `tsr` was
already taken); the import name is unchanged — `import tsr`.

### Packaging & licensing
- **Distribution renamed to `sstsr`.** `pip install sstsr`, then `import tsr`.
- **Relicensed to BSD-2-Clause**, matching the original PrPy/OpenRAVE-derived TSR
  core (resolves the prior README-vs-LICENSE mismatch).
- Grasp/hand classes (`ParallelJawGripper`, `Robotiq2F85`, `Robotiq2F140`,
  `FrankaHand`) are now exported at the top level alongside `StablePlacer`.
- `tsr.__version__` is now available.
- Tested on Python 3.10–3.14 (classifiers and CI matrix extended to 3.13 and 3.14).

### Breaking changes
- **`TablePlacer` removed** — use `tsr.placement.StablePlacer`.
- **`TSRChain.contains()` now tests serial-composition membership** — a chain is
  the set of poses reachable by serially composing a transform from each
  component TSR (consistent with `distance()`), not a match against a single
  world-frame pose.
- **Grasp factories return `[]` for infeasible geometry instead of raising.**
  `grasp_cylinder_*` and `grasp_sphere` previously raised `ValueError` when the
  object exceeded the aperture; they now return `[]`, consistent with the box and
  torus factories. Exceptions are reserved for invalid *arguments* (non-positive
  sizes, `k < 1`, reversed `angle_range`). See `docs/ARCHITECTURE.md`.
- **Gripper geometry corrections** (change sampled poses for existing callers):
  `Robotiq2F140` frame correction and `finger_length`; `FrankaHand.finger_length`
  corrected to **37 mm** (measured from the hand-body forward edge / collar to the
  pad tip, not the finger-joint origin); clearance is now scaled by
  graspable depth, not `finger_length`.

### Fixed
- **RPY near gimbal lock** (`rot_to_rpy`, `rot_within_rpy_bounds`): the
  singularity threshold was far too wide (`1e-3`), snapping pitch to ±90° up to
  ~2.5° early and discarding up to 0.04 rad of rotation. This made `contains()`
  reject poses that `sample()` produced. Tightened to a true-singularity
  threshold; round-trip error dropped from ~0.04 to ~1e-13.
- **`StablePlacer.place_mesh` seating**: objects whose center of mass is not at
  the object-frame origin floated above or sank through the table. The resting
  face now sits at `z = 0` for any COM.
- **`wrap_to_interval`** now guarantees a result in `[lower, lower + 2π)` (a
  floating-point modulo artifact could return exactly `lower + 2π`).
- **`TSRChain.distance`** no longer crashes on outer (wrapping) rotation
  intervals, and a single-TSR chain's `contains()` uses the exact closed-form
  check instead of the optimiser.
- Degenerate meshes (fewer than 4 points, coplanar) raise a clear `ValueError`
  instead of a raw SciPy `QhullError`.
- `grasp_*(k=0)` raises `ValueError` instead of `IndexError`; reversed
  `angle_range` raises instead of silently collapsing yaw freedom.

### Internal
- The pure-math core (`TSR`, `TSRChain`, SE(3) helpers) moved to a `tsr.core`
  subpackage; deep imports such as `from tsr.tsr import TSR` still work via
  back-compat shims.
- Added property-based tests (`hypothesis`) covering the core consistency
  contract, serialization round-trips, and the factory invariants; added an
  architecture-enforcement test (`tsr.core` has no upward imports; `import tsr`
  pulls no heavy viz dependencies).
- Added `docs/REVIEW.md` (pre-release review) and `docs/ARCHITECTURE.md`.
- Removed the dead top-level `templates/` duplicate and the vestigial
  `MANIFEST.in` (hatchling bundles the packaged `tsr/templates/` automatically).

[2.0.0]: https://github.com/personalrobotics/sstsr/releases/tag/v2.0.0
