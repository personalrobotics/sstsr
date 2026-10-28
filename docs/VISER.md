# Interactive viewer (Viser)

`tsr.viser` is how you inspect grasp templates (adopted in #77, promoted in #139, sole
backend since 3.0). It stays **optional**: it is not in `[project.dependencies]`, and
`pip install sstsr` brings no visualization dependency.

```bash
pip install "sstsr[viz]"      # or the identical "sstsr[viser]"
```

The PyVista renderer it replaced was removed in 3.0 (#140);
[MIGRATION-VISER.md](MIGRATION-VISER.md) maps each of its public workflows to the
replacement here.

Visualization is explanatory and diagnostic. Nothing rendered here is evidence that a
template is correct — that is the analytic oracle's job (see ARCHITECTURE.md).

## Install and run

```bash
pip install "sstsr[viser]"          # or: uv sync --extra viser
uv run python examples/viser_cylinder_grasps.py
```

Then open <http://localhost:8080>. `Ctrl-C` stops the server.

```python
from tsr.hands import ParallelJawGripper
from tsr.viser import show_templates

gripper = ParallelJawGripper(finger_length=0.08, max_aperture=0.14)
templates = gripper.grasp_cylinder_side(0.03, 0.12)

server = show_templates(templates, cylinder=(0.03, 0.12), gripper=gripper, seed=0)
...                                  # draw more into the same server if you like
server.stop()
```

`show_templates` either takes a `viser.ViserServer` or creates one and returns it, so
the server's lifetime is always explicit and visible at the call site. Sampling takes a
`seed` or an `rng`, so a view can be reproduced exactly.

## Scrubbing the region with sliders

A static cloud of sampled poses shows a region as clutter. `explore_templates` instead
drives **one** grasp from a slider per free `Bw` coordinate, so a continuous freedom is
scrubbed rather than sampled:

```bash
uv run python examples/viser_cylinder_grasps.py --explore
```

```python
from tsr.viser import explore_templates, free_coordinates

free_coordinates(templates[0])
# [(2, 'z [m]', -0.057, 0.057), (5, 'yaw [rad]', 0.0, 6.283)]

server = explore_templates(templates, cylinder=(0.03, 0.12), gripper=gripper)
```

The sliders come from the template's own non-degenerate `Bw` rows, so a template is
never given a slider for a coordinate it does not actually have: a cylinder side grasp
offers `z` and `yaw`, a cap grasp offers `yaw` alone. A dropdown selects which template
(so the discrete depth levels are steps, not a slider), a checkbox overlays the sampled
cloud for comparison, and a read-only field shows the current coordinates.

Moving a slider re-evaluates `tsr.to_transform(ξ)` and assigns the node's `position` and
`wxyz`. The jaw geometry is uploaded once per template, not per frame, so scrubbing
sends two transforms rather than re-sending geometry.

## The studio: a bench for debugging generation

`show_templates` and `explore_templates` answer "what does *this* template look like".
`studio` answers the question you actually have while developing: *why did I get these
templates, and what changes them?*

```bash
uv run python -c "from tsr.viser import studio; studio().sleep_forever()"
```

Pick a primitive, drag its dimensions and the gripper's, choose a factory, change `k`,
the clearance or `n_minor` — the templates are regenerated and redrawn on every change.
The sliders and the factory list come from `PRIMITIVE_SPECS`, so they always match the
factories' real arguments.

**Show one depth at a time.** The `Show` folder isolates a single depth level or a
single mode, which is the usual move when a band looks wrong — the cloud colours
depths, but overlapping modes still pile up. The options are rebuilt from whatever the
current request produced, so they never offer a selection that is empty.

```python
from tsr.viser import filter_templates, mode_of

filter_templates(templates, depth_index=1)
filter_templates(templates, mode="side/radial/tangential")
filter_templates(templates, depth_index=1, mode="side/radial/tangential")
```

### Factory, mode, depth family

Three different things, easy to conflate:

| term | what it is | example |
|---|---|---|
| **factory** | the method you call | `grasp_cylinder` (combined) vs `grasp_cylinder_side` (one part) |
| **mode** (`mode_of`) | `mode/approach/finger_orientation` — the viewer's grouping | `grasp_cylinder` emits 3 |
| **depth family** | the ARCHITECTURE.md key: everything except the depth fields, *including* `symmetry` and `metadata` | `grasp_cylinder` emits 4 |

The last two differ because a mode can hold several depth families:

```
grasp_cylinder        12 templates | 3 modes | 4 depth families
grasp_cylinder_side    6 templates | 1 mode  | 2 depth families   (roll0, rollpi)
grasp_torus_side      30 templates | 1 mode  | 10 depth families  (2 flips x 5 minor angles)
```

The viewer groups by the coarse **mode**, since one rarely wants to choose between
`roll0` and `rollpi`, and deliberately does not call that a *family* — `depth_index`
and `depth_count` are defined over the finer key.

And because `depth_index` is per depth family, selecting `depth 0` keeps the first
depth of *each* family, which are different distances: on a cylinder, 8 mm for the caps
and 30 mm for the side. The panel lists each mode's depths in millimetres, so that is
visible rather than implied.

**An empty scene explains itself.** A grasp factory returns `[]` for a valid but
infeasible request and logs the reason at debug level; the studio captures that and
shows it, so you see *why* rather than an unexplained void:

| what you changed | what the panel says |
|---|---|
| object wider than the jaws | `exceeds_aperture; preshape=0.084, diameter=0.08, max_aperture=0.05` |
| fingers shorter than the radius | `finger_too_short; finger_length=0.02, radius=0.06` |
| clearance larger than the band | `insufficient_clearance_band; clearance=0.05, finger_length=0.08` |
| a linkage hand's closed fingertips would pass the far face | `closed_reach_too_long; box_x=0.072, box_y=0.088, clearance=0.0114, max_aperture=0.128` |
| a negative dimension | `invalid request: cylinder_radius must be a finite positive number` |

That last row is a different failure from the others, and the studio keeps them apart:
an exception means the *request* was invalid, an empty list means the *feasible set*
was empty. The panel also lists the families that were produced and how many templates
each contributed.

Captures show the 3D canvas only — Viser's `get_render` does not include the GUI panel.

## Over SSH

Viser serves HTTP and a websocket on one port; forward it and browse locally:

```bash
ssh -L 8080:localhost:8080 user@host
uv run python examples/viser_cylinder_grasps.py --port 8080   # on the remote host
```

Open <http://localhost:8080> on your own machine. Nothing needs to be installed
locally, and no X server, VTK or OpenGL is involved on the remote side — which is the
main practical difference from the PyVista path.

`server.request_share_url()` exists for sharing without a tunnel; it relays through a
third-party service, so it is a deliberate choice rather than the documented default.

## What it draws

For each sampled pose: the canonical end-effector axes (x red, y green, **z blue =
approach**) and the idealized jaw geometry at the template's own preshape, plus the
reference object in the template's frame. Approach direction, jaw orientation and
grasp depth are therefore readable directly, and orbiting shows the continuous yaw
freedom as a band of poses rather than a single picture.

## Measured behaviour

On an M-series laptop, viser 1.1.1, server-side scene construction:

| poses | line segments | build time |
|---|---|---|
| 12 | 48 | 0.003 s |
| 240 | 960 | 0.016 s |
| 1200 | 4800 | 0.072 s |
| 4800 | 19200 | 0.261 s |

Server startup plus a first scene is ~0.1 s. All poses share **two** scene nodes — one
batched-axes node and one line-segments node — so the browser updates two objects
rather than N, which is why this scales linearly rather than collapsing at a few
hundred poses. Browser-side frame rate has to be judged interactively; these numbers
only bound the Python side.

## Placement templates

`show_templates` and `explore_templates` take a placement collection as readily as a grasp
one, and label it from its own record — the resting face, the tipping angle and whether the
equilibrium is stable or neutral, rather than a depth family a placement does not have:

```python
from tsr.placement import StablePlacer
from tsr.viser import explore_templates, mode_of

templates = StablePlacer(table_x=0.6, table_y=0.4).place_box(lx=0.1, ly=0.08, lz=0.06)
server = explore_templates(templates)          # a slider per free Bw coordinate
sorted({mode_of(t.provenance) for t in templates})
# ['box/+0.00,+0.00,-1.00/stable', 'box/+0.00,+0.00,+1.00/stable', ...] one per face
```

To draw the object itself alongside the poses, add the primitive to the returned server's
scene with `add_primitive(..., centered=True)` — a placement gives the pose of the object's
**centre**, where the grasp factories put a box or cylinder at `z ∈ [0, height]`.

`mode_of` groups a placement by its **resting face normal**, so a box offers six entries
rather than one; an analytic box carries no `face_index`, which is why the normal names the
face and not the index. `filter_templates(..., depth_index=...)` is a grasp idea, so no
placement matches it — it returns nothing rather than raising.

A collection must be all one kind. The controls differ per kind, so a mixed grasp/placement
collection is **refused** with a message naming what it holds; draw them in separate calls.
Dispatch is on the record's `KIND` discriminator, so a third-party record is described
generically rather than refused.

## Image capture

Viser captures images through a **connected browser client** — there is no browser-free
path, and that is an accepted consequence of the migration (see MIGRATION-VISER.md):

```python
client = next(iter(server.get_clients().values()))    # requires an open browser
image = client.get_render(height=720, width=1280)     # numpy array
```

With no client connected there is nothing to render from: `get_render` lives on
`ClientHandle`, and `server.get_clients()` is empty until a browser attaches. viser
1.1.1 exposes no offscreen or screenshot API, and the scene serializer produces a
`.viser` file or a standalone interactive HTML page rather than an image.

Browser-free PNG generation ended with PyVista in 3.0. The images in this
repository are regenerated from the viewer:

```bash
uv run python scripts/render_readme_figures.py          # all three
uv run python scripts/render_readme_figures.py --only placements
```

It opens a browser, waits for it to connect, sets a fixed camera per figure and
captures a transparent PNG, which it crops to content and mattes onto the README's
background. **Keep the viewer tab visible while it runs**: browsers throttle background
tabs, and a throttled tab never returns a frame. The capture is bounded by a timeout
and retried rather than waiting forever.

The grasp figure is captured one primitive at a time and composed afterwards, so each
panel is framed tightly; the placement figures are single scenes, because their point
is several objects sharing one table.

## Constraints this respects

- Viser is not in `[project.dependencies]`; it lives in the `viz` and `viser` extras,
  which are the same set.
- `import tsr` imports neither Viser nor this module, and never starts a server
  (enforced in `tests/tsr/test_architecture.py`).
- The default test suite does not require the extra: the backend's tests skip when it
  is absent, and none of them start a server or open a port.
- There is no backend-neutral scene abstraction: this calls Viser's own mesh, frame and
  line-segment APIs directly, and passes plain NumPy arrays at the sstsr boundary.
- The GUI is Viser's own: sliders and dropdowns are created through `server.gui`, and no
  widget abstraction is introduced for one experimental backend.
