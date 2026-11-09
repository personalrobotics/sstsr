"""Regenerate the README figures from the Viser viewer (#139).

Viser renders in a browser, so this script opens one, waits for it to connect, and
captures through it — there is no browser-free path (see docs/MIGRATION-VISER.md).
Each capture is a transparent PNG, cropped to its content and composited onto the
README's background colour, so the result matches the previous PyVista figures in
style and is tightly framed regardless of camera distance.

The grasp figure is captured one primitive at a time and composed afterwards, which
keeps each panel tightly framed. The placement figures are single scenes: the whole
point of those is several objects resting on one table.

Usage:
    uv run python scripts/render_readme_figures.py             # all four
    uv run python scripts/render_readme_figures.py --only grasps
"""

from __future__ import annotations

import argparse
import time
import webbrowser
from pathlib import Path

import numpy as np
from PIL import Image

from tsr.hands import ParallelJawGripper
from tsr.placement import Box, Cylinder, OpeningPlacer, Shape, StablePlacer
from tsr.viser import add_primitive, gripper_segments, sample_poses, torus_mesh

try:
    import viser
except ImportError as e:  # pragma: no cover - dev tooling
    raise SystemExit('the Viser extra is required: pip install "sstsr[viser]"') from e

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
BACKGROUND = (13, 17, 23)  # #0d1117, matching the previous figures
GRIPPER = ParallelJawGripper(finger_length=0.055, max_aperture=0.14)
SEED = 0

# Depth-ordered palette for jaws, and a per-primitive tint for reference objects.
JAW = np.array([(120, 170, 255), (200, 120, 255), (255, 140, 120)], dtype=np.uint8)
OBJECT = (176, 182, 193)


def _capture(client, *, position, look_at, height=900, width=1200, pad=28, attempts=3) -> Image.Image:
    """One transparent capture, cropped to content and matted onto the background.

    A render is bounded by a timeout and retried: browsers throttle background tabs, so
    a client that is connected but not visible can simply never produce a frame, and
    `get_render` waits forever by default.
    """
    client.camera.position = np.asarray(position, dtype=float)
    client.camera.look_at = np.asarray(look_at, dtype=float)
    time.sleep(1.2)
    for attempt in range(attempts):
        try:
            frame = client.get_render(height=height, width=width, transport_format="png", timeout=25.0)
            break
        except TimeoutError:
            print(
                f"    no frame within 25 s (attempt {attempt + 1}/{attempts}) — is the viewer tab visible?", flush=True
            )
    else:
        raise SystemExit("the browser never returned a frame; bring the viewer tab to the foreground and re-run")
    rgba = Image.fromarray(frame)
    bbox = rgba.getchannel("A").getbbox()
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        rgba = rgba.crop((max(0, x0 - pad), max(0, y0 - pad), min(rgba.width, x1 + pad), min(rgba.height, y1 + pad)))
    matted = Image.new("RGB", rgba.size, BACKGROUND)
    matted.paste(rgba, mask=rgba.getchannel("A"))
    return matted


def _compose(panels, gap=24) -> Image.Image:
    """Lay panels out in a row, centred vertically on a common background."""
    height = max(p.height for p in panels)
    width = sum(p.width for p in panels) + gap * (len(panels) - 1)
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, (height - panel.height) // 2))
        x += panel.width + gap
    return sheet


def _draw_grasps(scene, kind, dims, templates, *, n_poses) -> None:
    """A reference primitive and a sampled cloud of jaws around it."""
    add_primitive(scene, f"/{kind}", kind, dims, color=OBJECT, opacity=0.75)
    per = max(1, round(n_poses / max(1, len(templates))))
    points, colors = [], []
    for template in templates:
        poses = sample_poses([template], per, seed=SEED)
        aperture = float(template.preshape[0])
        segments = gripper_segments(GRIPPER.finger_length, aperture)
        for pose in poses:
            R, t = pose[:3, :3], pose[:3, 3]
            points.append(segments @ R.T + t)
            colors.append(np.broadcast_to(JAW[template.provenance.depth_index % len(JAW)], (len(segments), 2, 3)))
    if points:
        scene.add_line_segments(
            f"/{kind}/jaws",
            points=np.concatenate(points),
            colors=np.concatenate(colors).astype(np.uint8),
            thickness=0.0016,
            thickness_units="world",
        )


def grasps(server, client) -> Image.Image:
    """Four primitives, captured one at a time and composed into a row."""
    scene = server.scene
    panels = []
    specs = (
        ("cylinder", (0.04, 0.12), GRIPPER.grasp_cylinder(0.04, 0.12), (0.30, 0.24, 0.22), (0, 0, 0.06)),
        ("sphere", (0.045,), GRIPPER.grasp_sphere(0.045), (0.26, 0.20, 0.18), (0, 0, 0)),
        ("torus", (0.05, 0.015), GRIPPER.grasp_torus(0.05, 0.015), (0.26, 0.20, 0.20), (0, 0, 0)),
        ("box", (0.08, 0.06, 0.14), GRIPPER.grasp_box(0.08, 0.06, 0.14), (0.30, 0.24, 0.26), (0, 0, 0.07)),
    )
    for kind, dims, templates, camera, focus in specs:
        scene.reset()
        _draw_grasps(scene, kind, dims, templates, n_poses=24)
        print(f"  {kind}: {len(templates)} templates", flush=True)
        panels.append(_capture(client, position=camera, look_at=focus))
    return _compose(panels)


# One colour per stable pose, so a cluster reads as "these are the distinct faces this
# object can rest on" rather than as several copies of the same thing.
POSE_COLORS = ((150, 190, 255), (196, 160, 255), (255, 170, 150), (150, 220, 190), (240, 210, 130), (198, 180, 230))


# A fixed colour per box face (-x, +x, -y, +y, -z, +z) and per cylinder cap, kept the
# same across every pose. That is what makes the figure evidence rather than decoration:
# if the placer finds all six box faces, six different colours end up against the table.
FACE_COLORS = {
    (-1, 0, 0): (120, 170, 255),
    (1, 0, 0): (255, 150, 120),
    (0, -1, 0): (150, 220, 170),
    (0, 1, 0): (240, 200, 110),
    (0, 0, -1): (200, 140, 255),
    (0, 0, 1): (255, 120, 190),
}
NEUTRAL = (176, 182, 193)


def _box_face_quad(dims, normal):
    """Vertices and faces of one box face, in the box's centre frame."""
    half = np.asarray(dims, dtype=float) / 2.0
    n = np.asarray(normal, dtype=float)
    axis = int(np.argmax(np.abs(n)))
    u_axis, v_axis = [a for a in range(3) if a != axis]
    center = n * half[axis]
    u, v = np.zeros(3), np.zeros(3)
    u[u_axis], v[v_axis] = half[u_axis], half[v_axis]
    verts = np.array([center - u - v, center + u - v, center + u + v, center - u + v])
    return verts, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)


def add_faceted(scene, name, kind, dims, *, pose, centered=True):
    """Draw a primitive with each stable face in its own, fixed colour."""
    R, t = pose[:3, :3], pose[:3, 3]
    if kind == "box":
        for i, normal in enumerate(FACE_COLORS):
            verts, faces = _box_face_quad(dims, normal)
            scene.add_mesh_simple(
                f"{name}_face{i}",
                vertices=verts @ R.T + t,
                faces=faces,
                color=FACE_COLORS[normal],
                flat_shading=True,
                side="double",  # a quad's winding is arbitrary; never cull a face
            )
        return
    if kind == "cylinder":
        radius, height = dims
        add_primitive(scene, name, kind, dims, pose=pose, centered=centered, color=NEUTRAL)
        for normal in ((0, 0, 1), (0, 0, -1)):
            n = np.asarray(normal, dtype=float)
            angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            rim = np.stack(
                [radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, n[2] * height / 2)], -1
            )
            verts = np.vstack([[0.0, 0.0, n[2] * height / 2], rim])
            faces = np.array([[0, i + 1, (i + 1) % len(rim) + 1] for i in range(len(rim))], dtype=np.uint32)
            scene.add_mesh_simple(
                f"{name}_cap{int(n[2])}",  # sibling, not child: these vertices are in world coordinates
                vertices=verts @ R.T + t + R @ (n * 0.0006),  # lift off the cap to avoid z-fighting
                faces=faces,
                color=FACE_COLORS[(0, 0, int(n[2]))],
                flat_shading=True,
                side="double",
            )
        return
    if kind == "torus":
        # A torus has no faces; its two stable poses are the two sides of the ring, so
        # colour the tube's upper and lower halves.
        major, minor = dims
        verts, faces = torus_mesh(major, minor)
        upper = verts[faces].mean(axis=1)[:, 2] >= 0
        for label, mask, color in (("up", upper, FACE_COLORS[(0, 0, 1)]), ("down", ~upper, FACE_COLORS[(0, 0, -1)])):
            scene.add_mesh_simple(f"{name}_{label}", vertices=verts @ R.T + t, faces=faces[mask], color=color)
        return
    add_primitive(scene, name, kind, dims, pose=pose, centered=centered, color=NEUTRAL)


def _face_down(kind, pose) -> str:
    """Which face of the object is resting on the table, from the placed orientation."""
    if kind == "sphere":
        return "any (rotationally symmetric)"
    local_down = pose[:3, :3].T @ np.array([0.0, 0.0, -1.0])
    if kind == "box":
        axis = int(np.argmax(np.abs(local_down)))
        return f"{'+' if local_down[axis] > 0 else '-'}{'xyz'[axis]}"
    side = "+z" if local_down[2] < 0 else "-z"
    return side


def _extents(kind, dims, R) -> tuple:
    """Half-extents in x, y, z of one placed object, at its orientation.

    Packing has to use the *placed* extents: a box resting on its long face occupies a
    different patch of table than the same box stood on end, so spacing computed from
    `max(dims)` either overlaps or wastes most of the table.
    """
    if kind == "box":
        bx, by, bz = dims
        corners = np.array(
            [[sx * bx, sy * by, sz * bz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)]
        )
        world = np.abs(corners @ R.T).max(axis=0)
        return tuple(float(v) for v in world)
    if kind == "cylinder":
        radius, height = dims
        upright = abs(float(R[2, 2])) > 0.5
        return (radius, radius, height / 2) if upright else (max(radius, height / 2), max(radius, height / 2), radius)
    if kind == "sphere":
        return dims[0], dims[0], dims[0]
    major, minor = dims
    return major + minor, major + minor, minor


def placements(server, client) -> Image.Image:
    """Every primitive in every stable pose, packed onto one shared table."""
    scene = server.scene
    scene.reset()
    placer = StablePlacer(table_x=2.0, table_y=2.0)

    layout = (
        ("cylinder", (0.04, 0.12), placer.place_cylinder(0.04, 0.12)),
        ("sphere", (0.05,), placer.place_sphere(0.05)),
        ("torus", (0.05, 0.012), placer.place_torus(0.05, 0.012)),
        ("box", (0.20, 0.08, 0.28), placer.place_box(0.20, 0.08, 0.28)),
    )

    items = []
    for kind, dims, templates in layout:
        print(f"  {kind}: {len(templates)} stable pose(s)", flush=True)
        for i, template in enumerate(templates):
            # Placement templates are free in x, y and yaw: a pose is chosen by picking
            # those coordinates, not by moving the reference frame.
            xyzrpy = (template.Bw[:, 0] + template.Bw[:, 1]) / 2.0
            xyzrpy[5] = 0.0
            R = template.instantiate(np.eye(4)).to_transform(xyzrpy)[:3, :3]
            items.append((kind, dims, template, xyzrpy, _extents(kind, dims, R), POSE_COLORS[i % len(POSE_COLORS)]))

    # Rows accumulate toward the camera, so the FIRST row is the front one: shortest
    # first keeps a 28 cm box from hiding the 5 cm sphere.
    items.sort(key=lambda item: item[4][2])

    gap, row_limit = 0.05, 0.62
    placed, cursor_x, row_y, row_depth = [], 0.0, 0.0, 0.0
    for kind, dims, template, xyzrpy, (ex, ey, _ez), color in items:
        if cursor_x + 2 * ex > row_limit and placed:
            cursor_x, row_y, row_depth = 0.0, row_y + row_depth + gap, 0.0
        placed.append((kind, dims, template, xyzrpy, cursor_x + ex, row_y + ey, color))
        cursor_x += 2 * ex + gap
        row_depth = max(row_depth, 2 * ey)

    # Centre the cluster on the table, then size the slab to hug it.
    xs = [x for *_, x, _y, _c in placed]
    ys = [y for *_, _x, y, _c in placed]
    dx, dy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    resting: dict = {}
    for index, (kind, dims, template, xyzrpy, x, y, color) in enumerate(placed):
        xyzrpy[0], xyzrpy[1] = x - dx, y - dy
        pose = template.instantiate(np.eye(4)).to_transform(xyzrpy)
        # Placements give the object's CENTRE, unlike the grasp factories.
        add_faceted(scene, f"/{kind}/{index}", kind, dims, pose=pose)
        resting.setdefault(kind, []).append(_face_down(kind, pose))

    for kind, faces in resting.items():
        print(f"    {kind}: rests on {len(set(faces))} distinct face(s) -> {sorted(set(faces))}", flush=True)

    table_x = max(xs) - min(xs) + 0.34
    table_y = max(ys) - min(ys) + 0.30
    scene.add_box("/table", dimensions=(table_x, table_y, 0.012), position=(0, 0, -0.006), color=(58, 64, 76))
    return _capture(client, position=(0.62, -0.78, 0.5), look_at=(0.0, -0.02, 0.06), width=1600, height=1000)


# ── Non-convex shapes, defined once ───────────────────────────────────────────
#
# Each shape is a list of parts in its own frame. The SAME list produces both the
# vertex cloud handed to `place_mesh` and the drawing, so the figure cannot drift from
# the geometry whose stable poses it illustrates.
#
#   ("box", (cx, cy, cz), (sx, sy, sz))            centre and full side lengths
#   ("cylinder", (cx, cy, cz), (radius, height))   centre, axis along +z

SHAPES = {
    "L-shape": [
        ("box", (0.00, 0.0, 0.070), (0.040, 0.060, 0.140)),  # upright stem
        ("box", (0.045, 0.0, 0.020), (0.130, 0.060, 0.040)),  # foot
    ],
    "T-shape": [
        ("box", (0.0, 0.0, 0.060), (0.040, 0.055, 0.120)),  # stem
        ("box", (0.0, 0.0, 0.140), (0.150, 0.055, 0.040)),  # bar across the top
    ],
    "mug": [
        ("cylinder", (0.0, 0.0, 0.050), (0.038, 0.100)),  # body
        ("box", (0.052, 0.0, 0.055), (0.026, 0.016, 0.050)),  # handle
    ],
}
SHAPE_COLORS = {"L-shape": (120, 170, 255), "T-shape": (255, 150, 120), "mug": (150, 220, 170)}

# Minimum stability margin per shape. The mug needs a higher bar than the polyhedra:
# its curved side is a hull of many near-identical facets, so "lying on its side"
# appears ~20 times with a few degrees of margin each. Raising the threshold keeps the
# distinct resting configurations and drops the discretisation artefacts.
SHAPE_MARGINS = {"L-shape": 5.0, "T-shape": 5.0, "mug": 10.0}


def _part_points(part):
    """Corner / rim samples of one part, in the shape's frame."""
    kind, center, dims = part
    c = np.asarray(center, dtype=float)
    if kind == "box":
        half = np.asarray(dims, dtype=float) / 2.0
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
        return c + signs * half
    radius, height = dims
    angles = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    rim = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)], axis=-1)
    return np.vstack([c + rim + (0, 0, height / 2), c + rim - (0, 0, height / 2)])


def _part_volume(part) -> float:
    kind, _center, dims = part
    return float(np.prod(dims)) if kind == "box" else float(np.pi * dims[0] ** 2 * dims[1])


def shape_cloud(parts):
    """``(vertices, com)`` for a composed shape, both in the shape's own frame.

    The vertices and the centre of mass must be expressed in the SAME frame: the
    stability test projects the COM onto each candidate resting face, so a COM
    displaced from its cloud silently reports stable faces as marginal.
    """
    vertices = np.vstack([_part_points(part) for part in parts])
    volumes = np.array([_part_volume(part) for part in parts])
    centers = np.array([np.asarray(part[1], dtype=float) for part in parts])
    com = (volumes[:, None] * centers).sum(axis=0) / volumes.sum()
    return vertices, com


def draw_shape(scene, name, parts, pose, color):
    """Draw a composed shape at a placed pose (the pose locates its COM)."""
    R, t = pose[:3, :3], pose[:3, 3]
    for i, (kind, center, dims) in enumerate(parts):
        part_pose = np.eye(4)
        part_pose[:3, :3] = R
        part_pose[:3, 3] = t + R @ np.asarray(center, dtype=float)
        if kind == "box":
            add_primitive(scene, f"{name}_p{i}", "box", dims, pose=part_pose, centered=True, color=color)
        else:
            add_primitive(scene, f"{name}_p{i}", "cylinder", dims, pose=part_pose, centered=True, color=color)


def meshes(server, client) -> Image.Image:
    """Non-convex shapes in every stable pose the hull analysis finds."""
    scene = server.scene
    scene.reset()
    placer = StablePlacer(table_x=2.0, table_y=2.0)

    items = []
    for shape, parts in SHAPES.items():
        vertices, com = shape_cloud(parts)
        margin = SHAPE_MARGINS[shape]
        templates = placer.place_mesh(vertices, com, subject=shape, min_margin_deg=margin)
        hull_faces = len(placer.place_mesh(vertices, com, subject=shape, min_margin_deg=0.0))
        print(
            f"  {shape}: {len(templates)} stable pose(s) at >= {margin:g} deg "
            f"(of {hull_faces} hull faces, {len(vertices)} vertices)",
            flush=True,
        )
        for template in templates:
            xyzrpy = (template.Bw[:, 0] + template.Bw[:, 1]) / 2.0
            xyzrpy[5] = 0.0
            pose = template.instantiate(np.eye(4)).to_transform(xyzrpy)
            corners = np.vstack([_part_points(part) for part in parts]) - com
            world = np.abs(corners @ pose[:3, :3].T)
            items.append((shape, parts, template, xyzrpy, tuple(world.max(axis=0)), SHAPE_COLORS[shape]))

    items.sort(key=lambda item: item[4][2])  # shortest to the front row

    gap, row_limit = 0.06, 0.70
    placed, cursor_x, row_y, row_depth = [], 0.0, 0.0, 0.0
    for shape, parts, template, xyzrpy, (ex, ey, _ez), color in items:
        if cursor_x + 2 * ex > row_limit and placed:
            cursor_x, row_y, row_depth = 0.0, row_y + row_depth + gap, 0.0
        placed.append((shape, parts, template, xyzrpy, cursor_x + ex, row_y + ey, color))
        cursor_x += 2 * ex + gap
        row_depth = max(row_depth, 2 * ey)

    xs = [x for *_, x, _y, _c in placed]
    ys = [y for *_, _x, y, _c in placed]
    dx, dy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    for index, (shape, parts, template, xyzrpy, x, y, color) in enumerate(placed):
        xyzrpy[0], xyzrpy[1] = x - dx, y - dy
        draw_shape(scene, f"/{shape}/{index}", parts, template.instantiate(np.eye(4)).to_transform(xyzrpy), color)

    table_x = max(xs) - min(xs) + 0.34
    table_y = max(ys) - min(ys) + 0.30
    scene.add_box("/table", dimensions=(table_x, table_y, 0.012), position=(0, 0, -0.006), color=(58, 64, 76))
    return _capture(client, position=(0.7, -0.85, 0.55), look_at=(0.0, -0.02, 0.06), width=1600, height=1000)


# A recycling bin's opening and the clearance a release keeps inside it, as the
# OpeningPlacer tests use them; the bin is drawn shallower than its 0.757 m, which
# changes nothing about the regions.
BIN_SIZE = (0.238, 0.238)
BIN_DRAWN_DEPTH = 0.26
BIN_WALL = 0.015
BIN_CLEARANCE = 0.026
RELEASE_LOWEST = (0.02, 0.06)
SLIDE = (110, 220, 140)


def _draw_bin(scene, name, centre) -> None:
    """An open-topped bin, its rim at ``centre``'s height."""
    (sx, sy), d, w = BIN_SIZE, BIN_DRAWN_DEPTH, BIN_WALL
    cx, cy, cz = centre
    color = (58, 64, 76)
    scene.add_box(
        f"{name}/floor", dimensions=(sx + 2 * w, sy + 2 * w, w), position=(cx, cy, cz - d - w / 2), color=color
    )
    for side, (dims, offset) in {
        "x-": ((w, sy + 2 * w, d), (-(sx + w) / 2, 0)),
        "x+": ((w, sy + 2 * w, d), ((sx + w) / 2, 0)),
        "y-": ((sx, w, d), (0, -(sy + w) / 2)),
        "y+": ((sx, w, d), (0, (sy + w) / 2)),
    }.items():
        scene.add_box(
            f"{name}/{side}", dimensions=dims, position=(cx + offset[0], cy + offset[1], cz - d / 2), color=color
        )


def drops(server, client) -> Image.Image:
    """Release regions into a bin: one bin per resting orientation that fits through its
    opening, the object at the region's centre and faint at its corners, and the region
    its origin may slide in drawn on the rim."""
    scene = server.scene
    scene.reset()
    placer = OpeningPlacer(size=BIN_SIZE, depth=0.757)
    objects = (
        ("cylinder", (0.035, 0.22), Shape([Cylinder(0.035, 0.22)])),  # a bottle: lying, it rolls
        ("box", (0.06, 0.10, 0.17), Shape([Box((0.06, 0.10, 0.17))])),  # lying flat, too wide
    )
    bins = []
    for kind, dims, shape in objects:
        templates = placer.place(shape, lowest=RELEASE_LOWEST, clearance=BIN_CLEARANCE, subject=kind)
        print(f"  {kind}: {len(templates)} release region(s): {[t.variant for t in templates]}", flush=True)
        bins += [(kind, dims, t) for t in templates]

    spacing = BIN_SIZE[0] + 2 * BIN_WALL + 0.07
    for i, (kind, dims, template) in enumerate(bins):
        centre = np.array([(i - (len(bins) - 1) / 2) * spacing, 0.0, 0.0])
        _draw_bin(scene, f"/bin{i}", centre)
        T_world_opening = np.eye(4)
        T_world_opening[:3, 3] = centre
        region = template.instantiate(T_world_opening)
        hx, hy = template.Bw[0, 1], template.Bw[1, 1]
        scene.add_box(
            f"/bin{i}/slide", dimensions=(2 * hx, 2 * hy, 0.002), position=tuple(centre + (0, 0, 0.001)), color=SLIDE
        )
        low, high = template.Bw[2]
        xyzrpy = np.zeros(6)
        xyzrpy[2], xyzrpy[5] = (low + high) / 2, 0.5
        add_faceted(scene, f"/{kind}/{i}", kind, dims, pose=region.to_transform(xyzrpy), centered=True)
        # The region's extremes, in sliding and in yaw: still inside the opening.
        for j, (sx, sy, yaw) in enumerate(((-1, -1, -1.2), (1, -1, 2.0), (1, 1, 0.9), (-1, 1, -2.6))):
            xyzrpy = np.array([sx * hx, sy * hy, low, 0.0, 0.0, yaw])
            add_primitive(
                scene,
                f"/{kind}/{i}_ghost{j}",
                kind,
                dims,
                pose=region.to_transform(xyzrpy),
                centered=True,
                color=NEUTRAL,
                opacity=0.22,
            )

    # A long row of bins: a narrow field of view from afar keeps the end bins square.
    width = len(bins) * spacing
    fov = client.camera.fov
    client.camera.fov = 0.3
    try:
        return _capture(
            client, position=(0.0, -2.2 * width, 1.0 * width), look_at=(0.0, 0.0, -0.05), width=2400, height=900
        )
    finally:
        client.camera.fov = fov


FIGURES = {
    "grasps": (grasps, "tsr_grasps.png"),
    "placements": (placements, "stable_placements.png"),
    "meshes": (meshes, "mesh_placements.png"),
    "drops": (drops, "drops.png"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(FIGURES), help="render one figure instead of all")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    url = f"http://localhost:{args.port}"
    webbrowser.open(url)
    print(f"waiting for a browser at {url} — keep the tab VISIBLE, browsers throttle", flush=True)
    print("background tabs and a throttled tab never returns a frame.", flush=True)
    for _ in range(90):
        if server.get_clients():
            break
        time.sleep(1)
    else:
        raise SystemExit("no browser connected; capture is impossible without one")
    client = next(iter(server.get_clients().values()))
    time.sleep(2)

    for name in [args.only] if args.only else list(FIGURES):
        build, filename = FIGURES[name]
        print(f"{name}:", flush=True)
        image = build(server, client)
        out = ASSETS / filename
        image.save(out)
        print(f"  wrote {out.relative_to(ROOT)} {image.size}", flush=True)
    server.stop()


if __name__ == "__main__":
    main()
