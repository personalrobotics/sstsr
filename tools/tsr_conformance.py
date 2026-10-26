# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""The conformance corpus: the Python's answers, for the C++ core to be checked against.

The Python in ``src/tsr/core/`` is the source of truth for the rules, and
``cpp/`` is a second implementation of them (issue #165). This records what the
Python answers, so the C++ can be held to it by ``cpp/tests/test_conformance.cpp``
without either implementation reading the other.

The corpus is a set of regions -- identity and non-identity frames, zero-width rows,
full rotations, outer rotational intervals, near-singular pitch -- and, for each, probe
transforms on and off the region with ``contains``, ``distance`` and
``closest_transform``. Everything here is in the part of the contract that *is*
specifiable across implementations; see docs/CPP.md for what is not, and why the chain's
cold inverse is checked by properties instead.

The corpus is checked in, so a rule change shows up as a diff rather than as a silently
different expectation. Regenerate with

    uv run python tools/tsr_conformance.py
    uv run python tools/tsr_conformance.py --check   # semantics unchanged?
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tsr import CHAIN_TOLERANCE, TSR, TSRChain

ARTIFACT = Path(__file__).resolve().parent.parent / "tests" / "reference" / "tsr_conformance.json"
SEED = 20260929
CHAIN_SEED = 20261001  # separate, so adding a chain cannot shift a region's recorded probes


def _frame(x, y, z, roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    return TSR.xyzrpy_to_trans(np.array([x, y, z, roll, pitch, yaw], dtype=float))


def regions() -> list[tuple[str, TSR]]:
    pi = np.pi
    box = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-pi, pi]])
    return [
        ("identity_box_free_yaw", TSR(np.eye(4), np.eye(4), box)),
        ("offset_frames", TSR(_frame(1.0, 2.0, 0.0, 0.0, 0.0, 0.5), _frame(0.1, 0.0, 0.0), box)),
        (
            "tilted_frame_band",
            TSR(
                _frame(0.5, -0.2, 0.3, 0.3, -0.4, 1.2),
                np.eye(4),
                np.array([[-2, 2], [-0.6, 0.6], [0, 0], [0, 0], [0, 0], [-pi, pi]]),
            ),
        ),
        ("point_pose", TSR(_frame(0.2, 0.2, 0.2, 0.1, 0.2, 0.3), np.eye(4), np.zeros((6, 2)))),
        (
            "full_rotations",
            TSR(np.eye(4), np.eye(4), np.array([[0, 0.1], [0, 0], [0, 0], [-pi, pi], [-pi, pi], [-pi, pi]])),
        ),
        (
            "outer_yaw_interval",
            TSR(np.eye(4), np.eye(4), np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [3 * pi / 4, -3 * pi / 4]])),
        ),
        (
            "outer_roll_and_pitch_band",
            TSR(
                _frame(0, 0, 1.0),
                _frame(0, 0, -0.1),
                np.array([[-0.3, 0.3], [-0.3, 0.3], [-0.1, 0.1], [2.5, -2.5], [-0.2, 0.2], [-1.0, 1.0]]),
            ),
        ),
        (
            "near_singular_pitch",
            TSR(
                np.eye(4),
                np.eye(4),
                np.array([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.5, 0.5], [pi / 2 - 1e-3, pi / 2], [-0.5, 0.5]]),
            ),
        ),
        (
            "wide_roll_window",
            TSR(
                _frame(0, 0, 0, 0, 0, -2.0),
                _frame(0, 0.05, 0),
                np.array([[-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02], [-3.0, 3.0], [-0.3, 0.3], [-0.1, 0.1]]),
            ),
        ),
        (
            # A millimetre-scale region the old 1e-3 slack was as wide as (#190): an approach
            # line +-2 mm along z, everything else fixed, posed away from the origin.
            "approach_line",
            TSR(
                _frame(0.6, -0.3, 0.9, 0.0, pi / 2, 0.4),
                _frame(0.0, 0.0, -0.15),
                np.array([[0, 0], [0, 0], [-0.002, 0.002], [0, 0], [0, 0], [0, 0]]),
            ),
        ),
        (
            # Pitch fixed 1e-5 under pi/2: inside the old gimbal band (|R[2,0]| within 1e-9 of
            # 1), where pitch was snapped to pi/2 and every pose of this region read 1e-5 out --
            # hidden by the 1e-3 slack until #190 removed it.
            "pitch_in_old_gimbal_band",
            TSR(
                _frame(0.1, 0.2, 0.3, 0.2, 0.1, -0.3),
                np.eye(4),
                np.array([[-0.05, 0.05], [0, 0], [0, 0], [0.3, 0.3], [pi / 2 - 1e-5, pi / 2 - 1e-5], [-0.7, -0.7]]),
            ),
        ),
    ]


def probes(t: TSR, rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    for _ in range(6):
        out.append(t.sample(rng=rng))
    for _ in range(6):
        s = t.sample(rng=rng)
        off = _frame(*(rng.uniform(-0.3, 0.3, 3)), *(rng.uniform(-1.0, 1.0, 3)))
        out.append(off @ s)
    for pitch in (np.pi / 2 - 1e-7, -np.pi / 2 + 1e-7, np.pi / 2, -np.pi / 2):
        out.append(t.T0_w @ _frame(0.01, -0.01, 0.0, 0.2, pitch, 0.4) @ t.Tw_e)
    out.append(t.T0_w @ _frame(0.0, 0.0, 0.0, np.pi, 0.0, np.pi) @ t.Tw_e)  # an RPY-equivalent representation
    out.append(t.T0_w @ _frame(0.0, 0.0, 0.0, 0.0, 0.0, -np.pi) @ t.Tw_e)  # yaw exactly at the seam
    out.extend(_band_sweep(t))
    return out


# Offsets past an upper bound that straddle the 1e-3 slack contains granted until #190: on
# the bound, inside the old band, and just beyond it. distance must grow continuously through
# all four -- it used to read 0, 0, 0 and then jump to 1.1e-3.
BAND_OFFSETS = (0.0, 2e-4, 9e-4, 1.1e-3)


def _band_sweep(t: TSR) -> list[np.ndarray]:
    """Poses at the chart's midpoint pushed past the upper bound of x, and of the first
    rotational row narrower than a turn, by each of ``BAND_OFFSETS``. Composed directly,
    since ``to_transform`` refuses out-of-bounds coordinates."""
    cont = t.continuous_bounds
    mid = (cont[:, 0] + cont[:, 1]) / 2.0
    rows = [0] + [i for i in range(3, 6) if cont[i, 1] - cont[i, 0] < 2 * np.pi][:1]
    out = []
    for i in rows:
        for d in BAND_OFFSETS:
            x = mid.copy()
            x[i] = cont[i, 1] + d
            out.append(t.T0_w @ TSR.xyzrpy_to_trans(x) @ t.Tw_e)
    return out


def chains() -> list[tuple[str, TSRChain]]:
    """The chain fixtures. Each exercises something a port can get wrong.

    Only the first link may carry a non-identity ``T0_w`` -- a later one is refused (#166) --
    so ``nonidentity_first_frame`` is the fixture that pins "only the first participates".
    """
    pi = np.pi
    identity = np.eye(4)
    free_yaw = np.array([[-0.1, 0.1], [0, 0], [0, 0], [0, 0], [0, 0], [-pi, pi]])
    return [
        # A door hinge with free yaw, then a handle a fixed distance along the door: the
        # motivating case from issue #165.
        (
            "door_hinge_handle",
            TSRChain(
                TSRs=[
                    TSR(_frame(1.0, 0.0, 0.8), _frame(0.6, 0.0, 0.0), np.array([[0, 0]] * 5 + [[-pi / 2, pi / 2]])),
                    TSR(identity, identity, np.array([[0, 0], [0, 0], [-0.05, 0.05], [0, 0], [-0.2, 0.2], [0, 0]])),
                ]
            ),
        ),
        (
            "two_yaws_and_a_slide",
            TSRChain(
                TSRs=[
                    TSR(_frame(0.3, 0.1, 0.5), _frame(0.2, 0, 0), free_yaw),
                    TSR(identity, _frame(0.15, 0, 0), np.array([[0, 0]] * 3 + [[-0.3, 0.3], [0, 0], [-pi, pi]])),
                ]
            ),
        ),
        (
            "three_links",
            TSRChain(
                TSRs=[
                    TSR(_frame(0.3, 0.1, 0.5), _frame(0.2, 0, 0), free_yaw),
                    TSR(identity, _frame(0.15, 0, 0), np.array([[0, 0]] * 3 + [[-0.3, 0.3], [0, 0], [-pi, pi]])),
                    TSR(identity, identity, np.array([[0, 0]] * 4 + [[-0.4, 0.4], [0, 0]])),
                ]
            ),
        ),
        # A full SO(3) ball on the second link: RPY goes through gimbal lock, which is where a
        # Jacobian-based inverse is hardest.
        (
            "rotation_rich",
            TSRChain(
                TSRs=[
                    TSR(_frame(0.3, 0.1, 0.5), _frame(0.2, 0, 0), free_yaw),
                    TSR(
                        identity,
                        _frame(0, 0, 0.1),
                        np.array([[-0.02, 0.02], [-0.02, 0.02], [0, 0], [-pi, pi], [-pi, pi], [-pi, pi]]),
                    ),
                ]
            ),
        ),
        # n == 1: solve reduces to the closed-form region check, both ways.
        (
            "single_link",
            TSRChain(
                TSR=TSR(
                    _frame(0.4, -0.2, 0.3, 0.1, 0.2, 0.3),
                    _frame(0.05, 0, 0),
                    np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-pi, pi]]),
                )
            ),
        ),
        # Every row zero-width: one candidate pose, no optimiser, and `free` must come out empty.
        (
            "all_fixed",
            TSRChain(
                TSRs=[
                    TSR(_frame(0.2, 0.2, 0.2, 0.1, 0.2, 0.3), _frame(0.1, 0, 0), np.zeros((6, 2))),
                    TSR(identity, _frame(0, 0.05, 0), np.zeros((6, 2))),
                ]
            ),
        ),
        # An outer rotational interval on the first link. The chart is [3pi/4, 3pi/4 + pi/2],
        # whose midpoint is exactly pi -- outside [-pi, pi), and it must stay that way.
        (
            "wrapping_yaw",
            TSRChain(
                TSRs=[
                    TSR(identity, _frame(0.3, 0, 0), np.array([[0, 0]] * 5 + [[3 * pi / 4, -3 * pi / 4]])),
                    TSR(identity, identity, np.array([[0, 0], [0, 0], [-0.1, 0.1], [2.5, -2.5], [0, 0], [0, 0]])),
                ]
            ),
        ),
        # A first link whose T0_w AND Tw_e are both non-identity and tilted.
        (
            "nonidentity_first_frame",
            TSRChain(
                TSRs=[
                    TSR(
                        _frame(0.5, -0.2, 0.3, 0.3, -0.4, 1.2),
                        _frame(0.1, 0.05, -0.02, 0.2, 0, 0),
                        np.array([[-0.2, 0.2], [0, 0], [0, 0], [0, 0], [-0.3, 0.3], [-1.0, 1.0]]),
                    ),
                    TSR(identity, _frame(0, 0, 0.12), np.array([[0, 0]] * 3 + [[0, 0], [0, 0], [-0.5, 0.5]])),
                ]
            ),
        ),
    ]


def _coordinate_probes(chain: TSRChain, rng: np.random.Generator) -> list[np.ndarray]:
    """Coordinate lists that exercise the chart: in bounds, at the corners, outside, and
    shifted by a full turn (which must wrap back to the same pose)."""
    cont = [t._Bw_cont for t in chain.TSRs]
    lower = np.array([c[:, 0] for c in cont])
    upper = np.array([c[:, 1] for c in cont])
    sampled = np.array(chain.sample_xyzrpy(rng=rng))
    turned = sampled.copy()
    turned[:, 3:6] += 2 * np.pi  # a full turn: same rotation, so the pose must not move
    unturned = sampled.copy()
    unturned[:, 3:6] -= 2 * np.pi
    out = [sampled, (lower + upper) / 2.0, lower, upper, lower - 0.1, upper + 0.1, turned, unturned]
    # A coordinate stated in [-pi, pi] that belongs to a wrapping interval: the #87 case, where
    # clipping without wrapping first would send it to an unrelated boundary.
    explicit = ((lower + upper) / 2.0).copy()
    explicit[0][5] = -3.0
    out.append(explicit)
    return out


def _witness_probes(chain: TSRChain, rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for _ in range(4):
        w = chain.sample_with_witness(rng=rng)
        out.append(
            {
                "coordinates": w.coordinates.tolist(),
                "target": w.pose.tolist(),
                "residual": float(chain._witness_residual(w.pose, w.coordinates)),
                "validate_witness": bool(chain.validate_witness(w.pose, w.coordinates)),
            }
        )
    # Coordinates that are in bounds but compose to a different pose, and coordinates far out
    # of bounds: both must be refused, for the two different reasons.
    w = chain.sample_with_witness(rng=rng)
    free = np.array([t._Bw_cont[:, 1] - t._Bw_cont[:, 0] for t in chain.TSRs]) > 0
    moved = w.coordinates.copy()
    if free.any():
        i, j = np.argwhere(free)[0]
        span = chain.TSRs[i]._Bw_cont[j, 1] - chain.TSRs[i]._Bw_cont[j, 0]
        moved[i][j] = chain.TSRs[i]._Bw_cont[j, 0] + 0.75 * span
    out.append(
        {
            "coordinates": moved.tolist(),
            "target": w.pose.tolist(),
            "residual": _or_none(chain._witness_residual(w.pose, moved)),
            "validate_witness": bool(chain.validate_witness(w.pose, moved)),
        }
    )
    out_of_bounds = w.coordinates + 10.0
    out.append(
        {
            "coordinates": out_of_bounds.tolist(),
            "target": w.pose.tolist(),
            "residual": _or_none(chain._witness_residual(w.pose, out_of_bounds)),
            "validate_witness": bool(chain.validate_witness(w.pose, out_of_bounds)),
        }
    )
    return out


def _or_none(value) -> float | None:
    """A residual the Python reports as absent (not a witness) or infinite, as JSON null.

    ``json.dumps(float('inf'))`` emits a bare ``Infinity``, which is not JSON; the C++ test
    reader would hand it to ``strtod``, whose acceptance of that spelling is libc-dependent.
    """
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _solve_record(path: str, chain: TSRChain, target: np.ndarray, **kw: Any) -> dict[str, Any]:
    guess = kw.pop("initial_guess", None)
    result = chain.solve(target, initial_guess=guess, **kw)
    return {
        "path": path,
        "target": target.tolist(),
        "initial_guess": None if guess is None else np.asarray(guess).tolist(),
        "max_starts": int(kw.get("max_starts", TSRChain._DEFAULT_MAX_STARTS)),
        "status": result.status,
        "coordinates": np.asarray(result.coordinates).tolist(),
        "residual": _or_none(result.residual),
        "nfev": int(result.nfev),
        "starts": int(result.starts),
    }


def _exact_solves(name: str, chain: TSRChain, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Only the paths that reach an answer without the optimiser.

    Everything else -- and therefore ``distance``, ``closest_transform``, ``contains`` and
    ``to_xyzrpy`` for a chain of two or more links with any free coordinate -- runs the cold
    inverse, whose iterates are optimiser-specific and deliberately NOT recorded. See the
    "What they do not agree on" section of docs/CPP.md.
    """
    out = []
    w = chain.sample_with_witness(rng=rng)
    out.append(_solve_record("warm_witness", chain, w.pose, initial_guess=w.coordinates))

    free = any((t._Bw_cont[:, 1] > t._Bw_cont[:, 0]).any() for t in chain.TSRs)
    if len(chain.TSRs) == 1:
        inside = chain.sample(rng=rng)
        out.append(_solve_record("single_tsr_contained", chain, inside))
        outside = _frame(3.0, -2.0, 1.0, 0.4, 0.2, -0.8)
        out.append(_solve_record("single_tsr_outside", chain, outside))
    elif not free:
        exact = chain.to_transform(np.array([t._Bw_cont[:, 0] for t in chain.TSRs]))
        out.append(_solve_record("all_fixed_satisfied", chain, exact))
        out.append(_solve_record("all_fixed_not_found", chain, _frame(4.0, 4.0, 4.0)))
    else:
        out.append(_solve_record("zero_starts", chain, chain.sample(rng=rng), max_starts=0))
    return out


def record_chain(name: str, chain: TSRChain, rng: np.random.Generator) -> dict[str, Any]:
    cont = [t._Bw_cont for t in chain.TSRs]
    lower = np.array([c[:, 0] for c in cont])
    upper = np.array([c[:, 1] for c in cont])
    entry: dict[str, Any] = {
        "name": name,
        "links": [
            {
                "T0_w": t.T0_w.tolist(),
                "Tw_e": t.Tw_e.tolist(),
                "Bw": t.Bw.tolist(),
                "continuous_bounds": t.continuous_bounds.tolist(),
            }
            for t in chain.TSRs
        ],
        "free_mask": (upper > lower).reshape(-1).tolist(),
        "midpoint": ((lower + upper) / 2.0).tolist(),
        "coordinate_probes": [
            {
                "input": c.tolist(),
                "continuous": chain._to_continuous(c).tolist(),
                "is_valid": [[bool(v) for v in row] for row in chain.is_valid(c)],
                "to_transform": chain.to_transform(c).tolist(),
            }
            for c in _coordinate_probes(chain, rng)
        ],
        "witness_probes": _witness_probes(chain, rng),
        "exact_solves": _exact_solves(name, chain, rng),
    }
    # The four delegating methods are only recordable where solve is exact: a single link, or
    # a chain with no free coordinate at all.
    free = any((t._Bw_cont[:, 1] > t._Bw_cont[:, 0]).any() for t in chain.TSRs)
    if len(chain.TSRs) == 1 or not free:
        target = chain.sample(rng=rng)
        dist, coords = chain.distance(target)
        cdist, closest = chain.closest_transform(target)
        entry["delegating"] = {
            "target": target.tolist(),
            "distance": _or_none(dist),
            "distance_coordinates": np.asarray(coords).tolist(),
            "closest_distance": _or_none(cdist),
            "closest_transform": closest.tolist(),
            "contains": bool(chain.contains(target)),
            "to_xyzrpy": [row.tolist() for row in chain.to_xyzrpy(target)],
        }
    return entry


def record_empty_chain() -> dict[str, Any]:
    """An empty chain, whose answers are asymmetric and easy to get wrong.

    ``closest_transform`` raises (there is no pose to compose) while ``to_xyzrpy`` returns an
    empty list; ``distance`` reports an infinite residual; ``validate_witness`` is False
    rather than an error.
    """
    chain = TSRChain()
    result = chain.solve(np.eye(4))
    dist, coords = chain.distance(np.eye(4))
    try:
        chain.closest_transform(np.eye(4))
        closest_raises = False
    except ValueError:
        closest_raises = True
    return {
        "solve": {
            "status": result.status,
            "coordinates": np.asarray(result.coordinates).tolist(),
            "residual": _or_none(result.residual),
            "nfev": int(result.nfev),
            "starts": int(result.starts),
        },
        "distance": _or_none(dist),
        "distance_coordinates": np.asarray(coords).tolist(),
        "closest_transform_raises": closest_raises,
        "to_xyzrpy": [row.tolist() for row in chain.to_xyzrpy(np.eye(4))],
        "validate_witness": bool(chain.validate_witness(np.eye(4), np.zeros((0, 6)))),
        "contains": bool(chain.contains(np.eye(4))),
    }


def record(name: str, t: TSR, rng: np.random.Generator) -> dict[str, Any]:
    entries = []
    for T in probes(t, rng):
        dist, closest = t.closest_transform(T)
        entries.append(
            {
                "T": T.tolist(),
                "contains": bool(t.contains(T)),
                "distance": float(dist),
                "closest": closest.tolist(),
                "closest_contained": bool(t.contains(closest)),
                # Recorded since #171, which hid precisely because it was not. `to_xyzrpy`
                # chooses between the two RPY representatives of the same rotation, and that
                # choice is invisible in `contains`, `distance` and `closest_transform` -- so a
                # port could disagree about it while matching every other recorded answer.
                "to_xyzrpy": t.to_xyzrpy(T).tolist(),
                "to_xyzrpy_is_valid": [bool(v) for v in t.is_valid(t.to_xyzrpy(T))],
            }
        )
    xy = t.sample_xyzrpy(rng=np.random.default_rng(7))
    return {
        "name": name,
        "T0_w": t.T0_w.tolist(),
        "Tw_e": t.Tw_e.tolist(),
        "Bw": t.Bw.tolist(),
        "continuous_bounds": t.continuous_bounds.tolist(),
        "volume": t.volume,
        "sample_xyzrpy_seed7_python": xy.tolist(),
        "probes": entries,
    }


def generate() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    chain_rng = np.random.default_rng(CHAIN_SEED)
    return {
        "artifact": "sstsr conformance corpus for the native TSR runtime",
        "issue": "https://github.com/personalrobotics/sscbirrt/issues/87",
        "seed": SEED,
        "chain_seed": CHAIN_SEED,
        "versions": {"sstsr": importlib.metadata.version("sstsr"), "numpy": np.__version__},
        "regions": [record(name, t, rng) for name, t in regions()],
        "chain_defaults": {
            "max_starts": TSRChain._DEFAULT_MAX_STARTS,
            "max_nfev": TSRChain._DEFAULT_MAX_NFEV,
            "tolerance": CHAIN_TOLERANCE,
        },
        "empty_chain": record_empty_chain(),
        "chains": [record_chain(name, c, chain_rng) for name, c in chains()],
    }


ATOL = 1e-9  # distances and transforms compare within this; sstsr on another platform differs in the last bits

# A geodesic residual needs a looser floor than everything else, and not for sloppiness. Its
# rotation term is acos((trace - 1) / 2), and acos is infinitely steep at 1: for two poses that
# agree to the last bit, a 1-ulp error in the trace becomes acos(1 - 2.2e-16) = 2.1e-08 of
# reported angle. Re-associating an algebraically identical product moves it by up to 4.2e-08.
# So a near-zero residual is only meaningful to about 1e-7, and comparing one at ATOL would fail
# on a different libm while nothing was actually wrong.
RESIDUAL_ATOL = 1e-7


def mismatches(stored: dict[str, Any], fresh: dict[str, Any], atol: float = ATOL) -> list[str]:
    """Where a fresh sstsr run disagrees with the stored corpus: booleans exactly, floats within atol."""
    out = []
    by_name = {r["name"]: r for r in stored["regions"]}
    for r in fresh["regions"]:
        old = by_name.get(r["name"])
        if old is None:
            out.append(f"{r['name']}: not in the stored corpus")
            continue
        if len(old["probes"]) != len(r["probes"]):
            out.append(f"{r['name']}: {len(old['probes'])} stored probes vs {len(r['probes'])} fresh")
            continue
        for i, (x, y) in enumerate(zip(old["probes"], r["probes"])):
            where = f"{r['name']}[{i}]"
            if x["contains"] != y["contains"]:
                out.append(f"{where}.contains: stored={x['contains']} fresh={y['contains']}")
            if abs(x["distance"] - y["distance"]) > atol:
                out.append(f"{where}.distance: stored={x['distance']!r} fresh={y['distance']!r}")
            if not np.allclose(x["closest"], y["closest"], atol=atol, rtol=0.0):
                diff = np.abs(np.array(x["closest"]) - np.array(y["closest"])).max()
                out.append(f"{where}.closest: max |diff| = {diff:.3g}")
            if x["closest_contained"] != y["closest_contained"]:
                out.append(f"{where}.closest_contained: stored={x['closest_contained']} fresh={y['closest_contained']}")
            if not _close(x["to_xyzrpy"], y["to_xyzrpy"], atol):
                out.append(f"{where}.to_xyzrpy moved")
            if x["to_xyzrpy_is_valid"] != y["to_xyzrpy_is_valid"]:
                out.append(
                    f"{where}.to_xyzrpy_is_valid: stored={x['to_xyzrpy_is_valid']} fresh={y['to_xyzrpy_is_valid']}"
                )
    out += _chain_mismatches(stored, fresh, atol)
    return out


def _close(a: Any, b: Any, atol: float) -> bool:
    """Nested float comparison that treats a recorded ``null`` as "absent or infinite"."""
    if a is None or b is None:
        return a is None and b is None
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=atol, rtol=0.0))


#: Residual fields carry RESIDUAL_ATOL; everything else the chain records is ordinary arithmetic.
_RESIDUAL_KEYS = frozenset({"residual", "distance", "closest_distance"})


def _tol_for(key: str, atol: float) -> float:
    return max(atol, RESIDUAL_ATOL) if key in _RESIDUAL_KEYS else atol


def _chain_mismatches(stored: dict[str, Any], fresh: dict[str, Any], atol: float) -> list[str]:
    """Where a fresh run disagrees with the stored corpus about the chains.

    Separate from the region walker only for length. Without it ``--check`` would pass on any
    chain regression, which would make half the corpus decorative -- the corpus exists so a
    rule change shows up as a diff rather than as a silently different expectation.
    """
    out: list[str] = []
    for key in ("chain_defaults", "empty_chain"):
        if stored.get(key) != fresh[key]:
            out.append(f"{key}: stored={stored.get(key)!r} fresh={fresh[key]!r}")

    by_name = {c["name"]: c for c in stored.get("chains", [])}
    for c in fresh["chains"]:
        old = by_name.get(c["name"])
        if old is None:
            out.append(f"chain {c['name']}: not in the stored corpus")
            continue
        where = f"chain {c['name']}"
        for key in ("free_mask", "links"):
            if old[key] != c[key]:
                out.append(f"{where}.{key} changed")
        if not _close(old["midpoint"], c["midpoint"], atol):
            out.append(f"{where}.midpoint moved")

        for section, exact_keys, float_keys in (
            ("coordinate_probes", ("is_valid",), ("continuous", "to_transform")),
            ("witness_probes", ("validate_witness",), ("coordinates", "target", "residual")),
            (
                "exact_solves",
                ("path", "status", "nfev", "starts", "max_starts"),
                ("coordinates", "residual", "target"),
            ),
        ):
            if len(old[section]) != len(c[section]):
                out.append(f"{where}.{section}: {len(old[section])} stored vs {len(c[section])} fresh")
                continue
            for i, (x, y) in enumerate(zip(old[section], c[section])):
                for key in exact_keys:
                    if x[key] != y[key]:
                        out.append(f"{where}.{section}[{i}].{key}: stored={x[key]!r} fresh={y[key]!r}")
                for key in float_keys:
                    if not _close(x[key], y[key], _tol_for(key, atol)):
                        out.append(f"{where}.{section}[{i}].{key} moved")

        if ("delegating" in old) != ("delegating" in c):
            out.append(f"{where}.delegating: present in only one of stored/fresh")
        elif "delegating" in c:
            x, y = old["delegating"], c["delegating"]
            if x["contains"] != y["contains"]:
                out.append(f"{where}.delegating.contains: stored={x['contains']} fresh={y['contains']}")
            for key in (
                "target",
                "distance",
                "distance_coordinates",
                "closest_distance",
                "closest_transform",
                "to_xyzrpy",
            ):
                if not _close(x[key], y[key], _tol_for(key, atol)):
                    out.append(f"{where}.delegating.{key} moved")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args(argv)
    fresh = generate()
    if args.check:
        stored = json.loads(args.output.read_text())
        bad = mismatches(stored, fresh)
        if bad:
            print("MISMATCH: sstsr's results on the conformance corpus changed:", file=sys.stderr)
            for line in bad:
                print(f"  {line}", file=sys.stderr)
            return 1
        print(f"conformance corpus unchanged ({_counts(fresh)}, within {ATOL}; versions {fresh['versions']})")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fresh, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.output} ({_counts(fresh)}, versions {fresh['versions']})")
    return 0


def _counts(corpus: dict[str, Any]) -> str:
    probes = sum(len(r["probes"]) for r in corpus["regions"])
    coords = sum(len(c["coordinate_probes"]) for c in corpus["chains"])
    witnesses = sum(len(c["witness_probes"]) for c in corpus["chains"])
    solves = sum(len(c["exact_solves"]) for c in corpus["chains"])
    return (
        f"{len(corpus['regions'])} regions, {probes} probes; {len(corpus['chains'])} chains, "
        f"{coords} coordinate probes, {witnesses} witness probes, {solves} exact solves"
    )


if __name__ == "__main__":
    sys.exit(main())
