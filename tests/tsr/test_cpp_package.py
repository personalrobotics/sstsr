# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""The C++ core is findable, and its copied constants still match the Python's (#165).

``cpp/`` is a second implementation of the rules in ``src/tsr/core/``. Agreement on
*behaviour* is the conformance corpus, checked by ctest; this file covers the two things
a C++ test cannot see:

* the package is **findable** — a planner locates the headers and the CMake package by
  asking the interpreter, so those entry points have to work from a wheel and say
  something useful from a checkout;
* the handful of **constants the C++ necessarily restates** still equal the Python's. A
  tolerance that drifts does not fail the corpus — it fails much later, as two
  implementations disagreeing about a boundary case, which is the failure mode #162 made
  `FRAME_ATOL` public to prevent.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import tsr
from tsr import ANGLE_ATOL, CHAIN_TOLERANCE, FRAME_ATOL, TSRChain
from tsr.core.utils import geodesic_distance


def _cpp_source(name: str) -> str:
    """A C++ source or header, from wherever the headers were found."""
    include = Path(tsr.get_include())
    for candidate in (include / "sstsr" / name, include.parent / "src" / name):
        if candidate.is_file():
            return candidate.read_text()
    pytest.skip(f"{name} is not in this install")  # a wheel ships both; be explicit if not


def _literal(source: str, declaration: str) -> float:
    """The floating-point literal a `constexpr double <name> = <value>;` declares."""
    match = re.search(rf"{re.escape(declaration)}\s*=\s*([0-9.eE+-]+)\s*;", source)
    assert match, f"no declaration of {declaration!r} to compare against"
    return float(match.group(1))


def test_the_headers_are_findable():
    include = Path(tsr.get_include())
    assert (include / "sstsr" / "tsr.hpp").is_file()
    assert (include / "sstsr" / "tsr_chain.hpp").is_file()
    assert (include / "sstsr" / "transform.hpp").is_file()
    assert (include / "sstsr" / "rng.hpp").is_file()


def test_the_cmake_package_is_findable_or_says_why_not():
    """Only a built wheel carries the CMake package, because a relocatable targets file
    has to be rendered. From a checkout the failure must name that, not just be absent."""
    try:
        directory = Path(tsr.get_cmake_dir())
    except FileNotFoundError as e:
        assert "wheel" in str(e), f"the error should say why there is no package: {e}"
        return
    assert (directory / "sstsr_cppConfig.cmake").is_file()
    assert (directory / "sstsr_cppTargets.cmake").is_file()
    assert (directory / "sstsr_cppConfigVersion.cmake").is_file()


def test_a_missing_tree_is_reported_rather_than_guessed(tmp_path, monkeypatch):
    """`get_include` probes a known header; if it is gone the install is broken and must
    say so, not return a plausible-looking empty directory."""
    import tsr.cpp as cpp_locators

    monkeypatch.setattr(cpp_locators, "_HERE", tmp_path)
    with pytest.raises(FileNotFoundError, match="headers are not installed"):
        cpp_locators.get_include()


def test_the_frame_tolerance_is_the_same_number_in_both():
    """`FRAME_ATOL` is public precisely so the two implementations reject the same frames
    at the same boundary (#162)."""
    assert _literal(_cpp_source("transform.hpp"), "constexpr double kFrameAtol") == FRAME_ATOL


def test_the_membership_round_off_is_the_same_in_both():
    """``contains`` has no slack beyond round-off (#190), so the round-off tolerances decide
    every boundary verdict: a drift would split the two implementations exactly there."""
    source = _cpp_source("tsr.hpp")
    assert _literal(source, "constexpr double kAngleAtol") == ANGLE_ATOL
    assert re.search(r"return 1e-9 \+ 1e-6 \* scale;", source), "length_atol drifted from tsr.core.utils"
    assert "return 1e-9 + 1e-6 * float(scale)" in inspect.getsource(tsr.core.utils.length_atol)


def test_the_chain_tolerance_is_the_same_number_in_both():
    assert _literal(_cpp_source("tsr.hpp"), "constexpr double kChainTolerance") == CHAIN_TOLERANCE


def test_the_gimbal_threshold_is_the_same_number_in_both():
    """Restated in two C++ translation units, so check both: one drifting would move the
    gimbal-lock branch in only half the code."""
    from tsr.core.tsr import _GIMBAL_EPSILON

    for name in ("transform.cpp", "tsr.cpp"):
        assert _literal(_cpp_source(name), "constexpr double kGimbalEpsilon") == _GIMBAL_EPSILON, name


def test_the_default_solve_budget_is_the_same_in_both():
    """The chain's inverse-solve budget (#165 stage 2). It is caller-visible -- a result reports
    the starts and evaluations it spent against it -- so a drift would make the two
    implementations disagree about how hard they tried before answering ``not_found``."""
    source = _cpp_source("tsr_chain.hpp")
    assert _literal(source, "constexpr int kDefaultMaxStarts") == TSRChain._DEFAULT_MAX_STARTS
    assert _literal(source, "constexpr int kDefaultMaxNfev") == TSRChain._DEFAULT_MAX_NFEV


def test_the_chain_solve_tolerance_is_the_librarys_chain_tolerance_in_both():
    """Both sides must take the default tolerance from the one named constant rather than
    restate it. ``kChainTolerance`` is already drift-checked against ``CHAIN_TOLERANCE`` above,
    so spelling the C++ default as ``kChainTolerance`` keeps a single literal in play."""
    source = _cpp_source("tsr_chain.hpp")
    assert re.search(
        r"double tolerance\s*=\s*kChainTolerance\s*;", source
    ), "SolveOptions::tolerance should default to kChainTolerance, not a restated number"
    assert inspect.signature(TSRChain.solve).parameters["tolerance"].default == CHAIN_TOLERANCE


def test_the_geodesic_rotation_weight_is_still_one():
    """The C++ ``geodesic_distance`` takes no rotation weight, because nothing in the library
    passes anything but 1.0 -- the same reason the C++ TSR omits ``rotation_weight``. If the
    Python's default ever moved, the omission would silently become a divergence instead of a
    simplification, and no corpus probe would show it."""
    assert inspect.signature(geodesic_distance).parameters["r"].default == 1.0


def test_the_cpp_package_states_no_hand_maintained_version():
    """#175: the version comes from the git tag, so a literal in CMake could only be stale.

    ``docs/RELEASING.md`` states the invariant -- "Do not edit a version number anywhere; there
    is none to edit" -- and ``project(sstsr_cpp VERSION ...)`` was the single exception to it.
    The two packagings rendered the number independently, so they drifted: a source install said
    3.2.0 while the wheel said 3.2.1, and a consumer pinning ``find_package(sstsr_cpp 3.3)``
    would have got different answers from a checkout and from the wheel it installs.

    Checked textually because the failure is a literal reappearing, which no behavioural test
    sees until a consumer pins a version and gets the wrong answer.
    """
    cmakelists = Path(tsr.get_include()).parent / "CMakeLists.txt"
    if not cmakelists.is_file():  # a wheel ships the sources, not the build file
        cmakelists = Path(__file__).resolve().parents[2] / "cpp" / "CMakeLists.txt"
    if not cmakelists.is_file():
        pytest.skip("cpp/CMakeLists.txt is not in this install")
    text = cmakelists.read_text()
    assert re.search(
        r"^project\(sstsr_cpp\s+LANGUAGES", text, re.MULTILINE
    ), "project(sstsr_cpp ...) must declare no VERSION: the release version comes from the tag"
    assert (
        "write_basic_package_version_file" not in text
    ), "a source install must not render a ConfigVersion file from a literal version"


def test_the_wheels_cmake_version_is_the_release_version():
    """The wheel is the one packaging that can state a version, and it must be *the* version.

    ``hatch_build.py`` renders it from the package metadata, so this catches that renderer
    drifting from what the tag produced.
    """
    try:
        directory = Path(tsr.get_cmake_dir())
    except FileNotFoundError:
        pytest.skip("only a built wheel carries the CMake package")
    version_file = directory / "sstsr_cppConfigVersion.cmake"
    assert version_file.is_file()
    match = re.search(r'set\(PACKAGE_VERSION "([^"]+)"\)', version_file.read_text())
    assert match, "the rendered ConfigVersion file should set PACKAGE_VERSION"
    expected = re.match(r"^(\d+(?:\.\d+)*)", tsr.__version__).group(1)
    assert match.group(1) == expected, f"the wheel's CMake package says {match.group(1)} but sstsr is {tsr.__version__}"
