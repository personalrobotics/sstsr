# The C++ core

`cpp/` is a second implementation of the pose region in `src/tsr/core/tsr.py` and the chain in
`src/tsr/core/tsr_chain.py`, for a planner that needs TSRs in C++ without paying a Python call
per edge sample.

**The Python is the source of truth for the rules.** They are stated in
[ARCHITECTURE.md](ARCHITECTURE.md); the C++ is held to them by a checked-in corpus of the
Python's own answers. Neither implementation reads the other.

## Using it

From C++, ask the installed Python package where the CMake package is — the same pattern
`sscbirrt` already uses for `ssik_cpp`:

```cmake
find_package(Python COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import tsr; print(tsr.get_cmake_dir(), end='')"
  OUTPUT_VARIABLE TSR_CMAKE_DIR RESULT_VARIABLE rc ERROR_QUIET)
if(rc EQUAL 0 AND TSR_CMAKE_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${TSR_CMAKE_DIR}")
endif()
find_package(sstsr_cpp CONFIG REQUIRED)
target_link_libraries(my_planner PRIVATE sstsr::sstsr_cpp)
```

```cpp
#include "sstsr/tsr.hpp"

sstsr::Bounds6 Bw{};
Bw.rows[0] = {-0.05, 0.05};
Bw.rows[5] = {-M_PI, M_PI};
const sstsr::TSR region(T0_w, Tw_e, Bw);

if (!region.contains(pose)) {
  const auto [distance, closest] = region.closest_transform(pose);
}
```

A chain composes several regions. Sampling hands back the coordinates that built the pose, and
those coordinates are a *constructive witness*: membership is then one bounds check and one
composition, with no optimiser involved.

```cpp
#include "sstsr/tsr_chain.hpp"

const sstsr::TSRChain chain({hinge, handle});   // only the first link's T0_w is read

sstsr::Rng rng(7);
const sstsr::ChainSample s = chain.sample_with_witness(rng);
assert(chain.validate_witness(s.pose, s.coordinates));

// Along a planner edge the neighbouring state's coordinates are a witness already, so the
// solve takes the exact path: nfev == 0, starts == 0, no search.
const auto result = chain.solve(s.pose, {.initial_guess = neighbour});
```

`cpp/examples/consumer` and `cpp/examples/wheel_consumer` are working versions of both
routes, and are run in CI.

## Two packagings, one config

The wheel stays `py3-none-any`. It carries the C++ **sources and headers**, not a
compiled library, so a consumer compiles them into its own build. That keeps one
artifact, one release pipeline and one install story.

| | consumed how | targets file | version |
|---|---|---|---|
| `cmake --install` of `cpp/` | `find_package(sstsr_cpp)` on the install prefix | exported by CMake; a real static library | **none** |
| the `sstsr` wheel | `find_package` on `tsr.get_cmake_dir()` | `cpp/cmake/sstsr_cppWheelTargets.cmake`, hand-written and relocatable; an `INTERFACE` target carrying the sources | the release version |

Both go through the one `cpp/cmake/sstsr_cppConfig.cmake.in`. The wheel needs its own
targets file because an exported one holds absolute paths from the machine that built it,
while a wheel lands wherever the virtualenv is. `hatch_build.py` renders it, and
`cpp/examples/wheel_consumer` is the only thing that reads it — including the version
file, which is why that example asks `find_package` for a version.

### Only the wheel states a version (#175)

`project()` in `cpp/CMakeLists.txt` declares no `VERSION`, and a source install ships no
`ConfigVersion` file. That is deliberate. The version comes from the git tag through
hatch-vcs — [RELEASING.md](RELEASING.md) puts it plainly, *"Do not edit a version number
anywhere; there is none to edit"* — so a bare checkout has no version to state and a literal
here could only be stale. It was, and it made this the one hand-maintained version in the
repository: a source install said `3.2.0` while the wheel said `3.2.1`.

So:

```cmake
find_package(sstsr_cpp CONFIG REQUIRED)        # works from either packaging
find_package(sstsr_cpp 3.3 CONFIG REQUIRED)    # works from the wheel; from a source
                                               # install, fails with "version: unknown"
```

Failing is the point. A consumer that needs the chain pins a version, and answering it with a
stale number from a checkout while the installed wheel answers differently is the one outcome
worth preventing — it passes in development and fails on install. If you need to pin, consume
the wheel; `tests/tsr/test_cpp_package.py` holds both halves of this in place.

`get_cmake_dir()` raises from an editable install: only a built wheel carries the CMake
package. Configure `cpp/` directly in that case, as `cpp/examples/consumer` does.

## No dependencies, on purpose

Standard library only, C++20. A 4×4 transform is `std::array<double, 16>` row-major and
the generator is `std::mt19937_64`. A consumer that wants a TSR should not inherit a
linear-algebra dependency to hold a pose, and that is also what lets
`sstsr_cppConfig.cmake` carry no `find_dependency` at all. If that stops being true, that
config file is the first thing that has to change.

## What the two implementations agree on

Everything in the region, checked on 282 probes across 11 regions, and the forward half of the
chain, checked on 72 coordinate probes, 48 witness probes and 18 exact solves across 8 chains
(`tests/reference/tsr_conformance.json`, generated by `tools/tsr_conformance.py`, verified
by `cpp/tests/test_conformance.cpp`):

| | tolerance | why |
|---|---|---|
| the continuous-bounds chart, `volume`, the chain's chart and midpoint | `1e-12` | arithmetic on the inputs |
| `contains`, `is_valid`, `validate_witness`, a solve's `status`, `nfev` and `starts`, and whether a closest pose is itself contained | exact | verdicts and counters |
| `distance`, `closest_transform`, the chain's `to_transform` and recovered coordinates | `1e-9` | through `asin`/`atan2` and a nine-candidate minimum |
| any **geodesic residual** | `1e-7` | see below |

A geodesic residual needs the looser floor, and not for sloppiness. Its rotation term is
`acos((trace − 1) / 2)`, and `acos` is infinitely steep at 1: for two poses that agree to the
last bit, a 1-ulp error in the trace is reported as `acos(1 − 2.2e-16) = 2.1e-08` of angle, and
merely re-associating an algebraically identical product moves it by up to `4.2e-08`. So a
near-zero residual is only meaningful to about `1e-7`, and comparing one at `1e-9` would fail on
a different libm while nothing was wrong. `RESIDUAL_ATOL` in the generator and `kResidualTol` in
the C++ test are the same number for this reason.

The constants the C++ necessarily restates — `kFrameAtol`, `kAngleAtol` and `length_atol`,
`kChainTolerance`, `kGimbalEpsilon`, `kDefaultMaxStarts`, `kDefaultMaxNfev` — are checked
against the Python's in `tests/tsr/test_cpp_package.py`, which also pins the chain's default
tolerance to `kChainTolerance`
rather than a third copy of `0.001`, and pins the Python's geodesic rotation weight at 1.0 since
the C++ omits that parameter. A drifting constant does not fail the corpus; it fails much later,
as two implementations disagreeing about a boundary case.

## What they do not agree on, and why

**Sampling draws from a different engine.** `sstsr::unit` takes the top 53 bits of
`std::mt19937_64` directly, because the standard library's distributions differ between
libstdc++ and libc++; numpy's `Generator` is a different engine again. So a seeded sample
is reproducible *within* an implementation, not across the two. Sampling is therefore
checked as a property — in bounds, and contained — not against recorded coordinates. The
corpus records `sample_xyzrpy_seed7_python` for reference; nothing asserts against it.

**A chain's cold inverse is specified by properties, not by numbers.** The chain is therefore
split in two halves, and only one of them is in the corpus.

The forward half **is** exactly specifiable, and all of it is checked: the continuous chart,
`to_transform`, the witness path (`sample_with_witness`, `validate_witness`), the argument
validation, and the `solve` paths that reach an answer without searching — an empty chain, a
warm validating witness, a single link, an all-fixed chain, and `max_starts == 0` — including
their `nfev == 0, starts == 0` accounting.

The **cold** inverse is not. `solve` reports the best point found by L-BFGS-B, and "best" is
taken over the optimiser's iterates *plus* its `epsilon=1e-8` finite-difference probes. A
different optimiser evaluates a different set of points, so its `status`, `residual`,
`coordinates` and `nfev` differ with no bound relating them beyond "both are ≥ the true
minimum". That is already the documented position — `ChainSolveResult.residual` is an upper
bound and not a certified distance, and `"not_found"` is not a proof of non-membership (issue
#85) — so it is a property of the problem, not of the code.

So the cold half is held to properties instead: `nfev <= max_nfev` strictly,
`starts <= max_starts`, `residual == geodesic_distance(to_transform(coordinates), T)`,
coordinates within the chart, `"satisfied" ⟹ residual < tolerance`, and run-to-run determinism
within one implementation.

Three of those properties carry a **documented exception**, each because the Python genuinely
behaves that way and the corpus pins it:

| exception | why |
|---|---|
| `"satisfied" ⟹ residual < tolerance` holds only for two or more links | the single-link path never re-checks the tolerance. Since #190 `contains` absorbs only round-off, so the residual is round-off too, but a caller tolerance below `kAngleAtol` could still see it. (Until #190 `contains` granted `1e-3` per coordinate, and residuals up to `2.4e-3` were measured against a `1e-3` tolerance.) |
| the single-link `not_found` residual is not a geodesic | it is the region's 6-vector displacement norm. The two coincide for a pure translation or a single-axis rotation and diverge otherwise — `3.464` against a geodesic of `1.435` for a three-axis rotation |
| the warm path's coordinates need not lie in the chart | they are returned **verbatim**, un-canonicalised. A guess within round-off (say `5e-7` rad) past a zero-width bound still passes `is_valid`, so it validates and comes back outside the chart. The residual is still the geodesic at those coordinates, because the forward map reapplies the chart |

"Correcting" any of these inside the port would make the two implementations disagree. If they
are worth changing, they are worth changing in the Python first.

The interior start-schedule fill also differs. The Python fills from numpy's `default_rng(0)`
(PCG64) and the C++ from its own `Rng` (`mt19937_64`), for the same reason sampling differs.
The starts that *are* specifiable — a canonicalised guess, the midpoint, the two opposite
corners — are arithmetic on the chart and agree.

### Both implementations now run the same algorithm, and it changes nothing here (#174)

The Python's cold inverse was L-BFGS-B over a finite-difference gradient; since #174 it is the
same projected Levenberg–Marquardt with the same analytic Jacobian as `cpp/`, for its own
reasons — measured over 560 cold solves on 70 random chains, 98.6% recall against 98.2% at
7.8× fewer objective evaluations and 3.1× faster.

**Sharing the algorithm does not make the cold inverse recordable in the corpus**, and it is
worth being explicit because the opposite is the natural assumption. When two solvers both
succeed they recover the *same coordinates* about 1% of the time — a redundant chain's solution
set is a continuum, and both answers are correct, each recomposing to the target within
tolerance. There is no single right answer to record. Nor would divergence be graceful: two
implementations of one algorithm track each other until a near-tie accept/reject flips, and then
land on different points in that continuum, which given the gaps between valid solutions
(median 0.094 rad, up to 6.28) is a wholly different recorded answer rather than a small
numerical difference. No comparison tolerance makes that meaningful.

So the cold half stays property-checked whichever optimiser either side uses, and
`tools/chain_solver_comparison.py` is how the recall claim is reproduced rather than asserted.

One existing test cannot cross over:
`test_deterministic_counterexample_witness_certifies_membership` asserts that the cold solve
**fails** on a pose that is provably a member. It pins a property of scipy's trajectory, and our
projected Levenberg–Marquardt *solves* that pose — so a port would fail that test while being
more correct. It stays Python-only.

### The cold inverse

`cpp/src/chain_solver.hpp` is **projected Levenberg–Marquardt with an analytic Jacobian**. It is
internal — under `src/`, not `include/` — because a consumer calls `TSRChain::solve`; it is a
header so the test can check the Jacobian against central differences.

Why least squares, and not an approximation of the Python's objective but the same one:
`3 − tr(AᵀB) = ‖A − B‖²_F / 2` for `A, B ∈ SO(3)`, so the Python's scalar
`Δt·Δt + (3 − tr(RᵀR′))` **is** `‖r‖²` for the 12-vector `r = [Δt, (R − R′)/√2]`.

Two details are load-bearing, each established by ablation rather than assumed:

| removed | `rotation_rich` recall | mean evaluations |
|---|---|---|
| nothing (as implemented) | **40/40** | 95.7 |
| the analytic Jacobian, finite differences instead | 18/40 *(measured in the Python prototype)* | — |
| the gradient projection | **17/40** | 1206, i.e. budget-bound |

The gradient projection holds a coordinate pinned at a bound whose gradient pushes further out,
which is what keeps a rank-deficient direction at RPY gimbal lock from stalling the whole solve.
Dropping it also costs `two_yaws_and_a_slide` 4× the evaluations (191.5 against 44.4) while still
finding every pose, so recall alone does not show how much it does.

It finds the pose the Python's cold solve cannot — the deterministic counterexample of issue #85,
solved on the first start in 21 evaluations at a residual of `1.4e-17`. That is the concrete
reason `test_deterministic_counterexample_witness_certifies_membership` cannot cross over: it
asserts the cold solve **fails** on a provable member, so it pins SciPy's trajectory rather than
any rule. Being better is still not a licence to record cold results in the corpus — `not_found`
is not a proof of non-membership in either implementation.

What the properties do **not** cover, and why that is the honest state rather than an omission:
the projected step's clamp is not load-bearing for correctness. `coords_of` runs every point
through the chart map, so the returned coordinates cannot leave the chart whether or not the step
was clamped; the clamp keeps `x` and the evaluated point the same point, which costs convergence
when removed, not correctness.

One property that looks true and is not: **the reported residual is not monotone in the budget.**
The search minimises the chordal objective because it is smooth, and reports the geodesic at
whichever point won. Those are different orderings, so a later evaluation can lower `‖r‖²` and
raise the geodesic. The objective *is* monotone, and that is what the test asserts — the Python
has the same property for the same reason.

The recall floor (≥90% aggregate, ≥85% per chain, against a measured 160/160) writes
`chain_properties_report.txt` into the build directory as its inspectable artifact. Reproduce it
with `ctest --test-dir build -R tsr_chain`.

## Working on it

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure

uv run python tools/tsr_conformance.py            # regenerate the corpus
uv run python tools/tsr_conformance.py --check    # still the same answers?

./cpp/examples/consumer/smoke.sh                  # the installed package
./cpp/examples/wheel_consumer/smoke.sh            # the package inside a wheel
```

Changing a rule means changing the Python, regenerating the corpus, and following it in
the C++ — in that order, so the diff shows which answers moved.

Adding a source file to `cpp/src/` means naming it in **two** places: `add_library` in
`cpp/CMakeLists.txt`, and `INTERFACE_SOURCES` in `cpp/cmake/sstsr_cppWheelTargets.cmake`.
`hatch_build.py` needs no change — it maps `cpp/include` and `cpp/src` as whole directories — and
that asymmetry is the trap: the wheel *ships* a file it does not *compile*, so forgetting the
second place produces an undefined symbol in a consumer and nowhere else.
`cpp/examples/consumer/main.cpp` is what catches it, because `wheel_consumer` compiles the same
file through the wheel's package.
