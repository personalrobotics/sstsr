// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
#pragma once

#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "sstsr/rng.hpp"
#include "sstsr/transform.hpp"
#include "sstsr/tsr.hpp"

namespace sstsr {

// One XyzRpy per link, in link order: the Python's (n, 6) coordinate array. Per-link rather
// than a flat vector<double> so a length mismatch is a size check instead of index
// arithmetic; the solver flattens internally.
using ChainCoords = std::vector<XyzRpy>;

// A sampled pose together with the coordinates that constructed it. Composing those
// coordinates forward reproduces the pose exactly, so they are a *constructive witness*:
// membership is checkable by validate_witness with no optimiser at all.
struct ChainSample {
  Transform pose{};
  ChainCoords coordinates{};
};

// The Python's status string as an enum: a planner gets -Wswitch coverage and no string
// comparison in its inner loop. to_string exists so the conformance corpus can be compared
// against the Python's recorded "satisfied"/"not_found" verbatim.
enum class ChainStatus { kSatisfied, kNotFound };
const char* to_string(ChainStatus s);

// The result of an inverse chain solve. `status` is kSatisfied when a witness within
// tolerance was found. kNotFound is **not** a certificate of non-membership -- a finite set
// of local solves cannot prove global infeasibility for a rotation-rich chain (issue #85).
// `residual` is an upper bound on the true minimum, not a certified distance. `starts` is
// the number of optimiser starts actually run (0 on an exact path) and `nfev` the total
// residual evaluations across them; starts <= max_starts and nfev <= max_nfev strictly.
struct ChainSolveResult {
  ChainStatus status{ChainStatus::kNotFound};
  ChainCoords coordinates{};
  double residual{std::numeric_limits<double>::infinity()};
  int nfev{0};
  int starts{0};
};

// The Python's TSRChain._DEFAULT_MAX_STARTS / _DEFAULT_MAX_NFEV. Restated here of necessity,
// and drift-checked against the Python in tests/tsr/test_cpp_package.py -- a budget that
// drifts does not fail the corpus, it fails much later as two implementations disagreeing
// about how hard they tried.
constexpr int kDefaultMaxStarts = 11;
constexpr int kDefaultMaxNfev = 2200;

// Arguments to solve, as a struct so a call site reads
// `chain.solve(T, {.initial_guess = neighbour})`. The Python validates these before
// importing SciPy; we validate them before touching the chain, and for the same reason: an
// out-of-range budget is a caller mistake and should say so immediately.
//
// Half of the Python's _require_int (rejecting bool, rejecting non-integral floats) is a
// Python typing problem the C++ type system already answers. What remains is range
// validation, which throws std::invalid_argument.
struct SolveOptions {
  std::optional<ChainCoords> initial_guess{};
  int max_starts = kDefaultMaxStarts;  // >= 0
  int max_nfev = kDefaultMaxNfev;      // >= 1
  double tolerance = kChainTolerance;         // finite, > 0
};

// A sequence of composed TSRs. The same chain as src/tsr/core/tsr_chain.py's TSRChain, rule
// for rule, held to it by the conformance corpus (docs/CPP.md).
//
// **Composition rule** (Berenson et al. 2011 Sec. 5.1). The first link is placed by its own
// T0_w. Every later link is placed *by the chain*: its frame **is** the previous link's end
// frame. So a later link's own T0_w is never read, and append refuses a non-identity one
// rather than ignoring it (issue #166) -- a dropped offset is a pose quietly wrong by
// exactly that offset, which is worse to debug than a rejected argument. To sit a later
// link's region a fixed distance from the previous end frame, put the offset in the
// PREVIOUS link's Tw_e.
//
// Not here, and deliberately, mirroring what TSR already omits: serialization, masked (NaN)
// sampling, the TSR=/TSRs=/tsr= keyword alias triple, and a rotation weight other than 1.
//
// The inverse comes in two kinds, and they carry different guarantees. The exact paths -- an
// empty chain, a validating witness, a single link, an all-fixed chain, max_starts == 0 -- are
// specified to the last bit and checked against the Python's recorded answers. The cold path
// reports the best point a bounded search found, so it is specified by properties instead: a
// different optimiser evaluates a different set of points, and nothing relates the two beyond
// "both are >= the true minimum". See docs/CPP.md.
class TSRChain {
 public:
  TSRChain() = default;
  explicit TSRChain(const std::vector<TSR>& tsrs);  // appends each, so the T0_w rule applies here too

  // Throws std::invalid_argument if a link after the first carries a T0_w the chain will not
  // read. See the composition rule above.
  void append(const TSR& tsr);

  const std::vector<TSR>& tsrs() const { return tsrs_; }
  std::size_t size() const { return tsrs_.size(); }

  // Throw std::invalid_argument on an empty chain or a coordinate count that is not size().
  std::vector<std::array<bool, 6>> is_valid(const ChainCoords& c) const;
  Transform to_transform(const ChainCoords& c) const;

  // The Python's _to_continuous, public here because it is the chart map: a planner holding a
  // neighbouring state's coordinates wants to canonicalise them before passing them as a
  // guess. Rotations are periodic, so each is wrapped into its link's continuous interval
  // *before* all six are clipped -- a valid wrapping coordinate (-3.0 for an interval
  // [3pi/4, -3pi/4]) becomes its in-chart equivalent rather than being clipped to an
  // unrelated boundary (issues #87, #90). The forward path and the solver's start
  // construction both route through here so they cannot drift apart.
  ChainCoords continuous_coordinates(const ChainCoords& c) const;

  ChainCoords sample_xyzrpy(Rng& rng) const;
  Transform sample(Rng& rng) const;
  ChainSample sample_with_witness(Rng& rng) const;

  // An exact positive certificate: one bounds check and one forward composition, no
  // optimiser. An empty chain has no witness, and a wrong coordinate count is not a witness,
  // so both return false rather than throwing (issue #87).
  bool validate_witness(const Transform& T, const ChainCoords& c, double tolerance = kChainTolerance) const;

  ChainSolveResult solve(const Transform& T, const SolveOptions& opt = {}) const;

  // All four delegate to solve and inherit its best-found (upper bound) status for a chain
  // with two or more links (issue #85). closest_transform throws on an empty chain, because
  // there is no pose to compose; to_xyzrpy returns empty. That asymmetry is the Python's.
  std::pair<double, ChainCoords> distance(const Transform& T) const;
  std::pair<double, Transform> closest_transform(const Transform& T) const;
  bool contains(const Transform& T, const std::optional<ChainCoords>& guess = std::nullopt) const;
  ChainCoords to_xyzrpy(const Transform& T) const;

 private:
  // The geodesic residual for coordinates that are a valid witness, else nullopt.
  std::optional<double> witness_residual(const Transform& T, const ChainCoords& c) const;

  std::vector<TSR> tsrs_;
};

}  // namespace sstsr
