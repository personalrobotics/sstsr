// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
#include "sstsr/tsr_chain.hpp"

#include "chain_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace sstsr {

namespace {

// A number formatted as the Python's "%.6g", for the append rejection message. The two
// implementations should complain about the same frame in recognisably the same words.
std::string g6(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  return std::string(buf);
}

// The flat 6n coordinate vector back to one XyzRpy per link. Only the chart bookkeeping
// needs the flat form; every public entry point is per-link.
ChainCoords unflatten(const std::vector<double>& x) {
  ChainCoords out(x.size() / 6);
  for (std::size_t i = 0; i < out.size(); ++i) {
    for (std::size_t j = 0; j < 6; ++j) out[i][j] = x[i * 6 + j];
  }
  return out;
}

bool all_finite(const ChainCoords& c) {
  for (const XyzRpy& row : c) {
    for (double v : row) {
      if (!std::isfinite(v)) return false;
    }
  }
  return true;
}

}  // namespace

const char* to_string(ChainStatus s) { return s == ChainStatus::kSatisfied ? "satisfied" : "not_found"; }

TSRChain::TSRChain(const std::vector<TSR>& tsrs) {
  for (const TSR& t : tsrs) append(t);
}

void TSRChain::append(const TSR& tsr) {
  if (!tsrs_.empty()) {
    const Transform& T = tsr.T0_w();
    const Transform eye = Transform::identity();
    bool identity = true;
    for (std::size_t k = 0; k < 16; ++k) {
      if (std::fabs(T.m[k] - eye.m[k]) > kFrameAtol) {  // numpy.allclose with rtol=0: <= atol passes
        identity = false;
        break;
      }
    }
    if (!identity) {
      const double trace = T.at(0, 0) + T.at(1, 1) + T.at(2, 2);
      throw std::invalid_argument(
          "a TSR after the first in a chain is positioned by the chain: its frame is the previous link's end "
          "frame, so its T0_w is never read and must be the identity (got a translation of [" +
          g6(T.at(0, 3)) + ", " + g6(T.at(1, 3)) + ", " + g6(T.at(2, 3)) + "] m and rotation trace " + g6(trace) +
          "). Put the offset in the PREVIOUS link's Tw_e, which is what moves the end frame this link hangs off.");
    }
  }
  tsrs_.push_back(tsr);
}

std::vector<std::array<bool, 6>> TSRChain::is_valid(const ChainCoords& c) const {
  if (tsrs_.empty()) throw std::invalid_argument("cannot validate against an empty TSR chain");
  if (c.size() != tsrs_.size()) {
    throw std::invalid_argument("coordinates must be of equal length to the TSR chain (got " +
                                std::to_string(c.size()) + " for " + std::to_string(tsrs_.size()) + " links)");
  }
  std::vector<std::array<bool, 6>> out;
  out.reserve(tsrs_.size());
  for (std::size_t i = 0; i < tsrs_.size(); ++i) out.push_back(tsrs_[i].is_valid(c[i]));
  return out;
}

ChainCoords TSRChain::continuous_coordinates(const ChainCoords& c) const {
  if (tsrs_.empty()) throw std::invalid_argument("an empty TSR chain has no chart to map coordinates into");
  if (c.size() != tsrs_.size()) {
    throw std::invalid_argument("coordinates must be of equal length to the TSR chain (got " +
                                std::to_string(c.size()) + " for " + std::to_string(tsrs_.size()) + " links)");
  }
  ChainCoords out(tsrs_.size());
  for (std::size_t i = 0; i < tsrs_.size(); ++i) {
    const Bounds6& Bw = tsrs_[i].continuous_bounds();
    XyzRpy row = c[i];
    // Wrap the three rotations into their intervals FIRST, then clip all six. Translations
    // are not periodic and are only clipped. Clipping without wrapping would send a valid
    // wrapping coordinate to an unrelated boundary (#87, #90).
    for (int k = 0; k < 3; ++k) {
      row[static_cast<std::size_t>(3 + k)] = wrap_to_interval(row[static_cast<std::size_t>(3 + k)], Bw.lo(3 + k));
    }
    for (int j = 0; j < 6; ++j) {
      row[static_cast<std::size_t>(j)] = std::clamp(row[static_cast<std::size_t>(j)], Bw.lo(j), Bw.hi(j));
    }
    out[i] = row;
  }
  return out;
}

Transform TSRChain::to_transform(const ChainCoords& c) const {
  if (tsrs_.empty()) throw std::invalid_argument("cannot compute a transform for an empty TSR chain");
  const ChainCoords cont = continuous_coordinates(c);  // also checks the length

  // Deliberately NOT TSR::to_transform, which validates and throws on an out-of-bounds
  // coordinate: the chain composes the raw xyzrpy_to_trans with each link's Tw_e, exactly as
  // the Python does, because the solver evaluates points at and just outside bounds.
  //
  // Only the FIRST link's T0_w participates. append forces identity on the rest, and this
  // loop never reads another one.
  Transform current = tsrs_[0].T0_w();
  for (std::size_t i = 0; i < tsrs_.size(); ++i) {
    current = current * (xyzrpy_to_trans(cont[i]) * tsrs_[i].Tw_e());
  }
  return current;
}

ChainCoords TSRChain::sample_xyzrpy(Rng& rng) const {
  ChainCoords out(tsrs_.size());
  for (std::size_t i = 0; i < tsrs_.size(); ++i) out[i] = tsrs_[i].sample_xyzrpy(rng);
  return out;
}

Transform TSRChain::sample(Rng& rng) const { return to_transform(sample_xyzrpy(rng)); }

ChainSample TSRChain::sample_with_witness(Rng& rng) const {
  ChainCoords coords = sample_xyzrpy(rng);
  Transform pose = to_transform(coords);
  return ChainSample{pose, std::move(coords)};
}

std::optional<double> TSRChain::witness_residual(const Transform& T, const ChainCoords& c) const {
  // The size check comes before composing, so a wrong coordinate count is "not a witness"
  // rather than an exception -- to_transform would throw on the same input.
  if (tsrs_.empty() || c.size() != tsrs_.size()) return std::nullopt;
  for (std::size_t i = 0; i < tsrs_.size(); ++i) {
    const std::array<bool, 6> ok = tsrs_[i].is_valid(c[i]);
    for (bool b : ok) {
      if (!b) return std::nullopt;
    }
  }
  return geodesic_distance(to_transform(c), T);
}

bool TSRChain::validate_witness(const Transform& T, const ChainCoords& c, double tolerance) const {
  const std::optional<double> residual = witness_residual(T, c);
  return residual.has_value() && *residual < tolerance;
}

ChainSolveResult TSRChain::solve(const Transform& T, const SolveOptions& opt) const {
  // Validate before anything else, including before the empty-chain return: an out-of-range
  // budget is a caller mistake whatever the chain looks like. This is the Python's order,
  // which validates before it will even import SciPy.
  if (opt.max_starts < 0) {
    throw std::invalid_argument("max_starts must be >= 0, got " + std::to_string(opt.max_starts));
  }
  if (opt.max_nfev < 1) {
    throw std::invalid_argument("max_nfev must be >= 1, got " + std::to_string(opt.max_nfev));
  }
  if (!std::isfinite(opt.tolerance)) {
    throw std::invalid_argument("tolerance must be a finite positive number");
  }
  if (opt.tolerance <= 0.0) {
    throw std::invalid_argument("tolerance must be > 0, got " + std::to_string(opt.tolerance));
  }

  const std::size_t n = tsrs_.size();
  if (n == 0) return ChainSolveResult{};  // kNotFound, no coordinates, +inf, 0, 0

  // A guess is eligible only if it is the right length and finite. Anything else is ignored,
  // not an error -- a planner passing a neighbour's coordinates should not have to know
  // whether that neighbour was itself solved.
  const ChainCoords* guess = nullptr;
  if (opt.initial_guess.has_value() && opt.initial_guess->size() == n && all_finite(*opt.initial_guess)) {
    guess = &*opt.initial_guess;
  }

  // Warm path: a supplied witness that already validates. One forward composition, no
  // optimiser. This precedes the single-link branch, so for n == 1 a valid guess wins and the
  // closed form is never reached. The coordinates come back VERBATIM, un-canonicalised.
  if (guess != nullptr) {
    const std::optional<double> residual = witness_residual(T, *guess);
    if (residual.has_value() && *residual < opt.tolerance) {
      return ChainSolveResult{ChainStatus::kSatisfied, *guess, *residual, 0, 0};
    }
  }

  // Exact path: a single link reduces to the closed-form region check.
  if (n == 1) {
    if (tsrs_[0].contains(T)) {
      ChainCoords coords{tsrs_[0].to_xyzrpy(T)};
      // No tolerance re-check, matching the Python. contains absorbs only round-off (#190), so
      // the residual is round-off too -- below any tolerance coarser than kAngleAtol.
      //
      // The residual is computed into a named value before the move, because the move leaves
      // `coords` empty and a braced initialiser evaluates left to right.
      const double residual = geodesic_distance(to_transform(coords), T);
      return ChainSolveResult{ChainStatus::kSatisfied, std::move(coords), residual, 0, 0};
    }
    const auto [dist, bw] = tsrs_[0].distance_bwopt(T);
    // `dist` is the region's 6-vector displacement norm, NOT a geodesic. The two coincide for
    // a pure translation or a single-axis rotation and diverge otherwise.
    return ChainSolveResult{ChainStatus::kNotFound, ChainCoords{bw}, dist, 0, 0};
  }

  // The chart over all 6n coordinates. continuous_bounds guarantees lower <= upper even for a
  // wrapping rotational interval. `free` is computed on the CONTINUOUS bounds, never on Bw: a
  // Bw row of [pi, pi] is fixed once mapped, [3pi/4, -3pi/4] is free with width pi/2.
  std::vector<double> lower, upper, x_full;
  lower.reserve(n * 6);
  upper.reserve(n * 6);
  x_full.reserve(n * 6);
  std::size_t free_count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Bounds6& b = tsrs_[i].continuous_bounds();
    for (int j = 0; j < 6; ++j) {
      lower.push_back(b.lo(j));
      upper.push_back(b.hi(j));
      // The midpoint is NOT re-wrapped. For a chart row like [3pi/4, 3pi/4 + pi/2] it is
      // exactly pi, which is outside [-pi, pi); normalising it would push it out of the chart,
      // where to_transform clips it to a boundary instead of the middle.
      x_full.push_back((b.lo(j) + b.hi(j)) / 2.0);
      if (b.hi(j) > b.lo(j)) ++free_count;
    }
  }
  ChainCoords midpoint = unflatten(x_full);

  // Two paths answer from the midpoint alone and spend nothing from the budget: an all-fixed
  // chain, which has exactly one candidate pose, and max_starts == 0, which is what the
  // Python's start schedule leaves in its best-point slot when it truncates to empty.
  if (free_count == 0 || opt.max_starts == 0) {
    const double res = geodesic_distance(to_transform(midpoint), T);
    const ChainStatus status = res < opt.tolerance ? ChainStatus::kSatisfied : ChainStatus::kNotFound;
    return ChainSolveResult{status, std::move(midpoint), res, 0, 0};
  }

  // --- The bounded multi-start cold inverse ----------------------------------------------
  //
  // Everything above answered without searching. From here the result is the best point an
  // optimiser found, which is why it is specified by properties rather than recorded in the
  // corpus: a different optimiser evaluates a different set of points, and nothing relates the
  // two beyond "both are >= the true minimum" (issue #85, docs/CPP.md).

  std::vector<std::size_t> free_index;  // flat coordinate indices the optimiser may move
  free_index.reserve(free_count);
  for (std::size_t k = 0; k < n * 6; ++k) {
    if (upper[k] > lower[k]) free_index.push_back(k);
  }
  const std::size_t m = free_index.size();
  std::vector<double> lo_f(m), hi_f(m);
  for (std::size_t a = 0; a < m; ++a) {
    lo_f[a] = lower[free_index[a]];
    hi_f[a] = upper[free_index[a]];
  }

  // Point-bound coordinates are held at their single value and never handed to the optimiser;
  // a zero-width interval has no descent direction and would only add a rank-deficient column.
  std::vector<double> x_base = x_full;

  int nfev = 0;
  std::vector<double> best_x(m);
  double best_sq = std::numeric_limits<double>::infinity();
  for (std::size_t a = 0; a < m; ++a) best_x[a] = x_full[free_index[a]];

  const auto coords_of = [&](const std::vector<double>& xf) {
    std::vector<double> flat = x_base;
    for (std::size_t a = 0; a < m; ++a) flat[free_index[a]] = xf[a];
    // Through the same chart map the forward path uses, so the objective the optimiser
    // descends is the one to_transform will evaluate at the end.
    return continuous_coordinates(unflatten(flat));
  };

  // One residual evaluation is one forward composition of the chain -- the same unit the
  // Python's counted objective uses. The budget is checked BEFORE the increment, so
  // nfev <= max_nfev holds strictly. The Jacobian rides along on the same sweep and is not
  // counted separately, because it costs no extra composition.
  const auto evaluate = [&](const std::vector<double>& xf, bool jacobian,
                            detail::ChainResidual& out) -> bool {
    if (nfev >= opt.max_nfev) return false;
    ++nfev;
    out = detail::chain_residual(tsrs_, coords_of(xf), T, jacobian);
    double sq = 0.0;
    for (double v : out.r) sq += v * v;
    if (sq < best_sq) {
      best_sq = sq;
      best_x = xf;
    }
    return true;
  };

  const auto geodesic_at = [&](const std::vector<double>& xf) {
    return geodesic_distance(to_transform(coords_of(xf)), T);
  };

  // The deterministic start schedule, in priority order and truncated to max_starts: a guess
  // that did not validate refines first (canonicalised through the same chart map, so a valid
  // wrapping coordinate is preserved rather than clipped to a boundary), then the midpoint, the
  // two opposite corners, then interior points.
  //
  // The interior points come from this implementation's own generator, so they are NOT the
  // Python's. That is the documented divergence of sstsr/rng.hpp applied here: porting numpy's
  // PCG64 would buy nothing, because a different optimiser takes a different trajectory from
  // identical starts anyway. The first starts, which are arithmetic on the chart, do agree.
  std::vector<std::vector<double>> schedule;
  if (guess != nullptr) {
    const ChainCoords canonical = continuous_coordinates(*guess);
    std::vector<double> start(m);
    for (std::size_t a = 0; a < m; ++a) {
      start[a] = canonical[free_index[a] / 6][free_index[a] % 6];
    }
    schedule.push_back(std::move(start));
  }
  {
    std::vector<double> mid(m);
    for (std::size_t a = 0; a < m; ++a) mid[a] = x_full[free_index[a]];
    schedule.push_back(std::move(mid));
  }
  schedule.push_back(lo_f);
  schedule.push_back(hi_f);
  Rng rng(0);
  while (schedule.size() < static_cast<std::size_t>(opt.max_starts)) {
    std::vector<double> point(m);
    for (std::size_t a = 0; a < m; ++a) point[a] = lo_f[a] + (hi_f[a] - lo_f[a]) * unit(rng);
    schedule.push_back(std::move(point));
  }
  if (schedule.size() > static_cast<std::size_t>(opt.max_starts)) {
    schedule.resize(static_cast<std::size_t>(opt.max_starts));
  }

  constexpr int kMaxIterations = 100;  // a hard cap, so the worst case is bounded and repeatable
  constexpr double kLambda0 = 1e-3, kLambdaMin = 1e-12, kLambdaMax = 1e14;

  int starts = 0;
  std::vector<double> M, rhs;
  for (const std::vector<double>& x0 : schedule) {
    if (nfev >= opt.max_nfev) break;  // before the increment, so a start that cannot afford a
    ++starts;                         // single evaluation is not counted as having run

    std::vector<double> x(m);
    for (std::size_t a = 0; a < m; ++a) x[a] = std::clamp(x0[a], lo_f[a], hi_f[a]);

    detail::ChainResidual cur;
    if (!evaluate(x, true, cur)) break;
    double f0 = 0.0;
    for (double v : cur.r) f0 += v * v;
    double lambda = kLambda0;

    for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
      if (f0 <= 0.0) break;

      // g = J^T r and H = J^T J over the free columns.
      const std::size_t cols = 6 * n;
      std::vector<double> g(m, 0.0);
      M.assign(m * m, 0.0);
      for (std::size_t a = 0; a < m; ++a) {
        const std::size_t ca = free_index[a];
        for (std::size_t row = 0; row < 12; ++row) g[a] += cur.J[row * cols + ca] * cur.r[row];
        for (std::size_t b = 0; b <= a; ++b) {
          const std::size_t cb = free_index[b];
          double s = 0.0;
          for (std::size_t row = 0; row < 12; ++row) s += cur.J[row * cols + ca] * cur.J[row * cols + cb];
          M[a * m + b] = s;
          M[b * m + a] = s;
        }
      }

      // Gradient projection: a coordinate sitting on a bound whose gradient pushes it further
      // out cannot move, so hold it fixed for this step instead of letting a rank-deficient or
      // outward direction stall the whole solve.
      std::vector<char> active(m, 0);
      std::size_t free_now = 0;
      for (std::size_t a = 0; a < m; ++a) {
        const bool at_lo = x[a] <= lo_f[a] && g[a] > 0.0;
        const bool at_hi = x[a] >= hi_f[a] && g[a] < 0.0;
        active[a] = (at_lo || at_hi) ? 1 : 0;
        if (!active[a]) ++free_now;
      }
      if (free_now == 0) break;  // every direction points out of the box: this start is done

      std::vector<double> diagonal(m);
      for (std::size_t a = 0; a < m; ++a) diagonal[a] = std::max(M[a * m + a], 1e-10);

      bool stepped = false;
      double step_inf = 0.0;
      for (int trial = 0; trial < 20 && lambda <= kLambdaMax; ++trial) {
        std::vector<double> system = M;
        rhs.assign(m, 0.0);
        for (std::size_t a = 0; a < m; ++a) {
          if (active[a]) {
            for (std::size_t b = 0; b < m; ++b) {
              system[a * m + b] = 0.0;
              system[b * m + a] = 0.0;
            }
            system[a * m + a] = 1.0;
            rhs[a] = 0.0;
          } else {
            system[a * m + a] += lambda * diagonal[a];
            rhs[a] = -g[a];
          }
        }
        if (!detail::cholesky_solve(system, rhs, m)) {
          lambda *= 10.0;  // not positive definite: a larger lambda restores dominance
          continue;
        }

        // The projected step. Note where the *guarantee* about the returned coordinates comes
        // from: not this clamp, but coords_of, which runs every point through the chart map. So
        // dropping the clamp would not let an out-of-chart coordinate escape -- it would only
        // desynchronise x from the point actually evaluated, costing convergence rather than
        // correctness. The clamp is here to keep those two the same point.
        std::vector<double> candidate(m);
        step_inf = 0.0;
        for (std::size_t a = 0; a < m; ++a) {
          candidate[a] = std::clamp(x[a] + rhs[a], lo_f[a], hi_f[a]);
          step_inf = std::max(step_inf, std::fabs(candidate[a] - x[a]));
        }

        detail::ChainResidual trial_residual;
        if (!evaluate(candidate, true, trial_residual)) {
          stepped = false;
          break;
        }
        double f1 = 0.0;
        for (double v : trial_residual.r) f1 += v * v;
        if (f1 < f0) {
          x = std::move(candidate);
          cur = std::move(trial_residual);
          f0 = f1;
          lambda = std::max(lambda * 0.3, kLambdaMin);
          stepped = true;
          break;
        }
        lambda *= 10.0;
      }

      if (!stepped) break;
      if (step_inf <= 1e-14) break;
      if (nfev >= opt.max_nfev) break;
    }

    if (geodesic_at(best_x) < opt.tolerance) break;
  }

  const double residual = geodesic_at(best_x);
  const ChainStatus status = residual < opt.tolerance ? ChainStatus::kSatisfied : ChainStatus::kNotFound;
  return ChainSolveResult{status, coords_of(best_x), residual, nfev, starts};
}

std::pair<double, ChainCoords> TSRChain::distance(const Transform& T) const {
  const ChainSolveResult result = solve(T);
  return {result.residual, result.coordinates};
}

std::pair<double, Transform> TSRChain::closest_transform(const Transform& T) const {
  const auto [dist, coords] = distance(T);
  // On an empty chain this throws out of to_transform, as the Python does: there is no pose to
  // report, and an identity transform would be a confident wrong answer.
  return {dist, to_transform(coords)};
}

bool TSRChain::contains(const Transform& T, const std::optional<ChainCoords>& guess) const {
  if (tsrs_.empty()) return false;
  if (tsrs_.size() == 1) return tsrs_[0].contains(T);
  SolveOptions opt{};
  opt.initial_guess = guess;
  return solve(T, opt).status == ChainStatus::kSatisfied;
}

ChainCoords TSRChain::to_xyzrpy(const Transform& T) const { return distance(T).second; }

}  // namespace sstsr
