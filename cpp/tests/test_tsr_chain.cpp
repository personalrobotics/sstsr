// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
//
// Hand-written cases for the chain. The systematic agreement with the Python is the
// conformance corpus (test_conformance.cpp); these cover the rejections, the asymmetries, and
// the properties a corpus of recorded answers cannot express -- sampling (which draws from a
// different engine by design, see sstsr/rng.hpp) and the construction rule.
//
// The cold inverse is held to properties here rather than to recorded numbers, because it
// reports the best point an optimiser found -- see the section below.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "chain_solver.hpp"
#include "harness.hpp"
#include "sstsr/transform.hpp"
#include "sstsr/tsr.hpp"
#include "sstsr/tsr_chain.hpp"

using namespace sstsr;
constexpr double kPi = std::numbers::pi;

namespace {

Transform frame(double x, double y, double z, double roll = 0, double pitch = 0, double yaw = 0) {
  return xyzrpy_to_trans({x, y, z, roll, pitch, yaw});
}
Bounds6 bounds(std::array<std::array<double, 2>, 6> rows) { return Bounds6{rows}; }

// The motivating chain from issue #165: a hinge with free yaw, then a handle a fixed distance
// along the door.
TSRChain door() {
  return TSRChain({
      TSR(frame(1.0, 0.0, 0.8), frame(0.6, 0.0, 0.0),
          bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {-kPi / 2, kPi / 2}}})),
      TSR(Transform::identity(), Transform::identity(),
          bounds({{{0, 0}, {0, 0}, {-0.05, 0.05}, {0, 0}, {-0.2, 0.2}, {0, 0}}})),
  });
}

// An outer rotational interval on the first link: the chart is [3pi/4, 3pi/4 + pi/2].
TSRChain wrapping() {
  return TSRChain({
      TSR(Transform::identity(), frame(0.3, 0, 0),
          bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {3 * kPi / 4, -3 * kPi / 4}}})),
      TSR(Transform::identity(), Transform::identity(),
          bounds({{{0, 0}, {0, 0}, {-0.1, 0.1}, {2.5, -2.5}, {0, 0}, {0, 0}}})),
  });
}

// An independent forward composition, written out rather than reusing to_transform, so the
// chart test has an oracle that is not the thing under test. Mirrors the Python's own
// independent-composition check in tests/tsr/test_tsr_chain.py.
Transform compose_independently(const TSRChain& chain, const ChainCoords& canonical) {
  Transform acc = chain.tsrs()[0].T0_w();
  for (std::size_t i = 0; i < chain.size(); ++i) {
    const Transform Tw = xyzrpy_to_trans(canonical[i]);
    Transform step = Tw * chain.tsrs()[i].Tw_e();
    acc = acc * step;
  }
  return acc;
}

// The four chains the cold inverse is measured on, spanning what makes the inverse hard: a
// single free hinge, two free yaws with a slide, three links, and a full SO(3) ball whose RPY
// runs through gimbal lock.
std::vector<TSRChain> cold_fixtures() {
  const Bounds6 free_yaw = bounds({{{-0.1, 0.1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {-kPi, kPi}}});
  const TSR a(frame(0.3, 0.1, 0.5), frame(0.2, 0, 0), free_yaw);
  const TSR b(Transform::identity(), frame(0.15, 0, 0),
              bounds({{{0, 0}, {0, 0}, {0, 0}, {-0.3, 0.3}, {0, 0}, {-kPi, kPi}}}));
  const TSR c(Transform::identity(), Transform::identity(),
              bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {-0.4, 0.4}, {0, 0}}}));
  const TSR ball(Transform::identity(), frame(0, 0, 0.1),
                 bounds({{{-0.02, 0.02}, {-0.02, 0.02}, {0, 0}, {-kPi, kPi}, {-kPi, kPi}, {-kPi, kPi}}}));
  return {door(), TSRChain({a, b}), TSRChain({a, b, c}), TSRChain({a, ball})};
}

double max_abs_diff(const Transform& a, const Transform& b) {
  double worst = 0.0;
  for (std::size_t k = 0; k < 16; ++k) worst = std::max(worst, std::fabs(a.m[k] - b.m[k]));
  return worst;
}

}  // namespace

TEST(a_later_link_carrying_a_frame_the_chain_would_ignore_is_refused) {
  // Issue #166: the chain positions every link after the first by the previous link's end
  // frame, so a later T0_w is never read. Dropping it silently would make the pose wrong by
  // exactly that offset, which is worse to debug than a rejected argument.
  const Bounds6 b = bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {-0.5, 0.5}}});
  const TSR first(frame(1.0, 2.0, 0.5), Transform::identity(), b);
  const TSR offset(frame(0.1, 0, 0), Transform::identity(), b);
  const TSR identity_framed(Transform::identity(), Transform::identity(), b);

  TSRChain chain;
  chain.append(first);  // the FIRST link may sit anywhere
  CHECK_THROWS(chain.append(offset), std::invalid_argument);
  chain.append(identity_framed);  // and an identity-framed one is fine
  CHECK(chain.size() == 2);

  // The vector constructor goes through append, so it refuses the same thing.
  CHECK_THROWS(TSRChain(std::vector<TSR>{first, offset}), std::invalid_argument);

  // A rotation-only later frame is refused too, not just a translation.
  const TSR rotated(frame(0, 0, 0, 0, 0, 0.3), Transform::identity(), b);
  TSRChain other;
  other.append(first);
  CHECK_THROWS(other.append(rotated), std::invalid_argument);
}

TEST(the_remedy_the_rejection_prescribes_actually_works) {
  // The message tells the caller to put the offset in the PREVIOUS link's Tw_e. That has to be
  // true, or the error is a dead end.
  const Bounds6 hinge = bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {-0.4, 0.4}}});
  const Bounds6 fixed = bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}});
  const TSRChain chain({
      TSR(frame(1.0, 0, 0), frame(0.25, 0, 0), hinge),  // the offset lives here
      TSR(Transform::identity(), Transform::identity(), fixed),
  });
  const ChainCoords at_zero{XyzRpy{0, 0, 0, 0, 0, 0}, XyzRpy{0, 0, 0, 0, 0, 0}};
  const Transform pose = chain.to_transform(at_zero);
  CHECK_NEAR(pose.at(0, 3), 1.25, 1e-12);  // 1.0 from T0_w, 0.25 from the previous Tw_e
}

TEST(an_empty_chain_is_asymmetric_in_exactly_the_pythons_way) {
  const TSRChain empty;
  CHECK(empty.size() == 0);
  CHECK(!empty.contains(Transform::identity()));
  CHECK(!empty.validate_witness(Transform::identity(), {}));
  CHECK(empty.to_xyzrpy(Transform::identity()).empty());
  CHECK(std::isinf(empty.distance(Transform::identity()).first));
  // These throw: there is nothing to compose. Returning identity would be a confident wrong
  // answer, which is the whole objection of #166.
  CHECK_THROWS(empty.to_transform({}), std::invalid_argument);
  CHECK_THROWS(empty.is_valid({}), std::invalid_argument);
  CHECK_THROWS(empty.continuous_coordinates({}), std::invalid_argument);
  CHECK_THROWS(empty.closest_transform(Transform::identity()), std::invalid_argument);
}

TEST(a_wrong_coordinate_count_throws_from_the_chart_but_is_merely_not_a_witness) {
  const TSRChain chain = door();
  const ChainCoords too_few{XyzRpy{0, 0, 0, 0, 0, 0}};
  CHECK_THROWS(chain.to_transform(too_few), std::invalid_argument);
  CHECK_THROWS(chain.is_valid(too_few), std::invalid_argument);
  CHECK_THROWS(chain.continuous_coordinates(too_few), std::invalid_argument);
  // validate_witness answers the question "are these a witness?", and the answer for the wrong
  // number of coordinates is no, not an exception.
  CHECK(!chain.validate_witness(Transform::identity(), too_few));
  // And solve ignores an ineligible guess rather than failing on it.
  Rng rng(1);
  const ChainSample s = chain.sample_with_witness(rng);
  SolveOptions opt{};
  opt.initial_guess = too_few;
  opt.max_starts = 0;  // stay on an exact path, so this checks the guess handling alone
  const ChainSolveResult result = chain.solve(s.pose, opt);
  CHECK(result.starts == 0 && result.nfev == 0);
}

TEST(the_chart_wraps_rotations_before_it_clips_them) {
  // Issue #87/#90. A yaw of -3.0 belongs to the interval [3pi/4, -3pi/4] -- it is 2pi away from
  // the in-chart equivalent. Clipping without wrapping first would send it to a boundary of an
  // unrelated interval, and the composed pose would be wrong rather than merely rounded.
  const TSRChain chain = wrapping();
  const Bounds6& chart = chain.tsrs()[0].continuous_bounds();
  CHECK_NEAR(chart.lo(5), 3 * kPi / 4, 1e-12);
  CHECK_NEAR(chart.hi(5), 3 * kPi / 4 + kPi / 2, 1e-12);

  ChainCoords c{XyzRpy{0, 0, 0, 0, 0, -3.0}, XyzRpy{0, 0, 0, 0, 0, 0}};
  const ChainCoords canonical = chain.continuous_coordinates(c);
  CHECK_NEAR(canonical[0][5], -3.0 + 2 * kPi, 1e-12);  // wrapped into the chart, not clipped
  CHECK(canonical[0][5] >= chart.lo(5) && canonical[0][5] <= chart.hi(5));

  // A full turn must not move the composed pose.
  ChainCoords turned = c;
  turned[0][5] += 2 * kPi;
  CHECK(max_abs_diff(chain.to_transform(c), chain.to_transform(turned)) < 1e-12);

  // And the composition agrees with an independent product of the canonical coordinates.
  CHECK(max_abs_diff(chain.to_transform(c), compose_independently(chain, canonical)) < 1e-12);
}

TEST(the_midpoint_of_a_wrapping_chart_row_is_left_outside_minus_pi_to_pi) {
  // The chart for [3pi/4, -3pi/4] is [3pi/4, 3pi/4 + pi/2], whose midpoint is exactly pi. A
  // port that "normalises" it to -pi would push it out of the chart, where the forward map
  // clips it to a boundary instead of the middle -- changing what max_starts == 0 reports.
  const TSRChain chain = wrapping();
  const Bounds6& chart = chain.tsrs()[0].continuous_bounds();
  const double midpoint = (chart.lo(5) + chart.hi(5)) / 2.0;
  CHECK_NEAR(midpoint, kPi, 1e-12);
  CHECK(midpoint >= kPi);  // i.e. NOT inside [-pi, pi)

  SolveOptions opt{};
  opt.max_starts = 0;
  const ChainSolveResult result = chain.solve(Transform::identity(), opt);
  CHECK(result.starts == 0 && result.nfev == 0);
  CHECK_NEAR(result.coordinates[0][5], kPi, 1e-12);
}

TEST(sampling_stays_in_bounds_and_every_sample_carries_a_valid_witness) {
  // Sampling draws from a different engine than numpy by design (sstsr/rng.hpp), so it is
  // checked as a property rather than against recorded coordinates. The property that matters
  // is the one a planner relies on: a sampled pose is a member, provably, via its witness.
  Rng rng(20261001);
  for (const TSRChain& chain : {door(), wrapping()}) {
    for (int k = 0; k < 200; ++k) {
      const ChainSample s = chain.sample_with_witness(rng);
      CHECK(s.coordinates.size() == chain.size());
      // Checked through is_valid rather than against the raw chart interval: a sampled rotation
      // comes back wrapped into [-pi, pi), which for a wrapping interval sits outside the
      // chart's own [lo, hi] while still being a member of it.
      for (std::size_t i = 0; i < chain.size(); ++i) {
        for (bool ok : chain.tsrs()[i].is_valid(s.coordinates[i])) CHECK(ok);
      }
      // The witness is a constructive certificate: no optimiser needed.
      CHECK(chain.validate_witness(s.pose, s.coordinates));
    }
  }
}

TEST(a_validating_witness_takes_the_exact_warm_path_and_comes_back_verbatim) {
  Rng rng(7);
  const TSRChain chain = door();
  const ChainSample s = chain.sample_with_witness(rng);

  SolveOptions opt{};
  opt.initial_guess = s.coordinates;
  const ChainSolveResult result = chain.solve(s.pose, opt);

  CHECK(result.status == ChainStatus::kSatisfied);
  CHECK(result.nfev == 0 && result.starts == 0);  // no optimiser ran
  // Verbatim, not canonicalised: the Python returns the caller's coordinates unchanged.
  CHECK(result.coordinates.size() == s.coordinates.size());
  for (std::size_t i = 0; i < s.coordinates.size(); ++i) {
    CHECK(result.coordinates[i] == s.coordinates[i]);
  }
  // contains takes the same path when handed the witness.
  CHECK(chain.contains(s.pose, s.coordinates));
}

TEST(an_out_of_chart_guess_within_round_off_is_still_accepted_verbatim) {
  // is_valid absorbs round-off (kAngleAtol) on each side of every bound, while the chart clips.
  // So a guess a round-off past a zero-width bound validates and is returned as given --
  // outside the chart. The property "coordinates lie in the chart" therefore cannot apply to
  // this path, and this case is why. Until #190 the band was 1e-3 wide; now it is round-off.
  const TSRChain chain({
      TSR(frame(1.0, 0, 0.8), frame(0.6, 0, 0), bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}})),
      TSR(Transform::identity(), Transform::identity(),
          bounds({{{0, 0}, {0, 0}, {-0.05, 0.05}, {0, 0}, {0, 0}, {0, 0}}})),
  });
  Rng rng(3);
  const ChainSample s = chain.sample_with_witness(rng);
  ChainCoords guess = s.coordinates;
  guess[0][5] += 5e-7;  // inside kAngleAtol, outside the [0, 0] chart row

  SolveOptions opt{};
  opt.initial_guess = guess;
  const ChainSolveResult result = chain.solve(s.pose, opt);
  CHECK(result.status == ChainStatus::kSatisfied);
  CHECK(result.nfev == 0 && result.starts == 0);
  CHECK(result.coordinates[0][5] == guess[0][5]);
  CHECK(result.coordinates[0][5] > chain.tsrs()[0].continuous_bounds().hi(5));
}

TEST(an_all_fixed_chain_answers_from_one_pose_with_no_optimiser) {
  const Bounds6 fixed = bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}});
  const TSRChain chain({
      TSR(frame(0.2, 0.2, 0.2, 0.1, 0.2, 0.3), frame(0.1, 0, 0), fixed),
      TSR(Transform::identity(), frame(0, 0.05, 0), fixed),
  });
  const ChainCoords zeros{XyzRpy{0, 0, 0, 0, 0, 0}, XyzRpy{0, 0, 0, 0, 0, 0}};
  const Transform only = chain.to_transform(zeros);

  const ChainSolveResult hit = chain.solve(only);
  CHECK(hit.status == ChainStatus::kSatisfied);
  CHECK(hit.nfev == 0 && hit.starts == 0);

  const ChainSolveResult miss = chain.solve(frame(4, 4, 4));
  CHECK(miss.status == ChainStatus::kNotFound);
  CHECK(miss.nfev == 0 && miss.starts == 0);
  CHECK(miss.residual > 1.0);
}

TEST(a_single_link_chain_uses_the_closed_form_and_never_the_optimiser) {
  const TSRChain chain({TSR(frame(0.4, -0.2, 0.3, 0.1, 0.2, 0.3), frame(0.05, 0, 0),
                            bounds({{{-0.05, 0.05}, {-0.05, 0.05}, {0, 0}, {0, 0}, {0, 0}, {-kPi, kPi}}}))});
  Rng rng(5);
  const Transform inside = chain.sample(rng);
  const ChainSolveResult hit = chain.solve(inside);
  CHECK(hit.status == ChainStatus::kSatisfied);
  CHECK(hit.nfev == 0 && hit.starts == 0);
  CHECK(chain.contains(inside));

  const ChainSolveResult miss = chain.solve(frame(3.0, -2.0, 1.0, 0.4, 0.2, -0.8));
  CHECK(miss.status == ChainStatus::kNotFound);
  CHECK(miss.nfev == 0 && miss.starts == 0);
  CHECK(!chain.contains(frame(3.0, -2.0, 1.0, 0.4, 0.2, -0.8)));
}

TEST(a_single_link_refuses_a_pose_just_outside_its_bounds) {
  // Until #190 contains granted 1e-3 of slack per coordinate, so the closed-form path reported
  // kSatisfied here with a residual at or above the 1e-3 tolerance. Membership is now exact up
  // to round-off: a pose 2e-4 past a bound is not contained, and its distance is that 2e-4.
  const TSRChain chain({TSR(frame(0, 0, 0), Transform::identity(), bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}}))});
  const Transform just_outside = frame(2e-4, 0, 0);
  const ChainSolveResult result = chain.solve(just_outside);
  CHECK(result.status == ChainStatus::kNotFound);
  CHECK_NEAR(result.residual, 2e-4, 1e-12);
  CHECK(!chain.tsrs()[0].contains(just_outside));
}

TEST(a_chain_whose_only_freedom_is_an_outer_interval_is_not_mistaken_for_all_fixed) {
  // `free` must be read off the CONTINUOUS bounds, never off Bw. An outer rotational interval
  // has Bw.hi < Bw.lo, so a port that asks "Bw.hi > Bw.lo" concludes the row is fixed -- and a
  // chain whose only freedom is that row would then be answered from a single pose instead of
  // being searched. The two disagree nowhere else, which is why this needs its own chain.
  const Bounds6 outer_yaw_only =
      bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {3 * kPi / 4, -3 * kPi / 4}}});
  const Bounds6 all_fixed = bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}});
  const TSRChain chain({
      TSR(Transform::identity(), frame(0.3, 0, 0), outer_yaw_only),
      TSR(Transform::identity(), Transform::identity(), all_fixed),
  });

  // The chart says the yaw row spans a quarter turn, so it is free.
  const Bounds6& chart = chain.tsrs()[0].continuous_bounds();
  CHECK(chart.hi(5) > chart.lo(5));
  CHECK_NEAR(chart.hi(5) - chart.lo(5), kPi / 2, 1e-12);

  // Therefore solve must reach the bounded search rather than the all-fixed shortcut. A chain
  // misread as all-fixed answers from a single pose without running the optimiser at all, so
  // `starts` is the observable that separates the two.
  const ChainSolveResult result = chain.solve(Transform::identity());
  CHECK(result.starts >= 1);
  CHECK(result.nfev >= 1);
}

TEST(a_residual_exactly_at_the_tolerance_is_not_satisfied) {
  // The comparison is strictly `residual < tolerance`, matching the Python. Sampling will not
  // land on the boundary, so construct it: take the residual an all-fixed chain reports, then
  // ask again with the tolerance set to exactly that number.
  const Bounds6 fixed = bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}});
  const TSRChain chain({
      TSR(frame(0.2, 0.2, 0.2, 0.1, 0.2, 0.3), frame(0.1, 0, 0), fixed),
      TSR(Transform::identity(), frame(0, 0.05, 0), fixed),
  });
  const Transform target = frame(1.0, 0.5, -0.3, 0.2, 0.1, 0.4);

  const double residual = chain.solve(target).residual;
  CHECK(residual > 0.0);

  SolveOptions at_boundary{};
  at_boundary.tolerance = residual;  // exactly equal, so `<` fails and `<=` would pass
  CHECK(chain.solve(target, at_boundary).status == ChainStatus::kNotFound);

  SolveOptions just_above{};
  just_above.tolerance = std::nextafter(residual, 1.0e9);
  CHECK(chain.solve(target, just_above).status == ChainStatus::kSatisfied);

  // validate_witness compares the same way.
  const ChainCoords zeros{XyzRpy{0, 0, 0, 0, 0, 0}, XyzRpy{0, 0, 0, 0, 0, 0}};
  const double witness = geodesic_distance(chain.to_transform(zeros), target);
  CHECK(!chain.validate_witness(target, zeros, witness));
  CHECK(chain.validate_witness(target, zeros, std::nextafter(witness, 1.0e9)));
}

TEST(solve_validates_its_budget_before_it_looks_at_the_chain) {
  const TSRChain chain = door();
  const TSRChain empty;
  for (const TSRChain* c : {&chain, &empty}) {
    SolveOptions bad_starts{};
    bad_starts.max_starts = -1;
    CHECK_THROWS(c->solve(Transform::identity(), bad_starts), std::invalid_argument);

    SolveOptions bad_nfev{};
    bad_nfev.max_nfev = 0;
    CHECK_THROWS(c->solve(Transform::identity(), bad_nfev), std::invalid_argument);

    SolveOptions bad_tol{};
    bad_tol.tolerance = 0.0;
    CHECK_THROWS(c->solve(Transform::identity(), bad_tol), std::invalid_argument);

    SolveOptions negative_tol{};
    negative_tol.tolerance = -1.0;
    CHECK_THROWS(c->solve(Transform::identity(), negative_tol), std::invalid_argument);

    SolveOptions nan_tol{};
    nan_tol.tolerance = std::nan("");
    CHECK_THROWS(c->solve(Transform::identity(), nan_tol), std::invalid_argument);
  }
}

TEST(a_non_finite_guess_is_ignored_rather_than_rejected) {
  // A planner passing a neighbouring state's coordinates should not have to know whether that
  // neighbour was itself solved, so an unusable guess is dropped, not an error.
  const TSRChain chain = door();
  Rng rng(11);
  const ChainSample s = chain.sample_with_witness(rng);
  ChainCoords nan_guess = s.coordinates;
  nan_guess[0][5] = std::nan("");

  SolveOptions opt{};
  opt.initial_guess = nan_guess;
  opt.max_starts = 0;  // stay on an exact path, so this checks the guess handling alone
  const ChainSolveResult result = chain.solve(s.pose, opt);
  CHECK(result.starts == 0 && result.nfev == 0);
  // The guess was dropped, so the answer is the midpoint rather than the (unusable) guess.
  for (std::size_t j = 0; j < 6; ++j) CHECK(std::isfinite(result.coordinates[0][j]));
}

// --- the cold inverse ---------------------------------------------------------------------
//
// The cold path reports the best point an optimiser found, so it is held to properties rather
// than to recorded numbers -- a different optimiser evaluates a different set of points and
// nothing relates the two beyond "both are >= the true minimum" (#85, docs/CPP.md).
//
// Three of the properties carry a documented exception, each because the Python genuinely
// behaves that way and the corpus pins it. They are scoped here, and the cases that establish
// them live above: a_single_link_may_report_satisfied_with_a_residual_at_or_above_the_tolerance
// and an_out_of_chart_guess_within_the_containment_slack_is_still_accepted_verbatim.

TEST(the_cold_inverse_respects_its_budget_and_reports_what_it_spent) {
  for (const TSRChain& chain : cold_fixtures()) {
    Rng rng(99);
    for (int k = 0; k < 10; ++k) {
      const Transform T = chain.sample_with_witness(rng).pose;
      for (const int budget : {1, 5, 40, 2200}) {
        SolveOptions opt{};
        opt.max_nfev = budget;
        const ChainSolveResult result = chain.solve(T, opt);
        // Strictly within, not "about": the Python enforces this with its own counter rather
        // than trusting SciPy's maxfun, and a planner budgets against it.
        CHECK(result.nfev <= budget);
        CHECK(result.starts <= kDefaultMaxStarts);
        CHECK(result.nfev >= 0 && result.starts >= 0);
      }
      SolveOptions few{};
      few.max_starts = 2;
      CHECK(chain.solve(T, few).starts <= 2);
    }
  }
}

TEST(the_cold_inverses_residual_is_the_geodesic_at_the_coordinates_it_returns) {
  // The reported residual has to be measured at the coordinates handed back, or a caller
  // cannot act on either. This is what makes `residual` an upper bound rather than a number.
  for (const TSRChain& chain : cold_fixtures()) {
    Rng rng(7);
    for (int k = 0; k < 15; ++k) {
      const Transform T = chain.sample_with_witness(rng).pose;
      const ChainSolveResult result = chain.solve(T);
      CHECK(result.coordinates.size() == chain.size());
      const double measured = geodesic_distance(chain.to_transform(result.coordinates), T);
      CHECK_NEAR(result.residual, measured, 1e-12);
      // And a satisfied verdict means that residual really is inside the tolerance.
      if (result.status == ChainStatus::kSatisfied) CHECK(result.residual < kChainTolerance);
    }
  }
}

TEST(the_cold_inverse_returns_coordinates_inside_the_chart) {
  // The optimiser projects every step back into the box, so unlike the warm path -- which
  // returns the caller's guess verbatim -- these coordinates must lie in the chart exactly.
  for (const TSRChain& chain : cold_fixtures()) {
    Rng rng(21);
    for (int k = 0; k < 15; ++k) {
      const Transform T = chain.sample_with_witness(rng).pose;
      const ChainSolveResult result = chain.solve(T);
      for (std::size_t i = 0; i < chain.size(); ++i) {
        const Bounds6& b = chain.tsrs()[i].continuous_bounds();
        for (int j = 0; j < 6; ++j) {
          const double v = result.coordinates[i][static_cast<std::size_t>(j)];
          CHECK(v >= b.lo(j) && v <= b.hi(j));
        }
      }
    }
  }
}

TEST(the_cold_inverse_never_returns_a_worse_point_than_it_started_from) {
  // `residual` is documented as an upper bound on the true minimum, and the search reports the
  // best point it actually evaluated. The midpoint is always the first or second start, and
  // max_starts == 0 reports exactly that point without optimising -- so the full search must
  // never come back worse than it. A solver that reported its last iterate, or a rejected trial
  // step, would violate this while still looking plausible.
  for (const TSRChain& chain : cold_fixtures()) {
    Rng rng(31);
    for (int k = 0; k < 15; ++k) {
      const Transform T = chain.sample_with_witness(rng).pose;
      SolveOptions none{};
      none.max_starts = 0;
      const double unoptimised = chain.solve(T, none).residual;
      const double searched = chain.solve(T).residual;
      CHECK(searched <= unoptimised + 1e-12);
    }
  }
}

TEST(spending_more_of_the_budget_never_makes_the_objective_worse) {
  // The schedule and trajectory are deterministic, so a smaller budget evaluates a strict
  // prefix of what a larger one evaluates. The search keeps the best point it has seen, and the
  // best over a prefix cannot beat the best over the whole -- so the objective at the returned
  // coordinates must be non-increasing in max_nfev.
  //
  // Measured on the OBJECTIVE, ||r||^2, not on `residual`. Those are different orderings: the
  // search minimises the chordal objective because it is smooth, while `residual` reports the
  // geodesic at whatever point won. A later evaluation can therefore lower ||r||^2 and raise the
  // geodesic, so the geodesic genuinely is NOT monotone in the budget -- in either
  // implementation, since the Python minimises the same chordal objective and reports the same
  // geodesic. Asserting monotonicity of `residual` would be asserting something untrue.
  //
  // This is also the only non-circular way to pin "reports the best point found". Comparing
  // against the midpoint cannot: even a rejected trial step late in the search beats the
  // midpoint easily, so a solver returning its last evaluation rather than its best passes that
  // check and fails this one.
  const auto objective_at = [](const TSRChain& chain, const ChainCoords& c, const Transform& T) {
    const detail::ChainResidual at = detail::chain_residual(chain.tsrs(), c, T, false);
    double sq = 0.0;
    for (double v : at.r) sq += v * v;
    return sq;
  };

  for (const TSRChain& chain : cold_fixtures()) {
    Rng rng(43);
    for (int k = 0; k < 10; ++k) {
      const Transform T = chain.sample_with_witness(rng).pose;
      double previous = std::numeric_limits<double>::infinity();
      for (const int budget : {1, 2, 3, 5, 8, 13, 21, 50, 120, 400, 2200}) {
        SolveOptions opt{};
        opt.max_nfev = budget;
        const ChainSolveResult result = chain.solve(T, opt);
        const double objective = objective_at(chain, result.coordinates, T);
        CHECK(objective <= previous + 1e-12);
        previous = objective;
      }
    }
  }
}

TEST(the_cholesky_helper_solves_what_it_accepts_and_refuses_what_it_cannot) {
  // Isolated deliberately. The LM loop answers a non-positive-definite system by raising lambda
  // and retrying, which is self-correcting -- so a helper that silently returned nonsense would
  // be invisible end to end, the search merely rejecting the bad steps it produced. The failure
  // mode is real (a rank-deficient J^T J at gimbal lock) and only reachable here economically.
  {  // a positive-definite system, against a known solution
    std::vector<double> M{4.0, 1.0, 1.0, 3.0};
    std::vector<double> b{1.0, 2.0};
    CHECK(detail::cholesky_solve(M, b, 2));
    // 4x + y = 1, x + 3y = 2  =>  x = 1/11, y = 7/11. Checked against the original system
    // rather than against a restatement of the algorithm.
    CHECK_NEAR(4.0 * b[0] + 1.0 * b[1], 1.0, 1e-12);
    CHECK_NEAR(1.0 * b[0] + 3.0 * b[1], 2.0, 1e-12);
  }
  {  // indefinite: eigenvalues 3 and -1, so there is no Cholesky factor
    std::vector<double> M{1.0, 2.0, 2.0, 1.0};
    std::vector<double> b{1.0, 1.0};
    CHECK(!detail::cholesky_solve(M, b, 2));
  }
  {  // singular, the rank-deficient case the damping exists to rescue
    std::vector<double> M{0.0, 0.0, 0.0, 0.0};
    std::vector<double> b{1.0, 1.0};
    CHECK(!detail::cholesky_solve(M, b, 2));
  }
  {  // and a larger well-conditioned system, so the loops are exercised beyond 2x2
    const std::size_t m = 4;
    std::vector<double> A{5, 1, 0, 1, 1, 4, 1, 0, 0, 1, 3, 1, 1, 0, 1, 6};
    const std::vector<double> original = A;
    std::vector<double> b{1, -2, 3, 0.5};
    const std::vector<double> rhs = b;
    CHECK(detail::cholesky_solve(A, b, m));
    for (std::size_t i = 0; i < m; ++i) {
      double row = 0.0;
      for (std::size_t j = 0; j < m; ++j) row += original[i * m + j] * b[j];
      CHECK_NEAR(row, rhs[i], 1e-10);
    }
  }
}

TEST(the_cold_inverse_is_deterministic_run_to_run) {
  // Fixed-order loops and a locally seeded generator, so the same query gives bit-identical
  // results. A future parallelisation should fail this test rather than surprise a planner.
  for (const TSRChain& chain : cold_fixtures()) {
    Rng rng(5);
    for (int k = 0; k < 5; ++k) {
      const Transform T = chain.sample_with_witness(rng).pose;
      const ChainSolveResult a = chain.solve(T);
      const ChainSolveResult b = chain.solve(T);
      CHECK(a.status == b.status);
      CHECK(a.nfev == b.nfev && a.starts == b.starts);
      CHECK(a.residual == b.residual);  // bit-identical, not merely close
      CHECK(a.coordinates == b.coordinates);
    }
  }
}

TEST(a_single_start_begins_at_the_midpoint_or_the_canonicalised_guess) {
  // The first entries of the start schedule are arithmetic on the chart, so they are the part
  // that agrees with the Python. With max_starts == 1 and max_nfev == 1 exactly one evaluation
  // happens, at the first scheduled start, and the returned coordinates are that point.
  const TSRChain chain = door();
  SolveOptions one{};
  one.max_starts = 1;
  one.max_nfev = 1;
  const ChainSolveResult midpoint_start = chain.solve(frame(5, 5, 5), one);
  CHECK(midpoint_start.starts == 1 && midpoint_start.nfev == 1);
  for (std::size_t i = 0; i < chain.size(); ++i) {
    const Bounds6& b = chain.tsrs()[i].continuous_bounds();
    for (int j = 0; j < 6; ++j) {
      CHECK_NEAR(midpoint_start.coordinates[i][static_cast<std::size_t>(j)], (b.lo(j) + b.hi(j)) / 2.0, 1e-12);
    }
  }

  // A guess that does not validate becomes the first start instead, canonicalised through the
  // chart rather than clipped -- so a wrapping coordinate survives (#90).
  Rng rng(3);
  ChainCoords guess = chain.sample_with_witness(rng).coordinates;
  guess[0][5] += 2 * kPi;  // same angle, stated outside the chart
  SolveOptions warm_start{};
  warm_start.max_starts = 1;
  warm_start.max_nfev = 1;
  warm_start.initial_guess = guess;
  const ChainSolveResult guided = chain.solve(frame(5, 5, 5), warm_start);
  CHECK(guided.starts == 1 && guided.nfev == 1);
  const ChainCoords canonical = chain.continuous_coordinates(guess);
  CHECK_NEAR(guided.coordinates[0][5], canonical[0][5], 1e-12);
}

TEST(the_analytic_jacobian_matches_central_differences) {
  // A mathematical invariant, and the one failure a behavioural test can only report as "fewer
  // poses solved": rpy_to_rot is Z-Y-X, and the Jacobian's generator order depends on exactly
  // that. A self-consistent refactor of rpy_to_rot would silently decorrelate the two, so this
  // is the test that pins them together.
  double worst = 0.0;
  for (const TSRChain& chain : cold_fixtures()) {
    Rng rng(17);
    const std::size_t n = chain.size();
    for (int trial = 0; trial < 5; ++trial) {
      const ChainCoords c = chain.sample_xyzrpy(rng);
      const Transform target = chain.sample(rng);
      const detail::ChainResidual at = detail::chain_residual(chain.tsrs(), c, target, true);
      CHECK(at.J.size() == 12 * 6 * n);

      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
          const double h = 1e-6;
          ChainCoords plus = c, minus = c;
          plus[i][j] += h;
          minus[i][j] -= h;
          const detail::ChainResidual rp = detail::chain_residual(chain.tsrs(), plus, target, false);
          const detail::ChainResidual rm = detail::chain_residual(chain.tsrs(), minus, target, false);
          for (std::size_t row = 0; row < 12; ++row) {
            const double numeric = (rp.r[row] - rm.r[row]) / (2 * h);
            const double analytic = at.J[row * 6 * n + i * 6 + j];
            worst = std::max(worst, std::fabs(numeric - analytic));
          }
        }
      }
    }
  }
  std::printf("        analytic Jacobian vs central differences: max |diff| = %.3e\n", worst);
  CHECK(worst <= 1e-7);
}

TEST(the_pose_the_pythons_cold_solve_cannot_find_is_found_here) {
  // The deterministic counterexample from issue #85, transcribed from
  // tests/tsr/test_tsr_chain.py::_rotation_rich_fixture. The pose is built from known-valid
  // coordinates, so it is provably a member -- and the Python's cold solve still settles in a
  // nonzero basin and reports "not_found". Its test asserts exactly that failure, which is why
  // that test is the one thing in the chain contract that cannot cross over: it pins a property
  // of SciPy's trajectory, not of the rules.
  //
  // This solver finds it, which is the concrete evidence for that paragraph in docs/CPP.md.
  // Being better is still not a licence to record cold results in the corpus: "not_found" is
  // not a proof of non-membership in either implementation (#85).
  const TSR first(
      Transform::from_rows({{{-0.05323397734171213, 0.8948213760632387, 0.443239042274791, 0.22846326913496368},
                             {-0.9701532216431458, -0.15150283374468387, 0.18933995326596015, -0.14982700307842703},
                             {0.2365774084561063, -0.41993046603887, 0.8761789392016737, -0.009460041281242981},
                             {0.0, 0.0, 0.0, 1.0}}}),
      Transform::from_rows({{{0.6293426752234571, -0.1728436326485694, 0.7576627718156862, -0.14378930957175456},
                             {-0.04665111723551994, -0.9815968691424982, -0.18517899381496544, -0.00873891803731176},
                             {0.7757264146612902, 0.08119522856973581, -0.6258241481872112, 0.06216845799526165},
                             {0.0, 0.0, 0.0, 1.0}}}),
      bounds({{{0.0, 0.0},
               {0.0, 0.0},
               {0.0, 0.0},
               {-0.08467294501938391, 3.050198946370484},
               {0.0, 0.0},
               {-2.7980033218941758, 1.030225743778528}}}));
  const TSR second(
      Transform::identity(),
      Transform::from_rows({{{0.18723087829170545, -0.6170802787911787, 0.7643013330755861, 0.026865480304672573},
                             {0.7086069079464304, 0.6236951990956814, 0.3299705269195986, -0.09004582889149249},
                             {-0.6803093768460903, 0.4798085328044926, 0.5540423482219428, -0.10766560142469507},
                             {0.0, 0.0, 0.0, 1.0}}}),
      bounds({{{-0.000750418289075433, 0.4666823942550439},
               {0.0, 0.0},
               {0.0, 0.0},
               {-2.5160093677919146, -0.6584844308933367},
               {0.0, 0.0},
               {-3.0738651972185425, 1.704964963361112}}}));
  const TSRChain chain({first, second});
  const ChainCoords witness{XyzRpy{0.0, 0.0, 0.0, -0.03763487464809723, 0.0, 0.3720955229542815},
                            XyzRpy{0.07774021221016615, 0.0, 0.0, -1.094901874015342, 0.0, 0.05323198828225317}};

  // The witness certifies membership with no optimiser, in both implementations.
  const Transform pose = chain.to_transform(witness);
  CHECK(chain.validate_witness(pose, witness));

  const ChainSolveResult cold = chain.solve(pose);
  std::printf("        issue #85 counterexample: %s, residual %.3e, %d evaluations over %d start(s)\n",
              to_string(cold.status), cold.residual, cold.nfev, cold.starts);
  CHECK(cold.status == ChainStatus::kSatisfied);
  CHECK(cold.residual < kChainTolerance);
}

TEST(the_cold_inverse_finds_a_witness_for_a_pose_that_provably_has_one) {
  // Every target here came from sample_with_witness, so a witness exists by construction and
  // "recall" means something. The floor is what catches a broken Jacobian or a dropped
  // projection: measured against this suite, a finite-difference Jacobian scores 31/40 and
  // 18/40 on two of these chains and dropping the gradient projection scores 18/40, while the
  // implementation here scores 40/40, 40/40, 40/40 and 38/40.
  //
  // "not_found" is never a proof of non-membership (#85), so this is a floor on a bounded
  // search, not a correctness oracle -- which is why it has real headroom rather than
  // demanding perfection.
  constexpr int kPerChain = 40;
  const std::vector<const char*> names{"door_hinge_handle", "two_yaws_and_a_slide", "three_links", "rotation_rich"};
  const std::vector<TSRChain> fixtures = cold_fixtures();

  std::string report = "chain cold-inverse recall (issue #165)\n\n";
  int total = 0, found = 0;
  std::size_t index = 0;
  for (const TSRChain& chain : fixtures) {
    Rng rng(20261002);  // the same poses every run
    int hits = 0;
    long nfev_sum = 0;
    for (int k = 0; k < kPerChain; ++k) {
      const ChainSample s = chain.sample_with_witness(rng);
      const ChainSolveResult result = chain.solve(s.pose);
      if (result.status == ChainStatus::kSatisfied) ++hits;
      nfev_sum += result.nfev;
      // Whatever the cold search concluded, the witness itself still certifies membership.
      CHECK(chain.validate_witness(s.pose, s.coordinates));
    }
    total += kPerChain;
    found += hits;
    char line[160];
    std::snprintf(line, sizeof(line), "  %-24s %2d/%2d  mean nfev %6.1f\n", names[index], hits, kPerChain,
                  static_cast<double>(nfev_sum) / kPerChain);
    report += line;
    std::printf("      %s", line);
    // Per-chain floor, well below the measured values, so a real regression trips it before
    // ordinary numerical drift does.
    CHECK(hits * 100 >= kPerChain * 85);
    ++index;
  }
  char summary[160];
  std::snprintf(summary, sizeof(summary), "\n  aggregate %d/%d (%.1f%%)\n", found, total,
                100.0 * found / total);
  report += summary;
  std::printf("    %s", summary);
  CHECK(found * 100 >= total * 90);

  // The inspectable artifact: deterministic, repeatable, and reproduced by
  // `ctest --test-dir build -R tsr_chain`.
  std::ofstream out("chain_properties_report.txt");
  if (out) out << report;
}

TEST(geodesic_distance_is_zero_on_equal_poses_and_symmetric) {
  const Transform a = frame(0.3, -0.2, 0.5, 0.4, -0.3, 1.1);
  const Transform b = frame(-0.1, 0.4, 0.2, -0.2, 0.5, -0.7);
  // The acos clamp is what keeps this exactly 0 rather than NaN: the trace of R^T R lands a
  // rounding step above 3 for a pose compared with itself.
  CHECK(geodesic_distance(a, a) == 0.0);
  CHECK_NEAR(geodesic_distance(a, b), geodesic_distance(b, a), 1e-15);
  const Transform shifted = frame(0.3 + 0.25, -0.2, 0.5, 0.4, -0.3, 1.1);
  CHECK_NEAR(geodesic_distance(a, shifted), 0.25, 1e-12);
}

HARNESS_MAIN()
