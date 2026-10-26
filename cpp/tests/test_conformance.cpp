// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
//
// The C++ core answers what the Python answers.
//
// The Python in src/tsr/core/ is the source of truth for the rules; this holds the C++ to
// them on a checked-in corpus of the Python's recorded answers
// (tools/tsr_conformance.py -> tests/reference/tsr_conformance.json). Neither
// implementation reads the other, and a rule change shows up as a corpus diff.
//
// Tolerances, and why they differ: the continuous-bounds chart and the volume are
// arithmetic on the inputs, so they must agree to 1e-12; contains is a verdict, so it must
// agree exactly; distance and the closest transform run through asin/atan2 and a
// nine-candidate minimum, so 1e-9 is the honest floor across standard libraries. These are
// the same tolerances the corpus's own --check mode uses.
//
// What this does NOT check, deliberately: the sampled coordinates the corpus records under
// `sample_xyzrpy_seed7_python`. The two implementations draw from different engines by
// design (see sstsr/rng.hpp), so sampling is checked as a property instead -- in bounds and
// contained -- in test_tsr.cpp. See docs/CPP.md.
#include <cmath>
#include <cstdio>
#include <string>

#include <stdexcept>
#include <vector>

#include "harness.hpp"
#include "json.hpp"
#include "sstsr/transform.hpp"
#include "sstsr/tsr.hpp"
#include "sstsr/tsr_chain.hpp"

using namespace sstsr;

namespace {

constexpr double kChartTol = 1e-12;  // arithmetic on the inputs
constexpr double kPoseTol = 1e-9;    // through asin/atan2 and a candidate minimum

// A geodesic residual gets a looser floor, and not for sloppiness: its rotation term is
// acos((trace - 1) / 2), and acos is infinitely steep at 1. For two poses agreeing to the last
// bit, a 1-ulp error in the trace reports acos(1 - 2.2e-16) = 2.1e-08 of angle, and
// re-associating an algebraically identical product moves it by up to 4.2e-08. A near-zero
// residual is therefore only meaningful to about 1e-7. Matches RESIDUAL_ATOL in
// tools/tsr_conformance.py.
constexpr double kResidualTol = 1e-7;

Transform transform_from(const testjson::Value& rows) {
  std::array<std::array<double, 4>, 4> m{};
  for (std::size_t r = 0; r < 4; ++r) {
    for (std::size_t c = 0; c < 4; ++c) m[r][c] = rows[r][c].number();
  }
  return Transform::from_rows(m);
}

Bounds6 bounds_from(const testjson::Value& rows) {
  Bounds6 b{};
  for (std::size_t i = 0; i < 6; ++i) b.rows[i] = {rows[i][0].number(), rows[i][1].number()};
  return b;
}

const testjson::Value& corpus() {
  static const testjson::Value v = testjson::parse_file(SSTSR_CONFORMANCE_CORPUS);
  return v;
}

std::string where(const std::string& region, std::size_t probe) {
  return region + " probe " + std::to_string(probe);
}

// Keyed lookup throws on a missing key (deliberately, so a grown corpus fails legibly), so an
// optional section needs an explicit presence check.
bool has(const testjson::Value& v, const std::string& key) {
  const auto& o = v.object();
  return o.find(key) != o.end();
}

ChainCoords coords_from(const testjson::Value& rows) {
  const auto& a = rows.array();
  ChainCoords out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (std::size_t j = 0; j < 6; ++j) out[i][j] = a[i][j].number();
  }
  return out;
}

TSRChain chain_from(const testjson::Value& entry) {
  std::vector<TSR> links;
  for (const auto& link : entry["links"].array()) {
    links.emplace_back(transform_from(link["T0_w"]), transform_from(link["Tw_e"]), bounds_from(link["Bw"]));
  }
  return TSRChain(links);
}

void check_transform(const Transform& got, const testjson::Value& expected, const std::string& what, double tol) {
  for (std::size_t r = 0; r < 4; ++r) {
    for (std::size_t c = 0; c < 4; ++c) {
      if (std::fabs(got.at(static_cast<int>(r), static_cast<int>(c)) - expected[r][c].number()) > tol) {
        harness::fail((what + " entry (" + std::to_string(r) + "," + std::to_string(c) + ") disagrees").c_str(),
                      __FILE__, __LINE__);
      }
    }
  }
}

void check_coords(const ChainCoords& got, const testjson::Value& expected, const std::string& what, double tol) {
  const auto& a = expected.array();
  if (got.size() != a.size()) harness::fail((what + ": wrong link count").c_str(), __FILE__, __LINE__);
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (std::size_t j = 0; j < 6; ++j) {
      if (std::fabs(got[i][j] - a[i][j].number()) > tol) {
        harness::fail((what + " link " + std::to_string(i) + " coordinate " + std::to_string(j) + " disagrees").c_str(),
                      __FILE__, __LINE__);
      }
    }
  }
}

}  // namespace

TEST(corpus_is_the_one_this_build_expects) {
  // A corpus that cannot be read, or that is empty, must fail rather than vacuously pass:
  // this test is the only thing holding the C++ to the Python.
  const testjson::Value& c = corpus();
  const auto& regions = c["regions"].array();
  CHECK(regions.size() == 11);
  std::size_t probes = 0;
  for (const auto& r : regions) probes += r["probes"].array().size();
  CHECK(probes == 282);

  // The chain half, counted for the same reason: a missing `chains` key fails loudly at the
  // keyed lookup, but a zero-length array would pass every chain case vacuously.
  const auto& chains = c["chains"].array();
  CHECK(chains.size() == 8);
  std::size_t coordinate_probes = 0, witness_probes = 0, exact_solves = 0;
  for (const auto& ch : chains) {
    coordinate_probes += ch["coordinate_probes"].array().size();
    witness_probes += ch["witness_probes"].array().size();
    exact_solves += ch["exact_solves"].array().size();
  }
  CHECK(coordinate_probes == 72);
  CHECK(witness_probes == 48);
  CHECK(exact_solves == 18);

  std::printf("        corpus: sstsr %s, numpy %s, %zu regions, %zu probes; %zu chains, %zu/%zu/%zu chain probes\n",
              c["versions"]["sstsr"].string().c_str(), c["versions"]["numpy"].string().c_str(), regions.size(),
              probes, chains.size(), coordinate_probes, witness_probes, exact_solves);
}

TEST(construction_and_the_continuous_chart_agree) {
  for (const auto& region : corpus()["regions"].array()) {
    const std::string name = region["name"].string();
    const TSR t(transform_from(region["T0_w"]), transform_from(region["Tw_e"]), bounds_from(region["Bw"]));

    const Bounds6 expected = bounds_from(region["continuous_bounds"]);
    for (int i = 0; i < 6; ++i) {
      if (std::fabs(t.continuous_bounds().lo(i) - expected.lo(i)) > kChartTol ||
          std::fabs(t.continuous_bounds().hi(i) - expected.hi(i)) > kChartTol) {
        harness::fail(("continuous bounds row " + std::to_string(i) + " of " + name).c_str(), __FILE__, __LINE__);
      }
    }
    CHECK_NEAR(t.volume(), region["volume"].number(), kChartTol);
  }
}

TEST(every_probe_agrees) {
  for (const auto& region : corpus()["regions"].array()) {
    const std::string name = region["name"].string();
    const TSR t(transform_from(region["T0_w"]), transform_from(region["Tw_e"]), bounds_from(region["Bw"]));
    const auto& probes = region["probes"].array();

    for (std::size_t p = 0; p < probes.size(); ++p) {
      const auto& probe = probes[p];
      const Transform T = transform_from(probe["T"]);

      if (t.contains(T) != probe["contains"].boolean()) {
        harness::fail(("contains disagrees at " + where(name, p)).c_str(), __FILE__, __LINE__);
      }
      if (std::fabs(t.distance(T) - probe["distance"].number()) > kPoseTol) {
        harness::fail(("distance disagrees at " + where(name, p)).c_str(), __FILE__, __LINE__);
      }

      const auto [dist, closest] = t.closest_transform(T);
      if (std::fabs(dist - probe["distance"].number()) > kPoseTol) {
        harness::fail(("closest_transform's distance disagrees at " + where(name, p)).c_str(), __FILE__, __LINE__);
      }
      const Transform expected_closest = transform_from(probe["closest"]);
      for (std::size_t k = 0; k < 16; ++k) {
        if (std::fabs(closest.m[k] - expected_closest.m[k]) > kPoseTol) {
          harness::fail(("closest transform entry " + std::to_string(k) + " disagrees at " + where(name, p)).c_str(),
                        __FILE__, __LINE__);
        }
      }
      // The Python records whether its own closest pose is itself contained. A closest
      // pose that falls outside the region it was projected into is the failure this
      // catches, and it is a verdict, so it must agree exactly.
      if (t.contains(closest) != probe["closest_contained"].boolean()) {
        harness::fail(("closest_contained disagrees at " + where(name, p)).c_str(), __FILE__, __LINE__);
      }

      // to_xyzrpy chooses between the two RPY representatives of the same rotation, and that
      // choice is invisible in contains, distance and closest_transform -- which is exactly
      // how #171 stayed hidden behind 162 agreeing probes. Recorded and checked since.
      const XyzRpy xyzrpy = t.to_xyzrpy(T);
      const std::array<bool, 6> valid = t.is_valid(xyzrpy);
      for (std::size_t k = 0; k < 6; ++k) {
        if (std::fabs(xyzrpy[k] - probe["to_xyzrpy"][k].number()) > kPoseTol) {
          harness::fail(("to_xyzrpy coordinate " + std::to_string(k) + " disagrees at " + where(name, p)).c_str(),
                        __FILE__, __LINE__);
        }
        if (valid[k] != probe["to_xyzrpy_is_valid"][k].boolean()) {
          harness::fail(
              ("to_xyzrpy_is_valid at coordinate " + std::to_string(k) + " disagrees at " + where(name, p)).c_str(),
              __FILE__, __LINE__);
        }
      }
    }
  }
}

// --- chains (issue #165 stage 2) --------------------------------------------------------
//
// The forward half of the chain contract, which IS specifiable across implementations: the
// chart, the composition, the witness verdicts, and the solve paths that reach an answer
// without the optimiser. The cold inverse is not here and never will be -- it reports the best
// point a particular optimiser found, so it is checked by properties in test_tsr_chain.cpp
// instead. See docs/CPP.md.

TEST(chain_charts_and_forward_composition_agree) {
  for (const auto& entry : corpus()["chains"].array()) {
    const std::string name = entry["name"].string();
    const TSRChain chain = chain_from(entry);
    CHECK(chain.size() == entry["links"].array().size());

    // Each link's continuous chart, and the free mask and midpoint derived from it. The
    // midpoint is what `max_starts == 0` reports, and for a wrapping interval it lands outside
    // [-pi, pi) on purpose -- re-normalising it would move the recorded answer.
    //
    // Note what this does and does not establish. The chart values are production output, so
    // comparing them to the Python's is real conformance. The free mask, though, is recomputed
    // here from those values, so it only shows the two implementations agree about which rows
    // are free -- not that solve reads the mask off the chart rather than off Bw. That is a
    // behavioural question, and test_tsr_chain.cpp's
    // a_chain_whose_only_freedom_is_an_outer_interval_is_not_mistaken_for_all_fixed covers it.
    const auto& free_mask = entry["free_mask"].array();
    for (std::size_t i = 0; i < chain.size(); ++i) {
      const Bounds6& got = chain.tsrs()[i].continuous_bounds();
      const Bounds6 expected = bounds_from(entry["links"][i]["continuous_bounds"]);
      for (int j = 0; j < 6; ++j) {
        if (std::fabs(got.lo(j) - expected.lo(j)) > kChartTol || std::fabs(got.hi(j) - expected.hi(j)) > kChartTol) {
          harness::fail((name + " link " + std::to_string(i) + " chart row " + std::to_string(j)).c_str(), __FILE__,
                        __LINE__);
        }
        const bool is_free = got.hi(j) > got.lo(j);
        if (is_free != free_mask[i * 6 + static_cast<std::size_t>(j)].boolean()) {
          harness::fail((name + " free mask at link " + std::to_string(i) + " coordinate " + std::to_string(j)).c_str(),
                        __FILE__, __LINE__);
        }
        const double midpoint = (got.lo(j) + got.hi(j)) / 2.0;
        if (std::fabs(midpoint - entry["midpoint"][i][static_cast<std::size_t>(j)].number()) > kChartTol) {
          harness::fail((name + " midpoint at link " + std::to_string(i) + " coordinate " + std::to_string(j)).c_str(),
                        __FILE__, __LINE__);
        }
      }
    }

    const auto& probes = entry["coordinate_probes"].array();
    for (std::size_t p = 0; p < probes.size(); ++p) {
      const ChainCoords input = coords_from(probes[p]["input"]);
      const std::string at = name + " coordinate probe " + std::to_string(p);

      check_coords(chain.continuous_coordinates(input), probes[p]["continuous"], at + " continuous", kChartTol);
      check_transform(chain.to_transform(input), probes[p]["to_transform"], at + " to_transform", kPoseTol);

      const std::vector<std::array<bool, 6>> valid = chain.is_valid(input);
      for (std::size_t i = 0; i < valid.size(); ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
          if (valid[i][j] != probes[p]["is_valid"][i][j].boolean()) {
            harness::fail((at + " is_valid at link " + std::to_string(i) + " coordinate " + std::to_string(j)).c_str(),
                          __FILE__, __LINE__);
          }
        }
      }
    }
  }
}

TEST(chain_witness_verdicts_agree) {
  for (const auto& entry : corpus()["chains"].array()) {
    const std::string name = entry["name"].string();
    const TSRChain chain = chain_from(entry);
    const auto& probes = entry["witness_probes"].array();

    for (std::size_t p = 0; p < probes.size(); ++p) {
      const ChainCoords c = coords_from(probes[p]["coordinates"]);
      const Transform T = transform_from(probes[p]["target"]);
      const std::string at = name + " witness probe " + std::to_string(p);

      // The verdict is the contract and must agree exactly.
      if (chain.validate_witness(T, c) != probes[p]["validate_witness"].boolean()) {
        harness::fail((at + ": validate_witness disagrees").c_str(), __FILE__, __LINE__);
      }

      // The Python records a residual only when the coordinates are in bounds; otherwise they
      // are not a witness at all and it records null. In the in-bounds case the residual is the
      // geodesic from the recomposed pose to the target, which is reachable through the public
      // API -- so this checks the number without reaching into a private helper.
      bool in_bounds = true;
      for (const std::array<bool, 6>& row : chain.is_valid(c)) {
        for (bool ok : row) in_bounds = in_bounds && ok;
      }
      if (probes[p]["residual"].is_null()) {
        if (in_bounds) harness::fail((at + ": the Python saw no witness here but the C++ does").c_str(), __FILE__, __LINE__);
      } else {
        if (!in_bounds) harness::fail((at + ": the Python saw a witness here but the C++ does not").c_str(), __FILE__, __LINE__);
        const double residual = geodesic_distance(chain.to_transform(c), T);
        if (std::fabs(residual - probes[p]["residual"].number()) > kResidualTol) {
          harness::fail((at + ": witness residual disagrees").c_str(), __FILE__, __LINE__);
        }
      }
    }
  }
}

TEST(chain_exact_solves_agree) {
  for (const auto& entry : corpus()["chains"].array()) {
    const std::string name = entry["name"].string();
    const TSRChain chain = chain_from(entry);

    for (const auto& probe : entry["exact_solves"].array()) {
      const std::string at = name + " exact solve " + probe["path"].string();
      SolveOptions opt{};
      opt.max_starts = static_cast<int>(probe["max_starts"].number());
      if (!probe["initial_guess"].is_null()) opt.initial_guess = coords_from(probe["initial_guess"]);

      const ChainSolveResult result = chain.solve(transform_from(probe["target"]), opt);

      if (std::string(to_string(result.status)) != probe["status"].string()) {
        harness::fail((at + ": status disagrees").c_str(), __FILE__, __LINE__);
      }
      // nfev and starts are the whole point of an exact path: it reached its answer without
      // spending any of the budget, and that accounting is caller-visible.
      if (result.nfev != static_cast<int>(probe["nfev"].number())) {
        harness::fail((at + ": nfev disagrees").c_str(), __FILE__, __LINE__);
      }
      if (result.starts != static_cast<int>(probe["starts"].number())) {
        harness::fail((at + ": starts disagrees").c_str(), __FILE__, __LINE__);
      }
      check_coords(result.coordinates, probe["coordinates"], at + " coordinates", kPoseTol);
      if (std::fabs(result.residual - probe["residual"].number()) > kResidualTol) {
        harness::fail((at + ": residual disagrees").c_str(), __FILE__, __LINE__);
      }
    }
  }
}

TEST(chain_delegating_methods_agree_where_solve_is_exact) {
  // distance, closest_transform, contains and to_xyzrpy all delegate to solve, so they are only
  // comparable where solve is exact: a single link, or a chain with no free coordinate. The
  // corpus records them nowhere else, and this walks exactly what it recorded.
  std::size_t checked = 0;
  for (const auto& entry : corpus()["chains"].array()) {
    if (!has(entry, "delegating")) continue;
    const std::string name = entry["name"].string();
    const TSRChain chain = chain_from(entry);
    const auto& d = entry["delegating"];
    const Transform T = transform_from(d["target"]);
    ++checked;

    const auto [dist, coords] = chain.distance(T);
    if (std::fabs(dist - d["distance"].number()) > kResidualTol) {
      harness::fail((name + ": delegating distance disagrees").c_str(), __FILE__, __LINE__);
    }
    check_coords(coords, d["distance_coordinates"], name + " distance coordinates", kPoseTol);

    const auto [cdist, closest] = chain.closest_transform(T);
    if (std::fabs(cdist - d["closest_distance"].number()) > kResidualTol) {
      harness::fail((name + ": delegating closest distance disagrees").c_str(), __FILE__, __LINE__);
    }
    check_transform(closest, d["closest_transform"], name + " closest_transform", kPoseTol);

    if (chain.contains(T) != d["contains"].boolean()) {
      harness::fail((name + ": delegating contains disagrees").c_str(), __FILE__, __LINE__);
    }
    check_coords(chain.to_xyzrpy(T), d["to_xyzrpy"], name + " to_xyzrpy", kPoseTol);
  }
  CHECK(checked == 2);  // single_link and all_fixed; a silent drop to 0 would pass vacuously
}

TEST(the_empty_chain_agrees) {
  // Four answers that are asymmetric in the Python and easy to "tidy up" in a port.
  const auto& e = corpus()["empty_chain"];
  const TSRChain empty;
  CHECK(empty.size() == 0);

  const ChainSolveResult result = empty.solve(Transform::identity());
  CHECK(std::string(to_string(result.status)) == e["solve"]["status"].string());
  CHECK(result.coordinates.empty());
  CHECK(result.nfev == 0 && result.starts == 0);
  CHECK(e["solve"]["residual"].is_null() && std::isinf(result.residual));

  CHECK(empty.contains(Transform::identity()) == e["contains"].boolean());
  CHECK(empty.validate_witness(Transform::identity(), {}) == e["validate_witness"].boolean());
  CHECK(empty.to_xyzrpy(Transform::identity()).empty() && e["to_xyzrpy"].array().empty());

  const auto [dist, coords] = empty.distance(Transform::identity());
  CHECK(e["distance"].is_null() && std::isinf(dist));
  CHECK(coords.empty());

  // closest_transform raises, because there is no pose to compose. An identity transform here
  // would be a confident wrong answer.
  CHECK(e["closest_transform_raises"].boolean());
  CHECK_THROWS(empty.closest_transform(Transform::identity()), std::invalid_argument);
}

HARNESS_MAIN()
