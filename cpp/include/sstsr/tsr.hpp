// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
#pragma once

#include <array>
#include <optional>
#include <utility>

#include "sstsr/rng.hpp"
#include "sstsr/transform.hpp"

namespace sstsr {

// core/utils.py's CHAIN_TOLERANCE: the default satisfaction tolerance of a chain's inverse
// solve. A solver tolerance, not part of any region's geometry: contains does not use it (#190).
constexpr double kChainTolerance = 0.001;

// Deprecated alias of kChainTolerance. Until #190 it was also the slack contains granted on
// every side of every bound; it no longer is.
[[deprecated("use kChainTolerance; contains no longer grants this slack (#190)")]]
constexpr double kEpsilon = kChainTolerance;

// core/utils.py's ANGLE_ATOL and length_atol: the round-off contains and is_valid absorb, and
// nothing more. Translation's scales with the lengths composed into the pose (see TSR::length_atol).
constexpr double kAngleAtol = 1e-6;
inline double length_atol(double scale) { return 1e-9 + 1e-6 * scale; }

// [lo, hi] for x, y, z, roll, pitch, yaw. A rotational row with hi < lo is an outer interval
// wrapping through +-pi.
struct Bounds6 {
  std::array<std::array<double, 2>, 6> rows{};
  double lo(int i) const { return rows[static_cast<std::size_t>(i)][0]; }
  double hi(int i) const { return rows[static_cast<std::size_t>(i)][1]; }
};

// A Task Space Region. The same region as src/tsr/core/tsr.py's TSR, rule for rule; the
// rules themselves are stated in docs/ARCHITECTURE.md and the two implementations are held
// to them by the conformance corpus (docs/CPP.md).
//
// Not here, and deliberately: serialization, the optional scipy distance path
// (distance_optimize), masked sampling, and a rotation_weight other than 1.
class TSR {
 public:
  TSR(Transform T0_w, Transform Tw_e, Bounds6 Bw);  // throws std::invalid_argument, same reasons as the Python

  const Transform& T0_w() const { return T0_w_; }
  const Transform& Tw_e() const { return Tw_e_; }
  const Bounds6& Bw() const { return Bw_; }
  const Bounds6& continuous_bounds() const { return cont_; }  // the Python's _Bw_cont

  bool contains(const Transform& T) const;  // within Bw up to round-off only (kAngleAtol, length_atol)
  double distance(const Transform& T) const;                      // 0 if contained; Berenson et al. 2011 Sec. 4.2 otherwise
  std::pair<double, XyzRpy> distance_bwopt(const Transform& T) const;  // the Python's distance(): (dist, bwopt)
  std::pair<double, Transform> closest_transform(const Transform& T) const;
  XyzRpy sample_xyzrpy(Rng& rng) const;                            // six unit draws in coordinate order
  Transform sample(Rng& rng) const;
  Transform to_transform(const XyzRpy& v) const;                   // T0_w * xyzrpy_to_trans(v) * Tw_e; validates
  XyzRpy to_xyzrpy(const Transform& T) const;
  double volume() const;                                           // the Python's _interval_sum

  std::array<bool, 6> is_valid(const XyzRpy& v) const;

 private:
  Transform local(const Transform& T) const;  // inv(T0_w) * T * inv(Tw_e)
  double length_atol(const std::array<double, 3>& xyz) const;  // the Python's TSR._length_atol
  std::array<bool, 3> xyz_within(const std::array<double, 3>& xyz) const;
  std::array<bool, 3> rpy_within(const Rpy& rpy) const;
  std::pair<std::array<bool, 3>, std::optional<Rpy>> rot_within_rpy_bounds(const Transform& local) const;
  std::pair<XyzRpy, XyzRpy> displacement(const Transform& T) const;  // (dx, dw)

  Transform T0_w_, Tw_e_, T0_w_inv_, Tw_e_inv_;
  Bounds6 Bw_, cont_;
};

}  // namespace sstsr
