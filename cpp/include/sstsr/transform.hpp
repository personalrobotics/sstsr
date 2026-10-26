// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
#pragma once

#include <array>
#include <optional>
#include <string>

namespace sstsr {

// A 4x4 homogeneous transform, row-major. Deliberately not Eigen: a C++ consumer of a
// TSR should not inherit a linear-algebra dependency to hold a pose, so this is the
// handful of operations the region math actually needs. See docs/CPP.md.
struct Transform {
  std::array<double, 16> m{};

  static Transform identity();
  static Transform from_rows(const std::array<std::array<double, 4>, 4>& rows);
  double at(int r, int c) const { return m[static_cast<std::size_t>(r * 4 + c)]; }
  double& at(int r, int c) { return m[static_cast<std::size_t>(r * 4 + c)]; }
  Transform operator*(const Transform& o) const;
  Transform inverse_rigid() const;  // for a rigid transform: transpose the rotation, negate-rotate the translation
  bool operator==(const Transform& o) const { return m == o.m; }
};

using Rpy = std::array<double, 3>;
using XyzRpy = std::array<double, 6>;

// The conventions of src/tsr/core/tsr.py, which is the source of truth for the rules.
// rpy_to_rot is Z-Y-X: yaw about z, then pitch about y, then
// roll about x. rot_to_rpy takes pitch = -asin(R[2,0]) off the gimbal lock (|R[2,0]| within 1e-14 of 1)
// and the coupled branches at the lock with yaw fixed to 0.
std::array<double, 9> rpy_to_rot(const Rpy& rpy);           // row-major 3x3
Rpy rot_to_rpy(const Transform& T);                          // from T's rotation block
Transform xyzrpy_to_trans(const XyzRpy& v);
XyzRpy trans_to_xyzrpy(const Transform& T);

// Wrap each angle into [lower, lower + 2pi), as core/utils.py's wrap_to_interval -- including
// its guard against a modulo result rounding up to exactly 2pi, which would push the result
// out of the half-open interval.
double wrap_to_interval(double angle, double lower);

// core/utils.py's geodesic_distance: || [t2 - t1, angle(R1^T R2)] ||, mixing metres and
// radians as Berenson et al. 2011 Sec. 4.2 does. The Python's rotation weight r is omitted
// for the same reason TSR omits a rotation_weight other than 1: nothing in the library uses
// anything else, and test_cpp_package.py pins the Python's default at 1.0 so a change there
// cannot silently diverge from this.
double geodesic_distance(const Transform& t1, const Transform& t2);

// The construction contract of docs/ARCHITECTURE.md (3.2.0, issue #162): finite, last row
// 0 0 0 1, R R^T = I and det R = +1 within FRAME_ATOL. Returns the violation, or nullopt.
//
// kFrameAtol is tsr.FRAME_ATOL. It is public in the Python for exactly this reason: both
// implementations must reject the same frames at the same boundary.
constexpr double kFrameAtol = 1e-6;
std::optional<std::string> why_not_frame(const Transform& T, const std::string& name);

}  // namespace sstsr
