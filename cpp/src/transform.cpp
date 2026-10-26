// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
#include "sstsr/transform.hpp"

#include <cmath>
#include <numbers>

namespace sstsr {

namespace {
constexpr double kPi = std::numbers::pi;
constexpr double kGimbalEpsilon = 1e-14;  // core/tsr.py's _GIMBAL_EPSILON
}  // namespace

Transform Transform::identity() {
  Transform t;
  for (int i = 0; i < 4; ++i) t.at(i, i) = 1.0;
  return t;
}

Transform Transform::from_rows(const std::array<std::array<double, 4>, 4>& rows) {
  Transform t;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) t.at(r, c) = rows[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
  }
  return t;
}

Transform Transform::operator*(const Transform& o) const {
  Transform out;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      double s = 0.0;
      for (int k = 0; k < 4; ++k) s += at(r, k) * o.at(k, c);
      out.at(r, c) = s;
    }
  }
  return out;
}

Transform Transform::inverse_rigid() const {
  Transform out = identity();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) out.at(r, c) = at(c, r);
  }
  for (int r = 0; r < 3; ++r) {
    double s = 0.0;
    for (int k = 0; k < 3; ++k) s += out.at(r, k) * at(k, 3);
    out.at(r, 3) = -s;
  }
  return out;
}

std::array<double, 9> rpy_to_rot(const Rpy& rpy) {
  const double r = rpy[0], p = rpy[1], y = rpy[2];
  const double cr = std::cos(r), sr = std::sin(r), cp = std::cos(p), sp = std::sin(p), cy = std::cos(y), sy = std::sin(y);
  return {cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy,
          cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy,
          -sp,     sr * cp,                cr * cp};
}

Rpy rot_to_rpy(const Transform& T) {
  double r20 = T.at(2, 0);
  r20 = r20 < -1.0 ? -1.0 : (r20 > 1.0 ? 1.0 : r20);
  Rpy rpy{};
  if (std::fabs(std::fabs(r20) - 1.0) >= kGimbalEpsilon) {
    const double p = -std::asin(r20);
    const double cp = std::cos(p);
    rpy[0] = std::atan2(T.at(2, 1) / cp, T.at(2, 2) / cp);
    rpy[1] = p;
    rpy[2] = std::atan2(T.at(1, 0) / cp, T.at(0, 0) / cp);
  } else if (r20 < 0) {
    rpy[0] = std::atan2(T.at(0, 1), T.at(0, 2));
    rpy[1] = kPi / 2;
    rpy[2] = 0.0;
  } else {
    rpy[0] = std::atan2(-T.at(0, 1), -T.at(0, 2));
    rpy[1] = -kPi / 2;
    rpy[2] = 0.0;
  }
  return rpy;
}

Transform xyzrpy_to_trans(const XyzRpy& v) {
  Transform t = Transform::identity();
  const std::array<double, 9> R = rpy_to_rot({v[3], v[4], v[5]});
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) t.at(r, c) = R[static_cast<std::size_t>(r * 3 + c)];
    t.at(r, 3) = v[static_cast<std::size_t>(r)];
  }
  return t;
}

XyzRpy trans_to_xyzrpy(const Transform& T) {
  const Rpy rpy = rot_to_rpy(T);
  return {T.at(0, 3), T.at(1, 3), T.at(2, 3), rpy[0], rpy[1], rpy[2]};
}

double wrap_to_interval(double angle, double lower) {
  const double two_pi = 2.0 * kPi;
  double frac = std::fmod(angle - lower, two_pi);  // truncated; numpy's % is floored
  if (frac < 0.0) frac += two_pi;
  if (frac >= two_pi) frac = 0.0;  // core/utils.py's guard against rounding up to exactly 2pi
  return frac + lower;
}

double geodesic_distance(const Transform& t1, const Transform& t2) {
  // core/utils.py's geodesic_error then its norm: the translation difference, and the angle
  // of the relative rotation R1^T R2 from trace(R) = 1 + 2 cos(theta).
  double s = 0.0;
  for (int r = 0; r < 3; ++r) {
    const double d = t2.at(r, 3) - t1.at(r, 3);
    s += d * d;
  }
  double trace = 0.0;  // trace(R1^T R2) without forming the product
  for (int k = 0; k < 3; ++k) {
    for (int j = 0; j < 3; ++j) trace += t1.at(j, k) * t2.at(j, k);
  }
  double cos_angle = (trace - 1.0) / 2.0;
  // The clamp is load-bearing, not defensive: for two nearly equal rotations the trace can
  // land a rounding step above 3, and acos of 1 + 1e-16 is NaN, which would poison every
  // residual computed from it.
  cos_angle = cos_angle < -1.0 ? -1.0 : (cos_angle > 1.0 ? 1.0 : cos_angle);
  const double angle = std::acos(cos_angle);
  return std::sqrt(s + angle * angle);
}

std::optional<std::string> why_not_frame(const Transform& T, const std::string& name) {
  for (double v : T.m) {
    if (!std::isfinite(v)) return name + " must be a finite 4x4 transform";
  }
  const std::array<double, 4> last{0.0, 0.0, 0.0, 1.0};
  for (int c = 0; c < 4; ++c) {
    if (std::fabs(T.at(3, c) - last[static_cast<std::size_t>(c)]) > kFrameAtol) return name + " last row must be [0, 0, 0, 1]";
  }
  for (int r = 0; r < 3; ++r) {  // R R^T = I
    for (int c = 0; c < 3; ++c) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += T.at(r, k) * T.at(c, k);
      if (std::fabs(s - (r == c ? 1.0 : 0.0)) > kFrameAtol) {
        return name + " rotation block is not orthonormal within 1e-06";
      }
    }
  }
  const double det = T.at(0, 0) * (T.at(1, 1) * T.at(2, 2) - T.at(1, 2) * T.at(2, 1)) -
                     T.at(0, 1) * (T.at(1, 0) * T.at(2, 2) - T.at(1, 2) * T.at(2, 0)) +
                     T.at(0, 2) * (T.at(1, 0) * T.at(2, 1) - T.at(1, 1) * T.at(2, 0));
  if (std::fabs(det - 1.0) > kFrameAtol) return name + " rotation block must have determinant +1 (a reflection is not a rotation)";
  return std::nullopt;
}

}  // namespace sstsr
