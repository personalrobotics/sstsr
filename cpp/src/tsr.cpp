// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
#include "sstsr/tsr.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace sstsr {

namespace {
constexpr double kPi = std::numbers::pi;
constexpr double kGimbalEpsilon = 1e-14;

double norm6(const XyzRpy& v) {
  double s = 0.0;
  for (double x : v) s += x * x;
  return std::sqrt(s);
}
}  // namespace

TSR::TSR(Transform T0_w, Transform Tw_e, Bounds6 Bw) : T0_w_(T0_w), Tw_e_(Tw_e), Bw_(Bw) {
  if (auto why = why_not_frame(T0_w_, "T0_w")) throw std::invalid_argument(*why);
  if (auto why = why_not_frame(Tw_e_, "Tw_e")) throw std::invalid_argument(*why);
  for (const auto& row : Bw_.rows) {
    for (double v : row) {
      if (!std::isfinite(v)) throw std::invalid_argument("Bw must be finite");
    }
  }
  for (int i = 0; i < 3; ++i) {
    if (Bw_.lo(i) > Bw_.hi(i)) throw std::invalid_argument("Bw translation bounds must be [min, max]");
  }
  T0_w_inv_ = T0_w_.inverse_rigid();
  Tw_e_inv_ = Tw_e_.inverse_rigid();

  // Continuous bounds: outer intervals wrap, widths clamp to one turn, lower wrapped into [-pi, pi).
  cont_ = Bw_;
  for (int i = 3; i < 6; ++i) {
    double width = Bw_.hi(i) - Bw_.lo(i);
    if (width < 0.0) width += 2.0 * kPi;
    width = std::min(width, 2.0 * kPi);
    const double lo = wrap_to_interval(Bw_.lo(i), -kPi);
    cont_.rows[static_cast<std::size_t>(i)] = {lo, lo + width};
  }
}

Transform TSR::local(const Transform& T) const { return T0_w_inv_ * (T * Tw_e_inv_); }

// The round-off of composing a pose with T0_w and Tw_e scales with the lengths in that
// composition, not with the region's width (#190). Mirrors core/tsr.py's _length_atol.
double TSR::length_atol(const std::array<double, 3>& xyz) const {
  const auto norm3 = [](double a, double b, double c) { return std::sqrt(a * a + b * b + c * c); };
  const double scale = norm3(T0_w_.at(0, 3), T0_w_.at(1, 3), T0_w_.at(2, 3)) +
                       norm3(Tw_e_.at(0, 3), Tw_e_.at(1, 3), Tw_e_.at(2, 3)) + norm3(xyz[0], xyz[1], xyz[2]);
  return sstsr::length_atol(scale);
}

std::array<bool, 3> TSR::xyz_within(const std::array<double, 3>& xyz) const {
  const double atol = length_atol(xyz);
  std::array<bool, 3> out{};
  for (int i = 0; i < 3; ++i) {
    const double x = xyz[static_cast<std::size_t>(i)];
    out[static_cast<std::size_t>(i)] = (x + atol) >= cont_.lo(i) && (x - atol) <= cont_.hi(i);
  }
  return out;
}

std::array<bool, 3> TSR::rpy_within(const Rpy& rpy_in) const {
  std::array<bool, 3> out{};
  for (int k = 0; k < 3; ++k) {
    const int i = 3 + k;
    const double lo = cont_.lo(i), hi = cont_.hi(i);
    const double a = wrap_to_interval(rpy_in[static_cast<std::size_t>(k)], lo - kAngleAtol);
    if (lo > hi + kAngleAtol) {
      out[static_cast<std::size_t>(k)] = (a + kAngleAtol) >= lo || (a - kAngleAtol) <= hi;  // outer interval
    } else {
      out[static_cast<std::size_t>(k)] = (a + kAngleAtol) >= lo && (a - kAngleAtol) <= hi;
    }
  }
  return out;
}

std::pair<std::array<bool, 3>, std::optional<Rpy>> TSR::rot_within_rpy_bounds(const Transform& L) const {
  double r20 = L.at(2, 0);
  r20 = std::clamp(r20, -1.0, 1.0);
  std::array<bool, 3> check{};
  if (std::fabs(std::fabs(r20) - 1.0) >= kGimbalEpsilon) {
    const double psol = -std::asin(r20);
    for (double p : {psol, kPi - psol}) {
      const double cp = std::cos(p);
      const Rpy rpy{std::atan2(L.at(2, 1) / cp, L.at(2, 2) / cp), p, std::atan2(L.at(1, 0) / cp, L.at(0, 0) / cp)};
      check = rpy_within(rpy);
      if (check[0] && check[1] && check[2]) return {check, rpy};
    }
    return {check, std::nullopt};
  }
  const double lo_r = cont_.lo(3), hi_r = cont_.hi(3), lo_y = cont_.lo(5), hi_y = cont_.hi(5);
  std::array<Rpy, 4> corners{};
  if (r20 < 0) {
    const double off = std::atan2(L.at(0, 1), L.at(0, 2));
    corners = {Rpy{lo_y + off, kPi / 2, lo_y}, Rpy{hi_y + off, kPi / 2, hi_y}, Rpy{lo_r, kPi / 2, lo_r - off},
               Rpy{hi_r, kPi / 2, hi_r - off}};
  } else {
    const double off = std::atan2(-L.at(0, 1), -L.at(0, 2));
    corners = {Rpy{-lo_y + off, -kPi / 2, lo_y}, Rpy{-hi_y + off, -kPi / 2, hi_y}, Rpy{lo_r, -kPi / 2, -lo_r + off},
               Rpy{hi_r, -kPi / 2, -hi_r + off}};
  }
  for (const Rpy& rpy : corners) {
    check = rpy_within(rpy);
    if (!check[1]) return {check, std::nullopt};  // no point checking anything if +-pi/2 is not in the bounds
    if (check[0] && check[2]) return {check, rpy};
  }
  return {check, std::nullopt};
}

bool TSR::contains(const Transform& T) const {
  const Transform L = local(T);
  const std::array<bool, 3> xyz = xyz_within({L.at(0, 3), L.at(1, 3), L.at(2, 3)});
  const auto [rot, rpy] = rot_within_rpy_bounds(L);
  return xyz[0] && xyz[1] && xyz[2] && rot[0] && rot[1] && rot[2];
}

std::array<bool, 6> TSR::is_valid(const XyzRpy& v) const {
  const std::array<bool, 3> xyz = xyz_within({v[0], v[1], v[2]});
  const std::array<bool, 3> rpy = rpy_within({v[3], v[4], v[5]});
  return {xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]};
}

std::pair<XyzRpy, XyzRpy> TSR::displacement(const Transform& T) const {
  const Transform L = local(T);
  const std::array<double, 3> xyz{L.at(0, 3), L.at(1, 3), L.at(2, 3)};
  const Rpy rpy = rot_to_rpy(L);
  const double r = rpy[0], p = rpy[1], y = rpy[2];
  const std::array<Rpy, 9> candidates{Rpy{r, p, y},
                                      Rpy{r + kPi, kPi - p, y + kPi}, Rpy{r + kPi, kPi - p, y - kPi},
                                      Rpy{r - kPi, kPi - p, y + kPi}, Rpy{r - kPi, kPi - p, y - kPi},
                                      Rpy{r + kPi, -kPi - p, y + kPi}, Rpy{r + kPi, -kPi - p, y - kPi},
                                      Rpy{r - kPi, -kPi - p, y + kPi}, Rpy{r - kPi, -kPi - p, y - kPi}};
  XyzRpy best_dx{}, best_dw{};
  double best_norm = std::numeric_limits<double>::infinity();
  for (const Rpy& cand : candidates) {
    XyzRpy dw{xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0};
    // Wrap each angle to the representative NEAREST the bound interval, not into
    // [lo, lo + 2pi) (#186). An angle slightly below a bound narrower than a full turn is a
    // full turn away from `lo`, so wrapping from the lower edge sent it to almost lo + 2pi --
    // far above the upper bound -- and no equivalent-RPY candidate recovered the true small
    // displacement. The interval centre is the right origin because the distance to [lo, hi]
    // is max(0, |x - c| - w/2): monotone in |x - c|, so the representative nearest the centre
    // is the one nearest the interval.
    for (int k = 0; k < 3; ++k) {
      const double centre = (cont_.lo(3 + k) + cont_.hi(3 + k)) / 2.0;
      dw[static_cast<std::size_t>(3 + k)] = wrap_to_interval(cand[static_cast<std::size_t>(k)], centre - kPi);
    }
    XyzRpy dx{};
    for (int i = 0; i < 6; ++i) {
      const double v = dw[static_cast<std::size_t>(i)];
      if (v < cont_.lo(i)) dx[static_cast<std::size_t>(i)] = v - cont_.lo(i);
      else if (v > cont_.hi(i)) dx[static_cast<std::size_t>(i)] = v - cont_.hi(i);
    }
    const double n = norm6(dx);
    if (n < best_norm) {
      best_norm = n;
      best_dx = dx;
      best_dw = dw;
    }
  }
  return {best_dx, best_dw};
}

XyzRpy TSR::to_xyzrpy(const Transform& T) const {
  const Transform L = local(T);
  auto [check, rpy] = rot_within_rpy_bounds(L);
  const Rpy r = (check[0] && check[1] && check[2] && rpy) ? *rpy : rot_to_rpy(L);
  return {L.at(0, 3), L.at(1, 3), L.at(2, 3), r[0], r[1], r[2]};
}

std::pair<double, XyzRpy> TSR::distance_bwopt(const Transform& T) const {
  if (contains(T)) return {0.0, to_xyzrpy(T)};
  const auto [dx, dw] = displacement(T);
  XyzRpy bwopt{};
  for (int i = 0; i < 6; ++i) bwopt[static_cast<std::size_t>(i)] = std::clamp(dw[static_cast<std::size_t>(i)], cont_.lo(i), cont_.hi(i));
  for (int i = 3; i < 6; ++i) bwopt[static_cast<std::size_t>(i)] = wrap_to_interval(bwopt[static_cast<std::size_t>(i)], -kPi);
  return {norm6(dx), bwopt};
}

double TSR::distance(const Transform& T) const { return distance_bwopt(T).first; }

std::pair<double, Transform> TSR::closest_transform(const Transform& T) const {
  const auto [dist, bwopt] = distance_bwopt(T);
  return {dist, T0_w_ * (xyzrpy_to_trans(bwopt) * Tw_e_)};
}

XyzRpy TSR::sample_xyzrpy(Rng& rng) const {
  XyzRpy s{};
  for (int i = 0; i < 6; ++i) s[static_cast<std::size_t>(i)] = cont_.lo(i) + (cont_.hi(i) - cont_.lo(i)) * unit(rng);
  for (int i = 3; i < 6; ++i) s[static_cast<std::size_t>(i)] = wrap_to_interval(s[static_cast<std::size_t>(i)], -kPi);
  return s;
}

Transform TSR::sample(Rng& rng) const { return to_transform(sample_xyzrpy(rng)); }

Transform TSR::to_transform(const XyzRpy& v) const {
  const std::array<bool, 6> ok = is_valid(v);
  for (bool b : ok) {
    if (!b) throw std::invalid_argument("xyzrpy violates the TSR bounds");
  }
  return T0_w_ * (xyzrpy_to_trans(v) * Tw_e_);
}

double TSR::volume() const {
  // core/sampling.py's _interval_sum: raw widths, rotational widths clamped to one turn, negatives (outer
  // intervals) counted as zero. The legacy volume measure, not a geometric volume.
  double s = 0.0;
  for (int i = 0; i < 6; ++i) {
    double width = Bw_.hi(i) - Bw_.lo(i);
    if (i >= 3) width = std::min(width, 2.0 * kPi);
    s += std::max(width, 0.0);
  }
  return s;
}

}  // namespace sstsr
