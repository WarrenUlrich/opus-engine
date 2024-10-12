#pragma once

#include <cmath>
#include <concepts>
#include <format>
#include <limits>
#include <numbers>

namespace math {
template <typename Numeric = float>
  requires std::is_floating_point_v<Numeric> || std::is_integral_v<Numeric>
class quaternion {
public:
  using numeric_type = Numeric;
  using quat_type = quaternion<numeric_type>;
  using vec3_type = vector3<numeric_type>;

  constexpr static bool is_double_precision =
      std::is_same_v<numeric_type, double>;

  constexpr static quat_type identity = quat_type(1.0, 0.0, 0.0, 0.0);

  numeric_type w;
  numeric_type x;
  numeric_type y;
  numeric_type z;

  constexpr quaternion() noexcept : w(1.0), x(0.0), y(0.0), z(0.0) {}

  constexpr quaternion(numeric_type w_, numeric_type x_, numeric_type y_,
                       numeric_type z_) noexcept
      : w(w_), x(x_), y(y_), z(z_) {}

  constexpr quaternion(const quat_type &other) noexcept
      : w(other.w), x(other.x), y(other.y), z(other.z) {}

  constexpr quat_type &operator=(const quat_type &other) noexcept {
    w = other.w;
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
  }

  constexpr bool operator==(const quat_type &other) const noexcept {
    return w == other.w && x == other.x && y == other.y && z == other.z;
  }

  constexpr bool operator!=(const quat_type &other) const noexcept {
    return !(*this == other);
  }

  constexpr quat_type operator+() const noexcept { return {+w, +x, +y, +z}; }

  constexpr quat_type operator+(const quat_type &other) const noexcept {
    return {w + other.w, x + other.x, y + other.y, z + other.z};
  }

  constexpr quat_type operator-() const noexcept { return {-w, -x, -y, -z}; }

  constexpr quat_type operator-(const quat_type &other) const noexcept {
    return {w - other.w, x - other.x, y - other.y, z - other.z};
  }

  constexpr quat_type operator*(const quat_type &other) const noexcept {
    return {w * other.w - x * other.x - y * other.y - z * other.z,
            w * other.x + x * other.w + y * other.z - z * other.y,
            w * other.y - x * other.z + y * other.w + z * other.x,
            w * other.z + x * other.y - y * other.x + z * other.w};
  }

  constexpr quat_type operator*(numeric_type scalar) const noexcept {
    return {w * scalar, x * scalar, y * scalar, z * scalar};
  }

  constexpr quat_type operator/(numeric_type scalar) const noexcept {
    return {w / scalar, x / scalar, y / scalar, z / scalar};
  }

  constexpr quat_type &operator*=(const quat_type &other) noexcept {
    *this = *this * other;
    return *this;
  }

  constexpr quat_type &operator*=(numeric_type scalar) noexcept {
    w *= scalar;
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
  }

  constexpr quat_type &operator/=(numeric_type scalar) noexcept {
    w /= scalar;
    x /= scalar;
    y /= scalar;
    z /= scalar;
    return *this;
  }

  constexpr bool approx_equal(
      const quat_type &other,
      numeric_type epsilon =
          std::numeric_limits<numeric_type>::epsilon()) const noexcept {
    return std::abs(w - other.w) < epsilon && std::abs(x - other.x) < epsilon &&
           std::abs(y - other.y) < epsilon && std::abs(z - other.z) < epsilon;
  }

  constexpr numeric_type dot(const quat_type &other) const noexcept {
    return w * other.w + x * other.x + y * other.y + z * other.z;
  }

  constexpr numeric_type norm() const noexcept {
    return w * w + x * x + y * y + z * z;
  }

  constexpr numeric_type length() const noexcept { return std::sqrt(norm()); }

  constexpr quat_type normalized() const noexcept {
    const numeric_type len = length();
    return (len == 0) ? *this : *this / len;
  }

  constexpr quat_type conjugate() const noexcept { return {w, -x, -y, -z}; }

  constexpr quat_type inverse() const noexcept { return conjugate() / norm(); }

  constexpr vec3_type rotate(const vec3_type &v) const noexcept {
    quat_type q_v(0, v.x, v.y, v.z);
    quat_type result = (*this) * q_v * inverse();
    return {result.x, result.y, result.z};
  }

  static constexpr quat_type from_axis_angle(const vec3_type &axis,
                                             numeric_type angle_rad) noexcept {
    numeric_type half_angle = angle_rad / 2;
    numeric_type s = std::sin(half_angle);
    return {std::cos(half_angle), axis.x * s, axis.y * s, axis.z * s};
  }

  constexpr void to_axis_angle(vec3_type &axis,
                               numeric_type &angle_rad) const noexcept {
    quat_type q = normalized();
    angle_rad = 2 * std::acos(q.w);
    numeric_type s = std::sqrt(1 - q.w * q.w);
    if (s < std::numeric_limits<numeric_type>::epsilon()) {
      axis = {1, 0, 0};
    } else {
      axis = {q.x / s, q.y / s, q.z / s};
    }
  }

  constexpr quat_type slerp(const quat_type &other,
                            numeric_type t) const noexcept {
    numeric_type dot_product = dot(other);
    quat_type end = other;

    if (dot_product < 0.0f) {
      dot_product = -dot_product;
      end = -other;
    }

    if (dot_product > 0.9995f) {
      return ((*this) + t * (end - *this)).normalized();
    }

    numeric_type theta_0 = std::acos(dot_product);
    numeric_type theta = theta_0 * t;
    numeric_type sin_theta = std::sin(theta);
    numeric_type sin_theta_0 = std::sin(theta_0);

    numeric_type s0 = std::cos(theta) - dot_product * sin_theta / sin_theta_0;
    numeric_type s1 = sin_theta / sin_theta_0;

    return (*this) * s0 + end * s1;
  }
};

} // namespace math