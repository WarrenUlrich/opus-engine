#pragma once

#include <cmath>
#include <concepts>
#include <format>
#include <limits>
#include <numbers>

#include "vector3.hpp"
#include "matrix4x4.hpp"

namespace math {
/**
 * @class quaternion
 * @brief A mathematical representation of a quaternion, used in 3D rotations
 * and orientation.
 *
 * Quaternions provide an efficient way to handle rotations in 3D space without
 * the problems of gimbal lock that arise with Euler angles. They are also
 * computationally more efficient than matrices for concatenating rotations.
 *
 * This class supports basic quaternion operations, such as multiplication,
 * normalization, conjugation, and spherical linear interpolation (SLERP). It is
 * designed for use in applications like computer graphics, physics simulations,
 * and robotics.
 *
 * @tparam Numeric The numeric type used (e.g., float or double). Supports both
 * floating-point and integral types, but floating-point types (float, double)
 * are recommended.
 */
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

  /**
   * @brief Checks if two quaternions are approximately equal, within a
   * tolerance.
   * @param other The quaternion to compare.
   * @param epsilon The tolerance value.
   * @return True if the quaternions are approximately equal, false otherwise.
   */
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

  /**
   * @brief Rotates a 3D vector using the quaternion's rotation.
   *
   * This function applies the rotation represented by the quaternion to a given
   * 3D vector. It is equivalent to rotating the vector in 3D space according to
   * the quaternion's orientation. This is particularly useful in 3D graphics to
   * rotate objects or camera directions.
   *
   * @param v The 3D vector to rotate.
   * @return A new 3D vector, rotated by the quaternion.
   */
  constexpr vec3_type rotate(const vec3_type &v) const noexcept {
    quat_type q_v(0, v.x, v.y, v.z);
    quat_type result = (*this) * q_v * inverse();
    return {result.x, result.y, result.z};
  }

  /**
   * @brief Creates a quaternion that represents a rotation around a given axis
   * by a specific angle.
   *
   * This function takes a 3D unit vector (axis) and an angle (in radians) as
   * input and produces a quaternion that rotates around that axis by the given
   * angle. It is commonly used to construct rotations for animation,
   * orientation, or object manipulation in 3D graphics. The axis should be
   * normalized (i.e., have a length of 1) for correct behavior.
   *
   * @param axis A normalized 3D vector representing the rotation axis.
   * @param angle_rad The rotation angle in radians (e.g., π/2 for 90 degrees).
   * @return A quaternion representing the specified rotation.
   *
   * @note If the axis is not normalized, the resulting rotation may not behave
   * as expected.
   */
  static constexpr quat_type from_axis_angle(const vec3_type &axis,
                                             numeric_type angle_rad) noexcept {
    numeric_type half_angle = angle_rad / 2;
    numeric_type s = std::sin(half_angle);
    return {std::cos(half_angle), axis.x * s, axis.y * s, axis.z * s};
  }

  /**
   * @brief Converts the quaternion to an axis-angle representation.
   *
   * This function extracts the axis of rotation and the angle (in radians) that
   * the quaternion represents. The quaternion should ideally be normalized for
   * meaningful results. This conversion is useful for situations where you need
   * to express rotations in a more intuitive way (e.g., rotating an object
   * around a particular axis by a specific amount).
   *
   * @param axis Output parameter that will store the rotation axis (as a 3D
   * vector).
   * @param angle_rad Output parameter that will store the rotation angle in
   * radians.
   *
   * @note If the quaternion represents no rotation (identity quaternion), the
   * axis will default to (1, 0, 0).
   */
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
  
  /**
   * @brief Performs spherical linear interpolation (SLERP) between two
   * quaternions.
   *
   * SLERP smoothly interpolates between two rotations (represented by
   * quaternions) along the shortest path on a 4D hypersphere. This is often
   * used in animations or smooth transitions between orientations.
   *
   * @param other The target quaternion to interpolate towards.
   * @param t A value between 0.0 and 1.0 indicating the interpolation factor
   *          (0.0 = start quaternion, 1.0 = target quaternion).
   * @return The interpolated quaternion at the given interpolation factor.
   *
   * @note If the quaternions are very close to each other, a linear
   * interpolation might be performed instead to avoid precision issues.
   */
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