#pragma once

#include <cmath>
#include <concepts>
#include <format>
#include <limits>
#include <numbers>

#include "vector3.hpp"

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

  static constexpr quat_type from_axis_angle(const vec3_type &axis,
                                             numeric_type angle) noexcept {
    // Ensure the axis is normalized
    vec3_type normalized_axis = axis.normalized();

    // Calculate half angle (quaternions use half angles for rotation)
    numeric_type half_angle = angle * 0.5f;

    // Calculate sine and cosine of half angle
    numeric_type sin_half = std::sin(half_angle);
    numeric_type cos_half = std::cos(half_angle);

    // Create quaternion (w, x, y, z)
    // w = cos(θ/2)
    // x = axis.x * sin(θ/2)
    // y = axis.y * sin(θ/2)
    // z = axis.z * sin(θ/2)
    return quat_type(cos_half,                     // w component (real part)
                     normalized_axis.x * sin_half, // x component
                     normalized_axis.y * sin_half, // y component
                     normalized_axis.z * sin_half  // z component
    );
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

  /**
   * @brief Creates a quaternion that represents the rotation required to look
   * from one point to another.
   *
   * The look_at function generates a quaternion that, when applied to an
   * object, will orient it to face from the 'eye' position toward the 'target'
   * position. The 'up' vector defines which direction is considered "up" in the
   * resulting orientation.
   *
   * This is particularly useful for:
   * - Camera positioning in 3D applications
   * - Making objects face toward points of interest
   * - Orienting game characters to look at targets
   *
   * @param eye The position from which we are looking (typically the object or
   * camera position).
   * @param target The position we want to look at.
   * @param up The up direction vector (typically (0,1,0) for y-up coordinate
   * systems).
   * @return A quaternion representing the orientation to look from 'eye' toward
   * 'target'.
   */
  static constexpr quat_type
  look_at(const vec3_type &eye, const vec3_type &target,
          const vec3_type &world_up = vec3_type(0, 1, 0)) noexcept {
    // Calculate forward vector (z-axis direction)
    vec3_type forward = (eye - target).normalized();

    // Check for degenerate case
    if (forward.length_squared() <
        std::numeric_limits<numeric_type>::epsilon()) {
      return identity;
    }

    // Calculate right vector (x-axis direction)
    vec3_type right = world_up.cross(forward).normalized();

    // Check if forward and up are parallel
    if (right.length_squared() < std::numeric_limits<numeric_type>::epsilon()) {
      // Choose a different up vector
      vec3_type alternate_up = (std::abs(forward.y) < 0.9f)
                                   ? vec3_type(0, 1, 0)
                                   : vec3_type(1, 0, 0);
      right = alternate_up.cross(forward).normalized();
    }

    // Calculate actual up vector (y-axis direction)
    vec3_type up = forward.cross(right);

    // Find rotation between two coordinate frames
    // This is more direct than going through a matrix

    // First, create a quaternion that rotates from world forward (0,0,1) to our
    // forward
    vec3_type world_forward(0, 0, 1);
    vec3_type rotation_axis = world_forward.cross(forward);
    numeric_type dot = world_forward.dot(forward);

    quat_type q1;

    // Handle special cases
    if (rotation_axis.length_squared() <
        std::numeric_limits<numeric_type>::epsilon()) {
      // Vectors are parallel
      if (dot > 0) {
        // Same direction
        q1 = identity;
      } else {
        // Opposite direction
        q1 = quat_type(0, 0, 1, 0); // 180 degree rotation around y-axis
      }
    } else {
      // Normal case - create quaternion from axis and angle
      numeric_type angle =
          std::acos(std::clamp(dot, numeric_type(-1), numeric_type(1)));
      q1 = from_axis_angle(rotation_axis.normalized(), angle);
    }

    // Then adjust for roll (rotation around forward axis)
    // We need to align the "up" direction
    vec3_type rotated_up = q1.rotate(vec3_type(0, 1, 0));
    vec3_type target_up = up;

    rotation_axis = rotated_up.cross(target_up);
    dot = rotated_up.dot(target_up);

    quat_type q2;

    if (rotation_axis.length_squared() <
        std::numeric_limits<numeric_type>::epsilon()) {
      // Vectors are parallel
      if (dot > 0) {
        q2 = identity;
      } else {
        q2 = from_axis_angle(forward, std::numbers::pi_v<numeric_type>);
      }
    } else {
      numeric_type angle =
          std::acos(std::clamp(dot, numeric_type(-1), numeric_type(1)));
      q2 = from_axis_angle(forward, angle);
    }

    // Combine rotations (q2 first, then q1)
    return q1 * q2;
  }
};

} // namespace math

template <typename Numeric>
  requires std::is_floating_point_v<Numeric> || std::is_integral_v<Numeric>
struct std::formatter<math::quaternion<Numeric>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }
  auto format(const math::quaternion<Numeric> &quat,
              std::format_context &ctx) const {
    return std::format_to(ctx.out(), "({}, {}, {}, {})", quat.w, quat.x, quat.y,
                          quat.z);
  }
};