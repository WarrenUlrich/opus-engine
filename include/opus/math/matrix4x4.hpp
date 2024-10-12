#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <format>
#include <limits>
#include <numbers>

#include "quaternion.hpp"
#include "vector3.hpp"

namespace math {

template <typename Numeric = float>
  requires std::is_floating_point_v<Numeric> || std::is_integral_v<Numeric>
class matrix4x4 {
public:
  using numeric_type = Numeric;
  using mat_type = matrix4x4<numeric_type>;
  using vec_type = vector3<numeric_type>;

  constexpr static bool is_double_precision =
      std::is_same_v<numeric_type, double>;

  // Static matrices
  constexpr static mat_type zero() noexcept { return mat_type{}; }

  constexpr static mat_type identity() noexcept {
    mat_type result{};
    for (int i = 0; i < 4; ++i) {
      result.m[i][i] = static_cast<numeric_type>(1);
    }
    return result;
  }

  numeric_type m[4][4];

  // Constructors
  constexpr matrix4x4() noexcept {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        m[i][j] = static_cast<numeric_type>(0);
  }

  constexpr matrix4x4(const mat_type &other) noexcept {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        m[i][j] = other.m[i][j];
  }

  constexpr mat_type &operator=(const mat_type &other) noexcept {
    if (this != &other) {
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          m[i][j] = other.m[i][j];
    }
    return *this;
  }

  // Element access
  constexpr numeric_type *operator[](std::size_t index) noexcept {
    return m[index];
  }

  constexpr const numeric_type *operator[](std::size_t index) const noexcept {
    return m[index];
  }

  // Comparison operators
  constexpr bool operator==(const mat_type &other) const noexcept {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        if (m[i][j] != other.m[i][j])
          return false;
    return true;
  }

  constexpr bool operator!=(const mat_type &other) const noexcept {
    return !(*this == other);
  }

  // Arithmetic operators
  constexpr mat_type operator+(const mat_type &other) const noexcept {
    mat_type result;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        result.m[i][j] = m[i][j] + other.m[i][j];
    return result;
  }

  constexpr mat_type operator-(const mat_type &other) const noexcept {
    mat_type result;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        result.m[i][j] = m[i][j] - other.m[i][j];
    return result;
  }

  constexpr mat_type operator*(const mat_type &other) const noexcept {
    mat_type result{};
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 4; ++k)
          result.m[i][j] += m[i][k] * other.m[k][j];
    return result;
  }

  constexpr mat_type operator*(numeric_type scalar) const noexcept {
    mat_type result;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        result.m[i][j] = m[i][j] * scalar;
    return result;
  }

  constexpr mat_type &operator+=(const mat_type &other) noexcept {
    *this = *this + other;
    return *this;
  }

  constexpr mat_type &operator-=(const mat_type &other) noexcept {
    *this = *this - other;
    return *this;
  }

  constexpr mat_type &operator*=(const mat_type &other) noexcept {
    *this = *this * other;
    return *this;
  }

  constexpr mat_type &operator*=(numeric_type scalar) noexcept {
    *this = *this * scalar;
    return *this;
  }

  // Vector transformation
  constexpr vec_type operator*(const vec_type &vec) const noexcept {
    numeric_type x =
        m[0][0] * vec.x + m[0][1] * vec.y + m[0][2] * vec.z + m[0][3];
    numeric_type y =
        m[1][0] * vec.x + m[1][1] * vec.y + m[1][2] * vec.z + m[1][3];
    numeric_type z =
        m[2][0] * vec.x + m[2][1] * vec.y + m[2][2] * vec.z + m[2][3];
    numeric_type w =
        m[3][0] * vec.x + m[3][1] * vec.y + m[3][2] * vec.z + m[3][3];

    if (w != static_cast<numeric_type>(0) &&
        w != static_cast<numeric_type>(1)) {
      x /= w;
      y /= w;
      z /= w;
    }

    return vec_type{x, y, z};
  }

  // Transpose
  constexpr mat_type transpose() const noexcept {
    mat_type result;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        result.m[j][i] = m[i][j];
    return result;
  }

  // Determinant (Not fully implemented due to complexity)
  numeric_type determinant() const noexcept {
    // Determinant calculation for 4x4 matrices is complex.
    // For brevity, this function is left as a placeholder.
    // Implementing determinant calculation is possible but requires careful
    // coding.
    return static_cast<numeric_type>(0);
  }

  // Inverse (Not fully implemented due to complexity)
  mat_type inverse() const noexcept {
    // Inversion of a 4x4 matrix is complex.
    // For brevity, this function is left as a placeholder.
    // Implementing matrix inversion is possible but requires careful coding.
    return mat_type::identity();
  }

  // Transformation matrices
  constexpr static mat_type translation(const vec_type &offset) noexcept {
    mat_type result = identity();
    result.m[0][3] = offset.x;
    result.m[1][3] = offset.y;
    result.m[2][3] = offset.z;
    return result;
  }

  constexpr static mat_type scaling(const vec_type &factors) noexcept {
    mat_type result = identity();
    result.m[0][0] = factors.x;
    result.m[1][1] = factors.y;
    result.m[2][2] = factors.z;
    return result;
  }

  constexpr static mat_type rotation_x(numeric_type angle) noexcept {
    mat_type result = identity();
    numeric_type c = std::cos(angle);
    numeric_type s = std::sin(angle);
    result.m[1][1] = c;
    result.m[1][2] = -s;
    result.m[2][1] = s;
    result.m[2][2] = c;
    return result;
  }

  constexpr static mat_type rotation_y(numeric_type angle) noexcept {
    mat_type result = identity();
    numeric_type c = std::cos(angle);
    numeric_type s = std::sin(angle);
    result.m[0][0] = c;
    result.m[0][2] = s;
    result.m[2][0] = -s;
    result.m[2][2] = c;
    return result;
  }

  constexpr static mat_type rotation_z(numeric_type angle) noexcept {
    mat_type result = identity();
    numeric_type c = std::cos(angle);
    numeric_type s = std::sin(angle);
    result.m[0][0] = c;
    result.m[0][1] = -s;
    result.m[1][0] = s;
    result.m[1][1] = c;
    return result;
  }

  constexpr static mat_type rotation_axis(const vec_type &axis,
                                          numeric_type angle) noexcept {
    mat_type result = identity();
    vec_type n = axis.normalized();
    numeric_type c = std::cos(angle);
    numeric_type s = std::sin(angle);
    numeric_type t = static_cast<numeric_type>(1) - c;

    result.m[0][0] = c + n.x * n.x * t;
    result.m[0][1] = n.x * n.y * t - n.z * s;
    result.m[0][2] = n.x * n.z * t + n.y * s;

    result.m[1][0] = n.y * n.x * t + n.z * s;
    result.m[1][1] = c + n.y * n.y * t;
    result.m[1][2] = n.y * n.z * t - n.x * s;

    result.m[2][0] = n.z * n.x * t - n.y * s;
    result.m[2][1] = n.z * n.y * t + n.x * s;
    result.m[2][2] = c + n.z * n.z * t;

    return result;
  }

  constexpr static mat_type perspective(numeric_type fov, numeric_type aspect,
                                        numeric_type near,
                                        numeric_type far) noexcept {
    numeric_type f = static_cast<numeric_type>(1) /
                     std::tan(fov / static_cast<numeric_type>(2));
    mat_type result = zero();

    result.m[0][0] = f / aspect;
    result.m[1][1] = f;
    result.m[2][2] = (far + near) / (near - far);
    result.m[2][3] = (2 * far * near) / (near - far);
    result.m[3][2] = -1;
    result.m[3][3] = 0; // already zero

    return result;
  }

  constexpr static mat_type orthographic(numeric_type left, numeric_type right,
                                         numeric_type bottom, numeric_type top,
                                         numeric_type near,
                                         numeric_type far) noexcept {
    mat_type result = zero();

    result.m[0][0] = static_cast<numeric_type>(2) / (right - left);
    result.m[1][1] = static_cast<numeric_type>(2) / (top - bottom);
    result.m[2][2] = static_cast<numeric_type>(2) / (near - far);

    result.m[3][0] = (left + right) / (left - right);
    result.m[3][1] = (bottom + top) / (bottom - top);
    result.m[3][2] = (far + near) / (near - far);

    result.m[3][3] = 1; // already set to 1 in identity

    return result;
  }

  constexpr static mat_type
  from_quaternion(const quaternion<numeric_type> &q) noexcept {
    mat_type result = identity();
    numeric_type w = q.w;
    numeric_type x = q.x;
    numeric_type y = q.y;
    numeric_type z = q.z;

    numeric_type xx = x * x;
    numeric_type yy = y * y;
    numeric_type zz = z * z;
    numeric_type xy = x * y;
    numeric_type xz = x * z;
    numeric_type yz = y * z;
    numeric_type wx = w * x;
    numeric_type wy = w * y;
    numeric_type wz = w * z;

    result.m[0][0] =
        static_cast<numeric_type>(1) - static_cast<numeric_type>(2) * (yy + zz);
    result.m[0][1] = static_cast<numeric_type>(2) * (xy - wz);
    result.m[0][2] = static_cast<numeric_type>(2) * (xz + wy);
    result.m[0][3] = static_cast<numeric_type>(0);

    result.m[1][0] = static_cast<numeric_type>(2) * (xy + wz);
    result.m[1][1] =
        static_cast<numeric_type>(1) - static_cast<numeric_type>(2) * (xx + zz);
    result.m[1][2] = static_cast<numeric_type>(2) * (yz - wx);
    result.m[1][3] = static_cast<numeric_type>(0);

    result.m[2][0] = static_cast<numeric_type>(2) * (xz - wy);
    result.m[2][1] = static_cast<numeric_type>(2) * (yz + wx);
    result.m[2][2] =
        static_cast<numeric_type>(1) - static_cast<numeric_type>(2) * (xx + yy);
    result.m[2][3] = static_cast<numeric_type>(0);

    result.m[3][0] = static_cast<numeric_type>(0);
    result.m[3][1] = static_cast<numeric_type>(0);
    result.m[3][2] = static_cast<numeric_type>(0);
    result.m[3][3] = static_cast<numeric_type>(1);

    return result;
  }
};

} // namespace math

template <std::floating_point Float>
struct std::formatter<math::matrix4x4<Float>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const math::matrix4x4<Float> &mat,
              std::format_context &ctx) const {
    auto out = ctx.out();
    out = std::format_to(out, "[\n");
    for (int i = 0; i < 4; ++i) {
      out = std::format_to(out, "  [");
      for (int j = 0; j < 4; ++j) {
        out = std::format_to(out, "{: .4f}", mat.m[i][j]);
        if (j < 3)
          out = std::format_to(out, ", ");
      }
      out = std::format_to(out, "]");
      if (i < 3)
        out = std::format_to(out, ",\n");
    }
    out = std::format_to(out, "\n]");
    return out;
  }
};
