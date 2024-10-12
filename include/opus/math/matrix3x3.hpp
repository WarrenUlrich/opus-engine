#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <format>
#include <limits>
#include <numbers>

#include "vector3.hpp"

namespace math {

template <typename Numeric = float>
  requires std::is_floating_point_v<Numeric> || std::is_integral_v<Numeric>
class matrix3x3 {
public:
  using numeric_type = Numeric;
  using mat_type = matrix3x3<numeric_type>;
  using vec_type = vector3<numeric_type>;

  constexpr static bool is_double_precision =
      std::is_same_v<numeric_type, double>;

  // Static matrices
  constexpr static mat_type zero() noexcept { return mat_type{}; }

  constexpr static mat_type identity() noexcept {
    mat_type result{};
    for (int i = 0; i < 3; ++i) {
      result.m[i][i] = static_cast<numeric_type>(1);
    }
    return result;
  }

  numeric_type m[3][3];

  // Constructors
  constexpr matrix3x3() noexcept {
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        m[i][j] = static_cast<numeric_type>(0);
  }

  constexpr matrix3x3(const mat_type &other) noexcept {
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        m[i][j] = other.m[i][j];
  }

  constexpr mat_type &operator=(const mat_type &other) noexcept {
    if (this != &other) {
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
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
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
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
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        result.m[i][j] = m[i][j] + other.m[i][j];
    return result;
  }

  constexpr mat_type operator-(const mat_type &other) const noexcept {
    mat_type result;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        result.m[i][j] = m[i][j] - other.m[i][j];
    return result;
  }

  constexpr mat_type operator*(const mat_type &other) const noexcept {
    mat_type result{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
          result.m[i][j] += m[i][k] * other.m[k][j];
    return result;
  }

  constexpr mat_type operator*(numeric_type scalar) const noexcept {
    mat_type result;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
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
    numeric_type x = m[0][0] * vec.x + m[0][1] * vec.y + m[0][2] * vec.z;
    numeric_type y = m[1][0] * vec.x + m[1][1] * vec.y + m[1][2] * vec.z;
    numeric_type z = m[2][0] * vec.x + m[2][1] * vec.y + m[2][2] * vec.z;
    return vec_type{x, y, z};
  }

  // Transpose
  constexpr mat_type transpose() const noexcept {
    mat_type result;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        result.m[j][i] = m[i][j];
    return result;
  }

  // Determinant
  constexpr numeric_type determinant() const noexcept {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  }

  // Inverse
  constexpr mat_type inverse() const noexcept {
    mat_type result{};
    numeric_type det = determinant();
    if (det == static_cast<numeric_type>(0)) {
      return zero(); // Singular matrix
    }

    numeric_type inv_det = static_cast<numeric_type>(1) / det;
    result.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    result.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    result.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;

    result.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    result.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    result.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;

    result.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    result.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    result.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;

    return result;
  }
};

} // namespace math
