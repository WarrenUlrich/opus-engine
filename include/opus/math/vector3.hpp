#pragma once

#include <cmath>
#include <concepts>
#include <format>
#include <limits>
#include <numbers>

namespace math {

template <typename Numeric = float>
  requires std::is_floating_point_v<Numeric> || std::is_integral_v<Numeric>
class vector3 {
public:
  using numeric_type = Numeric;
  using vec_type = vector3<numeric_type>;

  constexpr static bool is_double_precision =
      std::is_same_v<numeric_type, double>;

  constexpr static vec_type zero = vec_type(0.0, 0.0, 0.0);

  constexpr static vec_type min =
      vec_type(std::numeric_limits<numeric_type>::min(),
               std::numeric_limits<numeric_type>::min(),
               std::numeric_limits<numeric_type>::min());

  constexpr static vec_type max =
      vec_type(std::numeric_limits<numeric_type>::max(),
               std::numeric_limits<numeric_type>::max(),
               std::numeric_limits<numeric_type>::max());

  constexpr static vec_type infinity =
      vec_type(std::numeric_limits<numeric_type>::infinity(),
               std::numeric_limits<numeric_type>::infinity(),
               std::numeric_limits<numeric_type>::infinity());

  numeric_type x;
  numeric_type y;
  numeric_type z;

  constexpr vector3() noexcept : x(0.0), y(0.0), z(0.0) {}

  constexpr vector3(numeric_type x_, numeric_type y_, numeric_type z_) noexcept
      : x(x_), y(y_), z(z_) {}

  constexpr vector3(const vec_type &other) noexcept
      : x(other.x), y(other.y), z(other.z) {}

  constexpr vec_type &operator=(const vec_type &other) noexcept {
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
  }

  // Comparison operators
  constexpr bool operator==(const vec_type &other) const noexcept {
    return x == other.x && y == other.y && z == other.z;
  }

  constexpr bool operator!=(const vec_type &other) const noexcept {
    return !(*this == other);
  }

  // Unary operators
  constexpr vec_type operator+() const noexcept { return {+x, +y, +z}; }

  constexpr vec_type operator-() const noexcept { return {-x, -y, -z}; }

  // Arithmetic operators
  constexpr vec_type operator+(const vec_type &other) const noexcept {
    return {x + other.x, y + other.y, z + other.z};
  }

  constexpr vec_type operator-(const vec_type &other) const noexcept {
    return {x - other.x, y - other.y, z - other.z};
  }

  constexpr vec_type operator*(numeric_type scalar) const noexcept {
    return {x * scalar, y * scalar, z * scalar};
  }

  constexpr vec_type operator/(numeric_type scalar) const noexcept {
    return {x / scalar, y / scalar, z / scalar};
  }

  // Compound assignment operators
  constexpr vec_type &operator+=(const vec_type &other) noexcept {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }

  constexpr vec_type &operator-=(const vec_type &other) noexcept {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }

  constexpr vec_type &operator*=(numeric_type scalar) noexcept {
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
  }

  constexpr vec_type &operator/=(numeric_type scalar) noexcept {
    x /= scalar;
    y /= scalar;
    z /= scalar;
    return *this;
  }

  // Additional member functions
  constexpr bool approx_equal(
      const vec_type &other,
      numeric_type epsilon =
          std::numeric_limits<numeric_type>::epsilon()) const noexcept {
    return std::abs(x - other.x) < epsilon && std::abs(y - other.y) < epsilon &&
           std::abs(z - other.z) < epsilon;
  }

  constexpr bool is_zero() const noexcept {
    constexpr numeric_type epsilon =
        std::numeric_limits<numeric_type>::epsilon();
    return std::abs(x) < epsilon && std::abs(y) < epsilon &&
           std::abs(z) < epsilon;
  }

  constexpr numeric_type length() const noexcept {
    return std::sqrt(x * x + y * y + z * z);
  }

  constexpr numeric_type length_squared() const noexcept {
    return x * x + y * y + z * z;
  }
  
  constexpr vec_type normalized() const noexcept {
    const numeric_type len = length();
    return (len == 0) ? *this : *this / len;
  }

  constexpr numeric_type dot(const vec_type &other) const noexcept {
    return x * other.x + y * other.y + z * other.z;
  }

  constexpr vec_type cross(const vec_type &other) const noexcept {
    return {y * other.z - z * other.y, z * other.x - x * other.z,
            x * other.y - y * other.x};
  }

  constexpr numeric_type distance(const vec_type &other) const noexcept {
    return std::sqrt((x - other.x) * (x - other.x) +
                     (y - other.y) * (y - other.y) +
                     (z - other.z) * (z - other.z));
  }

  constexpr numeric_type angle(const vec_type &other) const noexcept {
    const numeric_type dot_prod = dot(other);
    const numeric_type lengths = length() * other.length();
    return std::acos(dot_prod / lengths);
  }

  constexpr vec_type lerp(const vec_type &other,
                          numeric_type t) const noexcept {
    return *this + (other - *this) * t;
  }

  constexpr numeric_type angle_deg(const vec_type &other) const noexcept {
    return angle(other) *
           static_cast<numeric_type>(180.0 / std::numbers::pi_v<numeric_type>);
  }

  constexpr vec_type projection(const vec_type &other) const noexcept {
    const numeric_type scalar = dot(other) / other.dot(other);
    return other * scalar;
  }

  constexpr vec_type clamp(const numeric_type min,
                           const numeric_type max) const noexcept {
    return {std::clamp(x, min, max), std::clamp(y, min, max),
            std::clamp(z, min, max)};
  }

  constexpr vec_type reflect(const vec_type &normal) const noexcept {
    return *this - normal * (2 * dot(normal));
  }
};

// Non-member operator overloads for scalar * vector3
template <typename Numeric>
constexpr vector3<Numeric> operator*(Numeric scalar,
                                     const vector3<Numeric> &vec) noexcept {
  return vec * scalar;
}

} // namespace math

// Formatter specialization for std::format
template <std::floating_point Float>
struct std::formatter<math::vector3<Float>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const math::vector3<Float> &vec, std::format_context &ctx) const {
    return std::format_to(ctx.out(), "({}, {}, {})", vec.x, vec.y, vec.z);
  }
};
