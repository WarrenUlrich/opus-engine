#pragma once

#include <cmath>
#include <concepts>
#include <format>
#include <limits>
#include <numbers>

#include "../math/math.hpp"

namespace scene {

template <typename Numeric = float>
  requires std::is_floating_point_v<Numeric> || std::is_integral_v<Numeric>
class transform3d {
public:
  using numeric_type = Numeric;
  using vec3_type = math::vector3<numeric_type>;
  using quat_type = math::quaternion<numeric_type>;
  using mat4_type = math::matrix4x4<numeric_type>;
  using transform_type = transform3d<numeric_type>;

  vec3_type position;
  quat_type rotation;
  vec3_type scale;

  constexpr transform3d() noexcept
      : position(vec3_type::zero), rotation(quat_type::identity),
        scale(vec3_type(1.0, 1.0, 1.0)) {}

  constexpr transform3d(const vec3_type &pos, const quat_type &rot,
                        const vec3_type &scl) noexcept
      : position(pos), rotation(rot), scale(scl) {}

  constexpr transform_type &operator=(const transform_type &other) noexcept {
    position = other.position;
    rotation = other.rotation;
    scale = other.scale;
    return *this;
  }

  constexpr bool operator==(const transform_type &other) const noexcept {
    return position == other.position && rotation == other.rotation &&
           scale == other.scale;
  }

  constexpr bool operator!=(const transform_type &other) const noexcept {
    return !(*this == other);
  }

  constexpr transform_type
  operator*(const transform_type &other) const noexcept {
    vec3_type new_position = position + rotation.rotate(other.position * scale);
    quat_type new_rotation = rotation * other.rotation;
    vec3_type new_scale = scale * other.scale;
    return {new_position, new_rotation, new_scale};
  }

  constexpr transform_type &operator*=(const transform_type &other) noexcept {
    *this = *this * other;
    return *this;
  }

  constexpr mat4_type to_matrix() const noexcept {
    mat4_type scale_matrix = mat4_type::scaling(scale);
    mat4_type rotation_matrix = mat4_type::from_quaternion(rotation);
    mat4_type translation_matrix = mat4_type::translation(position);
    return translation_matrix * rotation_matrix * scale_matrix;
  }

  constexpr vec3_type transform_point(const vec3_type &point) const noexcept {
    return position + rotation.rotate(point * scale);
  }

  constexpr vec3_type
  transform_direction(const vec3_type &direction) const noexcept {
    return rotation.rotate(direction);
  }

  constexpr transform_type inverse() const noexcept {
    quat_type inv_rotation = rotation.inverse();
    vec3_type inv_scale = vec3_type(1.0, 1.0, 1.0) / scale;
    vec3_type inv_position = inv_rotation.rotate(-position * inv_scale);
    return {inv_position, inv_rotation, inv_scale};
  }
};

} // namespace scene

template <std::floating_point Float>
struct std::formatter<scene::transform3d<Float>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const scene::transform3d<Float> &transform,
              std::format_context &ctx) const {
    return std::format_to(ctx.out(), "Position: {}, Rotation: {}, Scale: {}",
                          transform.position, transform.rotation,
                          transform.scale);
  }
};
