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
class directional_light3d {
public:
  using numeric_type = Numeric;
  using vec3_type = math::vector3<numeric_type>;

  vec3_type direction;
  vec3_type color;
  numeric_type intensity;

  constexpr directional_light3d() noexcept
      : direction(vec3_type(0.0, -1.0, 0.0)), color(vec3_type(1.0, 1.0, 1.0)),
        intensity(1.0) {}

  constexpr directional_light3d(const vec3_type &dir, const vec3_type &col,
                                numeric_type intens) noexcept
      : direction(dir.normalized()), color(col), intensity(intens) {}

  constexpr directional_light3d &
  operator=(const directional_light3d &other) noexcept {
    direction = other.direction;
    color = other.color;
    intensity = other.intensity;
    return *this;
  }

  constexpr bool operator==(const directional_light3d &other) const noexcept {
    return direction == other.direction && color == other.color &&
           intensity == other.intensity;
  }

  constexpr bool operator!=(const directional_light3d &other) const noexcept {
    return !(*this == other);
  }
};

} // namespace scene

template <std::floating_point Float>
struct std::formatter<scene::directional_light3d<Float>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const scene::directional_light3d<Float> &light,
              std::format_context &ctx) const {
    return std::format_to(ctx.out(), "Direction: {}, Color: {}, Intensity: {}",
                          light.direction, light.color, light.intensity);
  }
};
