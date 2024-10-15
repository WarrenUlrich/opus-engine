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
class point_light3d {
public:
  using numeric_type = Numeric;
  using vec3_type = math::vector3<numeric_type>;

  vec3_type position;
  vec3_type color;
  numeric_type intensity;
  numeric_type constant;
  numeric_type linear;
  numeric_type quadratic;

  constexpr point_light3d() noexcept
      : position(vec3_type::zero), color(vec3_type(1.0, 1.0, 1.0)),
        intensity(1.0), constant(1.0), linear(0.09), quadratic(0.032) {}

  constexpr point_light3d(const vec3_type &pos, const vec3_type &col,
                          numeric_type intens, numeric_type constAtten,
                          numeric_type linAtten,
                          numeric_type quadAtten) noexcept
      : position(pos), color(col), intensity(intens), constant(constAtten),
        linear(linAtten), quadratic(quadAtten) {}

  constexpr point_light3d &operator=(const point_light3d &other) noexcept {
    position = other.position;
    color = other.color;
    intensity = other.intensity;
    constant = other.constant;
    linear = other.linear;
    quadratic = other.quadratic;
    return *this;
  }

  constexpr bool operator==(const point_light3d &other) const noexcept {
    return position == other.position && color == other.color &&
           intensity == other.intensity && constant == other.constant &&
           linear == other.linear && quadratic == other.quadratic;
  }

  constexpr bool operator!=(const point_light3d &other) const noexcept {
    return !(*this == other);
  }
};

} // namespace scene

template <std::floating_point Float>
struct std::formatter<scene::point_light3d<Float>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const scene::point_light3d<Float> &light,
              std::format_context &ctx) const {
    return std::format_to(
        ctx.out(),
        "Position: {}, Color: {}, Intensity: {}, "
        "Attenuation(constant: {}, linear: {}, quadratic: {})",
        light.position, light.color, light.intensity, light.constant,
        light.linear, light.quadratic);
  }
};
