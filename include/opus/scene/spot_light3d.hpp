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
class spot_light3d {
public:
  using numeric_type = Numeric;
  using vec3_type = math::vector3<numeric_type>;

  vec3_type position;
  vec3_type direction;
  vec3_type color;
  numeric_type intensity;
  numeric_type cutOff;      // Inner cutoff angle in radians
  numeric_type outerCutOff; // Outer cutoff angle in radians
  numeric_type constant;
  numeric_type linear;
  numeric_type quadratic;

  constexpr spot_light3d() noexcept
      : position(vec3_type::zero), direction(vec3_type(0.0, -1.0, 0.0)),
        color(vec3_type(1.0, 1.0, 1.0)), intensity(1.0),
        cutOff(std::cos(std::numbers::pi_v<numeric_type> / 6.0)), // 30 degrees
        outerCutOff(
            std::cos(std::numbers::pi_v<numeric_type> / 4.0)), // 45 degrees
        constant(1.0), linear(0.09), quadratic(0.032) {}

  constexpr spot_light3d(const vec3_type &pos, const vec3_type &dir,
                         const vec3_type &col, numeric_type intens,
                         numeric_type cutOffAngle,
                         numeric_type outerCutOffAngle, numeric_type constAtten,
                         numeric_type linAtten, numeric_type quadAtten) noexcept
      : position(pos), direction(dir.normalized()), color(col),
        intensity(intens), cutOff(std::cos(cutOffAngle)),
        outerCutOff(std::cos(outerCutOffAngle)), constant(constAtten),
        linear(linAtten), quadratic(quadAtten) {}

  constexpr spot_light3d &operator=(const spot_light3d &other) noexcept {
    position = other.position;
    direction = other.direction;
    color = other.color;
    intensity = other.intensity;
    cutOff = other.cutOff;
    outerCutOff = other.outerCutOff;
    constant = other.constant;
    linear = other.linear;
    quadratic = other.quadratic;
    return *this;
  }

  constexpr bool operator==(const spot_light3d &other) const noexcept {
    return position == other.position && direction == other.direction &&
           color == other.color && intensity == other.intensity &&
           cutOff == other.cutOff && outerCutOff == other.outerCutOff &&
           constant == other.constant && linear == other.linear &&
           quadratic == other.quadratic;
  }

  constexpr bool operator!=(const spot_light3d &other) const noexcept {
    return !(*this == other);
  }
};

} // namespace scene

template <std::floating_point Float>
struct std::formatter<scene::spot_light3d<Float>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const scene::spot_light3d<Float> &light,
              std::format_context &ctx) const {
    return std::format_to(
        ctx.out(),
        "Position: {}, Direction: {}, Color: {}, Intensity: {}, "
        "CutOff: {}, OuterCutOff: {}, Attenuation(constant: {}, linear: {}, "
        "quadratic: {})",
        light.position, light.direction, light.color, light.intensity,
        light.cutOff, light.outerCutOff, light.constant, light.linear,
        light.quadratic);
  }
};
