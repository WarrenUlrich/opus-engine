#pragma once

#include "../math/vector3.hpp"
#include <format>

namespace scene3d {

template <typename Numeric = float>
struct light {
  using numeric_type = Numeric;
  using vec3_type = math::vector3<numeric_type>;

  enum class type_t { point = 0, directional = 1, spot = 2 };

  type_t type{type_t::point};

  // Common properties
  vec3_type color{1.0f, 1.0f, 1.0f};
  numeric_type intensity{1.0f};

  // For point and spot lights
  numeric_type range{10.0f};

  // For spot lights (angles in radians)
  numeric_type inner_cone_angle{0.261799f}; // 15 degrees in radians
  numeric_type outer_cone_angle{0.523599f}; // 30 degrees in radians

  constexpr bool operator==(const light &other) const noexcept {
    return type == other.type && color == other.color &&
           intensity == other.intensity && range == other.range &&
           inner_cone_angle == other.inner_cone_angle &&
           outer_cone_angle == other.outer_cone_angle;
  }

  constexpr bool operator!=(const light &other) const noexcept {
    return !(*this == other);
  }

  // Factory methods for different light types
  static light create_point_light(
      const vec3_type &color = {1.0f, 1.0f, 1.0f},
      numeric_type intensity = 1.0f, numeric_type range = 10.0f) {
    return {type_t::point, color, intensity, range};
  }

  static light create_directional_light(
      const vec3_type &color = {1.0f, 1.0f, 1.0f},
      numeric_type intensity = 1.0f) {
    return {type_t::directional, color, intensity};
  }

  static light create_spot_light(
      const vec3_type &color = {1.0f, 1.0f, 1.0f},
      numeric_type intensity = 1.0f, numeric_type range = 10.0f,
      numeric_type inner_cone_angle = 0.261799f, // 15 degrees in radians
      numeric_type outer_cone_angle = 0.523599f  // 30 degrees in radians
      ) {
    return {type_t::spot, color, intensity, range, inner_cone_angle,
            outer_cone_angle};
  }
};

} // namespace scene3d

// Formatter for std::format support
template <typename Float>
struct std::formatter<scene3d::light<Float>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const scene3d::light<Float> &light,
              std::format_context &ctx) const {
    const char *type_str = "";
    switch (light.type) {
    case scene3d::light<Float>::type_t::directional:
      type_str = "Directional";
      break;
    case scene3d::light<Float>::type_t::point:
      type_str = "Point";
      break;
    case scene3d::light<Float>::type_t::spot:
      type_str = "Spot";
      break;
    default:
      type_str = "Unknown";
      break;
    }

    return std::format_to(
        ctx.out(),
        "Light(Type: {}, Color: {}, Intensity: {}, Range: {}, Inner Cone "
        "Angle: {}, Outer Cone Angle: {})",
        type_str, light.color, light.intensity, light.range,
        light.inner_cone_angle, light.outer_cone_angle);
  }
};
