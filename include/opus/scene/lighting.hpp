#pragma once

#include "../math/vec3.hpp"

#include <cmath>

namespace scene {

enum class light_type : int {
	directional = 0,
	point = 1,
	spot = 2,
};

struct light {
	light_type type{light_type::directional};
	math::vec3 position{};
	math::vec3 direction{0.0f, -1.0f, 0.0f};
	math::vec3 color{1.0f, 1.0f, 1.0f};
	float intensity{1.0f};
	float range{10.0f};
	float inner_cone_cos{0.9063f}; // cos(~25deg)
	float outer_cone_cos{0.8192f}; // cos(~35deg)

	[[nodiscard]] static constexpr light directional(const math::vec3 &dir,
	                                                 const math::vec3 &col = {1.0f, 1.0f, 1.0f},
	                                                 float intensity = 1.0f) noexcept {
		return {.direction = dir, .color = col, .intensity = intensity};
	}
	[[nodiscard]] static constexpr light point(const math::vec3 &pos,
	                                           const math::vec3 &col = {1.0f, 1.0f, 1.0f},
	                                           float intensity = 1.0f, float range = 10.0f) noexcept {
		return {.type = light_type::point, .position = pos, .color = col, .intensity = intensity, .range = range};
	}
	// Cone angles are radians, stored as cosines (GPU-ready).
	[[nodiscard]] static inline light spot(const math::vec3 &pos, const math::vec3 &dir,
	                                       const math::vec3 &col = {1.0f, 1.0f, 1.0f},
	                                       float intensity = 1.0f, float range = 10.0f,
	                                       float inner_angle_rad = 0.4363f,
	                                       float outer_angle_rad = 0.6109f) noexcept {
		return {.type = light_type::spot, .position = pos, .direction = dir, .color = col,
		        .intensity = intensity, .range = range,
		        .inner_cone_cos = std::cos(inner_angle_rad), .outer_cone_cos = std::cos(outer_angle_rad)};
	}
};

} // namespace scene
