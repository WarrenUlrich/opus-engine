#pragma once

#include "../math/mat4.hpp"

#include <numbers>

namespace scene {

struct camera {
	float fov_y_rad{60.0f * (std::numbers::pi_v<float> / 180.0f)};
	float aspect_ratio{1.333f};
	float near_z{0.01f};
	float far_z{100.0f};

	[[nodiscard]] inline math::mat4 projection_matrix() const noexcept {
		return math::mat4::perspective(fov_y_rad, aspect_ratio, near_z, far_z);
	}
};

} // namespace scene