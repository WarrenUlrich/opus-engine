#pragma once

#include "../math/mat4.hpp"
#include "../math/vec3.hpp"
#include "transform.hpp"

#include <numbers>

namespace scene {

class camera {
public:
	// Camera direction state. Sync position from ECS transform component.

	// We retain forward/up as directional vectors because extracting
	// these from a quaternion (local_transform.rotation) requires math
	// you might not have implemented in your math library yet.
	math::vec3 forward{0.0f, 0.0f, -1.0f};
	math::vec3 up{0.0f, 1.0f, 0.0f};

	// Lens State
	float fov_y_rad{60.0f * (std::numbers::pi_v<float> / 180.0f)};
	float aspect_ratio{1.333f};
	float near_z{0.01f};
	float far_z{100.0f};

	constexpr camera() noexcept = default;

	// Forces the camera to stare at a specific point in space.
	inline void look_at(const math::vec3 &eye, const math::vec3 &target) noexcept {
		forward = (target - eye).normalized();
	}

	[[nodiscard]] inline math::mat4 projection_matrix() const noexcept {
		return math::mat4::perspective(fov_y_rad, aspect_ratio, near_z, far_z);
	}


};

} // namespace scene