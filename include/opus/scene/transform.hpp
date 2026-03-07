#pragma once

#include "../math/mat4.hpp"
#include "../math/quat.hpp"
#include "../math/vec3.hpp"

namespace scene {
class transform {
public:
	math::vec3 position{0.0f, 0.0f, 0.0f};
	math::quat rotation{};
	math::vec3 scale{1.0f, 1.0f, 1.0f};

	// T * R * S
	[[nodiscard]] constexpr math::mat4 matrix() const noexcept {
		return math::mat4::translate(position) * rotation.to_mat4() * math::mat4::scale(scale);
	}
};
} // namespace scene