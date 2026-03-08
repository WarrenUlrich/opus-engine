#pragma once

#include "../math/vec3.hpp"

namespace scene {

// Metallic-roughness PBR material (glTF 2.0 / Disney BRDF).
// All values linear-space; gamma correction happens in the shader.
struct material {
	math::vec3 albedo{0.5f, 0.5f, 0.5f};
	float alpha{1.0f};
	float metallic{0.0f};
	float roughness{0.5f};
	float ao{1.0f};

	[[nodiscard]] static constexpr material gold() noexcept {
		return {.albedo = {1.0f, 0.765f, 0.336f}, .metallic = 1.0f, .roughness = 0.3f};
	}
	[[nodiscard]] static constexpr material copper() noexcept {
		return {.albedo = {0.955f, 0.638f, 0.538f}, .metallic = 1.0f, .roughness = 0.25f};
	}
	[[nodiscard]] static constexpr material silver() noexcept {
		return {.albedo = {0.972f, 0.960f, 0.915f}, .metallic = 1.0f, .roughness = 0.2f};
	}
	[[nodiscard]] static constexpr material plastic(const math::vec3 &tint = {0.8f, 0.1f, 0.1f}) noexcept {
		return {.albedo = tint, .roughness = 0.45f};
	}
	[[nodiscard]] static constexpr material ceramic(const math::vec3 &tint = {0.95f, 0.95f, 0.92f}) noexcept {
		return {.albedo = tint, .roughness = 0.1f};
	}
};

} // namespace scene
