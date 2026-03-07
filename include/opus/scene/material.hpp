#pragma once

#include "../math/vec3.hpp"

namespace scene {

// Metallic-roughness PBR material (glTF 2.0 / Disney BRDF).
// All values linear-space; gamma correction happens in the shader.
class material {
public:
	math::vec3 albedo{0.5f, 0.5f, 0.5f};
	float alpha{1.0f};
	float metallic{0.0f};
	float roughness{0.5f};
	float ao{1.0f};

	constexpr material() noexcept = default;

	[[nodiscard]] static constexpr material gold() noexcept {
		material m;
		m.albedo = {1.0f, 0.765f, 0.336f};
		m.metallic = 1.0f;
		m.roughness = 0.3f;
		return m;
	}

	[[nodiscard]] static constexpr material copper() noexcept {
		material m;
		m.albedo = {0.955f, 0.638f, 0.538f};
		m.metallic = 1.0f;
		m.roughness = 0.25f;
		return m;
	}

	[[nodiscard]] static constexpr material silver() noexcept {
		material m;
		m.albedo = {0.972f, 0.960f, 0.915f};
		m.metallic = 1.0f;
		m.roughness = 0.2f;
		return m;
	}

	[[nodiscard]] static constexpr material plastic(const math::vec3 &tint = {0.8f, 0.1f,
	                                                                          0.1f}) noexcept {
		material m;
		m.albedo = tint;
		m.metallic = 0.0f;
		m.roughness = 0.45f;
		return m;
	}

	[[nodiscard]] static constexpr material ceramic(const math::vec3 &tint = {0.95f, 0.95f,
	                                                                          0.92f}) noexcept {
		material m;
		m.albedo = tint;
		m.metallic = 0.0f;
		m.roughness = 0.1f;
		return m;
	}
};

} // namespace scene
