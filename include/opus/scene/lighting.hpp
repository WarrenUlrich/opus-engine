#pragma once

#include "../math/vec3.hpp"

#include <cmath>

namespace scene {

/// Maximum lights per draw call. Must match the GPU shader constant.
static constexpr int MAX_LIGHTS = 8;

// ─── Light Types ─────────────────────────────────────────────────────────────

enum class light_type : int {
	directional = 0, // Infinite distance, direction only (sun, moon)
	point = 1,       // Omnidirectional with falloff (lightbulb)
	spot = 2,        // Cone-shaped with falloff (flashlight, stage light)
};

// ─── Light ───────────────────────────────────────────────────────────────────
//
// High-level light description with expressive factory methods.
// Cone angles are stored as cosines — GPU-ready, no trig in the shader.

class light {
public:
	light_type type{light_type::directional};
	math::vec3 position{0.0f, 0.0f, 0.0f};
	math::vec3 direction{0.0f, -1.0f, 0.0f};
	math::vec3 color{1.0f, 1.0f, 1.0f};
	float intensity{1.0f};
	float range{10.0f};
	float inner_cone_cos{0.9063f}; // cos(~25°)
	float outer_cone_cos{0.8192f}; // cos(~35°)

	constexpr light() noexcept = default;

	/// Infinite directional light — sun, moon, skybox key.
	[[nodiscard]] static constexpr light directional(const math::vec3 &dir,
	                                                 const math::vec3 &col = {1.0f, 1.0f, 1.0f},
	                                                 float intensity = 1.0f) noexcept {
		light l;
		l.type = light_type::directional;
		l.direction = dir;
		l.color = col;
		l.intensity = intensity;
		return l;
	}

	/// Omnidirectional point light with smooth inverse-square falloff.
	[[nodiscard]] static constexpr light point(const math::vec3 &pos,
	                                           const math::vec3 &col = {1.0f, 1.0f, 1.0f},
	                                           float intensity = 1.0f,
	                                           float range = 10.0f) noexcept {
		light l;
		l.type = light_type::point;
		l.position = pos;
		l.color = col;
		l.intensity = intensity;
		l.range = range;
		return l;
	}

	/// Spot light with smooth cone falloff.
	/// inner/outer angles are in radians and converted to cosines internally.
	[[nodiscard]] static inline light spot(const math::vec3 &pos, const math::vec3 &dir,
	                                       const math::vec3 &col = {1.0f, 1.0f, 1.0f},
	                                       float intensity = 1.0f, float range = 10.0f,
	                                       float inner_angle_rad = 0.4363f,
	                                       float outer_angle_rad = 0.6109f) noexcept {
		light l;
		l.type = light_type::spot;
		l.position = pos;
		l.direction = dir;
		l.color = col;
		l.intensity = intensity;
		l.range = range;
		l.inner_cone_cos = std::cos(inner_angle_rad);
		l.outer_cone_cos = std::cos(outer_angle_rad);
		return l;
	}
};

// ─── PBR Material ────────────────────────────────────────────────────────────
//
// Metallic-roughness workflow (glTF 2.0 / Disney BRDF standard).
// All values are linear-space; gamma correction happens in the shader.

class material {
public:
	math::vec3 albedo{0.5f, 0.5f, 0.5f}; // Base color (linear)
	float alpha{1.0f};                     // Opacity
	float metallic{0.0f};                  // 0 = dielectric, 1 = metal
	float roughness{0.5f};                 // 0 = mirror, 1 = diffuse
	float ao{1.0f};                        // Ambient occlusion

	constexpr material() noexcept = default;

	// ── Preset Materials ──

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

// ─── Lighting Environment ────────────────────────────────────────────────────
//
// Manages a fixed-capacity light array + ambient term + material.
// Packs everything into a GPU-ready struct with a single pack() call.
// The resulting gpu_data maps byte-for-byte to the GLSL uniform block.
//
// Memory layout:  SoA (Struct-of-Arrays) for light properties.
//                 Each vec4 array is contiguous — optimal for GPU cache lines.
// Total gpu_data: 576 bytes = 4 header vec4s + 4×MAX_LIGHTS light vec4s.

class lighting_environment {
public:
	math::vec3 ambient{0.03f, 0.03f, 0.03f};

	constexpr lighting_environment() noexcept = default;

	// ── Light Management ──

	constexpr void add(const light &l) noexcept {
		if (count_ < MAX_LIGHTS)
			lights_[count_++] = l;
	}

	constexpr void clear() noexcept { count_ = 0; }

	[[nodiscard]] constexpr int count() const noexcept { return count_; }

	[[nodiscard]] constexpr light &operator[](int i) noexcept { return lights_[i]; }

	[[nodiscard]] constexpr const light &operator[](int i) const noexcept { return lights_[i]; }

	// ── GPU Upload ──
	//
	// Tightly packed vec4 arrays — maps 1:1 to sequential GLSL uniforms.
	// Use with: sg_apply_uniforms(block_index, SG_RANGE(gpu_data_instance));

	struct gpu_data {
		float material[4];                          // {metallic, roughness, ao, 0}
		float albedo[4];                            // {r, g, b, alpha}
		float camera_pos[4];                        // {x, y, z, num_lights}
		float ambient_color[4];                     // {r, g, b, 0}
		float light_pos_type[MAX_LIGHTS][4];        // {x, y, z, type}
		float light_dir_range[MAX_LIGHTS][4];       // {dx, dy, dz, range}
		float light_color_intensity[MAX_LIGHTS][4]; // {r, g, b, intensity}
		float light_params[MAX_LIGHTS][4];          // {inner_cos, outer_cos, 0, 0}
	};

	/// Packs camera, material, ambient, and all active lights into one GPU block.
	[[nodiscard]] gpu_data pack(const math::vec3 &camera_pos,
	                            const material &mat) const noexcept {
		gpu_data d{};

		// Material
		d.material[0] = mat.metallic;
		d.material[1] = mat.roughness;
		d.material[2] = mat.ao;

		d.albedo[0] = mat.albedo.x;
		d.albedo[1] = mat.albedo.y;
		d.albedo[2] = mat.albedo.z;
		d.albedo[3] = mat.alpha;

		// Camera + light count
		d.camera_pos[0] = camera_pos.x;
		d.camera_pos[1] = camera_pos.y;
		d.camera_pos[2] = camera_pos.z;
		d.camera_pos[3] = static_cast<float>(count_);

		// Ambient
		d.ambient_color[0] = ambient.x;
		d.ambient_color[1] = ambient.y;
		d.ambient_color[2] = ambient.z;

		// Per-light SoA packing
		for (int i = 0; i < count_; ++i) {
			const auto &l = lights_[i];

			d.light_pos_type[i][0] = l.position.x;
			d.light_pos_type[i][1] = l.position.y;
			d.light_pos_type[i][2] = l.position.z;
			d.light_pos_type[i][3] = static_cast<float>(static_cast<int>(l.type));

			d.light_dir_range[i][0] = l.direction.x;
			d.light_dir_range[i][1] = l.direction.y;
			d.light_dir_range[i][2] = l.direction.z;
			d.light_dir_range[i][3] = l.range;

			d.light_color_intensity[i][0] = l.color.x;
			d.light_color_intensity[i][1] = l.color.y;
			d.light_color_intensity[i][2] = l.color.z;
			d.light_color_intensity[i][3] = l.intensity;

			d.light_params[i][0] = l.inner_cone_cos;
			d.light_params[i][1] = l.outer_cone_cos;
		}

		return d;
	}

private:
	light lights_[MAX_LIGHTS]{};
	int count_{0};
};

} // namespace scene
