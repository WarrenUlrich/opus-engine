#pragma once

#include "../math/vec3.hpp"

#include "sokol_gfx.h"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <span>
#include <vector>

namespace scene {

struct mesh_instance {
	struct vertex {
		math::vec3 position;
		math::vec3 normal;
	};

	sg_bindings bind{};
	int index_count{0};

	mesh_instance() = default;

	mesh_instance(std::span<const vertex> vertices, std::span<const uint16_t> indices) noexcept {
		sg_buffer_desc vbuf_desc = {};
		vbuf_desc.usage.vertex_buffer = true;
		vbuf_desc.data = {vertices.data(), vertices.size_bytes()};
		bind.vertex_buffers[0] = sg_make_buffer(&vbuf_desc);

		sg_buffer_desc ibuf_desc = {};
		ibuf_desc.usage.index_buffer = true;
		ibuf_desc.data = {indices.data(), indices.size_bytes()};
		bind.index_buffer = sg_make_buffer(&ibuf_desc);

		index_count = static_cast<int>(indices.size());
	}

	[[nodiscard]] static mesh_instance sphere(float radius, int sectors, int rings) {
		std::vector<vertex> vertices;
		std::vector<uint16_t> indices;
		vertices.reserve((rings + 1) * (sectors + 1));
		indices.reserve(rings * sectors * 6);

		constexpr float PI = std::numbers::pi_v<float>;

		for (int r = 0; r <= rings; ++r) {
			float phi = (float)r / rings * PI;
			for (int s = 0; s <= sectors; ++s) {
				float theta = (float)s / sectors * 2.0f * PI;
				float x = std::cos(theta) * std::sin(phi);
				float y = std::cos(phi);
				float z = std::sin(theta) * std::sin(phi);

				math::vec3 normal = math::vec3(x, y, z).normalized();
				vertices.push_back({normal * radius, normal});
			}
		}

		for (int r = 0; r < rings; ++r) {
			for (int s = 0; s < sectors; ++s) {
				uint16_t current = r * (sectors + 1) + s;
				uint16_t next = current + sectors + 1;

				indices.insert(indices.end(),
				               {current, next, static_cast<uint16_t>(current + 1),
				                static_cast<uint16_t>(current + 1), next, static_cast<uint16_t>(next + 1)});
			}
		}
		return mesh_instance(vertices, indices);
	}

	[[nodiscard]] static mesh_instance plane(float width, float depth, int width_segments = 1,
	                                         int depth_segments = 1) {
		std::vector<vertex> vertices;
		std::vector<uint16_t> indices;
		vertices.reserve((width_segments + 1) * (depth_segments + 1));
		indices.reserve(width_segments * depth_segments * 6);

		float hw = width * 0.5f;
		float hd = depth * 0.5f;
		math::vec3 normal = math::vec3(0.0f, 1.0f, 0.0f);

		for (int z = 0; z <= depth_segments; ++z) {
			float normalized_z = (float)z / depth_segments;
			float pz = -hd + normalized_z * depth;
			for (int x = 0; x <= width_segments; ++x) {
				float normalized_x = (float)x / width_segments;
				float px = -hw + normalized_x * width;
				vertices.push_back({math::vec3(px, 0.0f, pz), normal});
			}
		}

		for (int z = 0; z < depth_segments; ++z) {
			for (int x = 0; x < width_segments; ++x) {
				uint16_t current = z * (width_segments + 1) + x;
				uint16_t next = current + width_segments + 1;

				indices.insert(indices.end(),
				               {current, static_cast<uint16_t>(current + 1), next,
				                static_cast<uint16_t>(current + 1), static_cast<uint16_t>(next + 1), next});
			}
		}
		return mesh_instance(vertices, indices);
	}

	[[nodiscard]] static mesh_instance cube(float width, float height, float depth) {
		std::vector<vertex> vertices;
		std::vector<uint16_t> indices;
		vertices.reserve(24);
		indices.reserve(36);

		float hw = width * 0.5f;
		float hh = height * 0.5f;
		float hd = depth * 0.5f;

		auto add_face = [&](math::vec3 n, math::vec3 bl, math::vec3 br, math::vec3 tr, math::vec3 tl) {
			uint16_t i = static_cast<uint16_t>(vertices.size());
			vertices.push_back({bl, n});
			vertices.push_back({br, n});
			vertices.push_back({tr, n});
			vertices.push_back({tl, n});
			indices.insert(indices.end(),
			               {i, static_cast<uint16_t>(i + 2), static_cast<uint16_t>(i + 1), i,
			                static_cast<uint16_t>(i + 3), static_cast<uint16_t>(i + 2)});
		};

		add_face(math::vec3(0, 0, 1), math::vec3(-hw, -hh, hd), math::vec3(hw, -hh, hd),
		         math::vec3(hw, hh, hd), math::vec3(-hw, hh, hd)); // Front
		add_face(math::vec3(0, 0, -1), math::vec3(hw, -hh, -hd), math::vec3(-hw, -hh, -hd),
		         math::vec3(-hw, hh, -hd), math::vec3(hw, hh, -hd)); // Back
		add_face(math::vec3(1, 0, 0), math::vec3(hw, -hh, hd), math::vec3(hw, -hh, -hd),
		         math::vec3(hw, hh, -hd), math::vec3(hw, hh, hd)); // Right
		add_face(math::vec3(-1, 0, 0), math::vec3(-hw, -hh, -hd), math::vec3(-hw, -hh, hd),
		         math::vec3(-hw, hh, hd), math::vec3(-hw, hh, -hd)); // Left
		add_face(math::vec3(0, 1, 0), math::vec3(-hw, hh, hd), math::vec3(hw, hh, hd),
		         math::vec3(hw, hh, -hd), math::vec3(-hw, hh, -hd)); // Top
		add_face(math::vec3(0, -1, 0), math::vec3(-hw, -hh, -hd), math::vec3(hw, -hh, -hd),
		         math::vec3(hw, -hh, hd), math::vec3(-hw, -hh, hd)); // Bottom

		return mesh_instance(vertices, indices);
	}

	[[nodiscard]] static mesh_instance torus(float radius, float tube_radius, int radial_segments,
	                                         int tubular_segments) {
		std::vector<vertex> vertices;
		std::vector<uint16_t> indices;
		vertices.reserve((radial_segments + 1) * (tubular_segments + 1));
		indices.reserve(radial_segments * tubular_segments * 6);

		constexpr float PI2 = std::numbers::pi_v<float> * 2.0f;

		for (int r = 0; r <= radial_segments; ++r) {
			float u = (float)r / radial_segments * PI2;
			math::vec3 center = math::vec3(radius * std::cos(u), 0.0f, radius * std::sin(u));

			for (int t = 0; t <= tubular_segments; ++t) {
				float v = (float)t / tubular_segments * PI2;

				float cx = std::cos(v) * tube_radius;
				float cy = std::sin(v) * tube_radius;

				float px = center.x + cx * std::cos(u);
				float py = cy;
				float pz = center.z + cx * std::sin(u);

				math::vec3 pos = math::vec3(px, py, pz);
				math::vec3 normal =
				    math::vec3(pos.x - center.x, pos.y - center.y, pos.z - center.z).normalized();

				vertices.push_back({pos, normal});
			}
		}

		for (int r = 0; r < radial_segments; ++r) {
			for (int t = 0; t < tubular_segments; ++t) {
				uint16_t current = r * (tubular_segments + 1) + t;
				uint16_t next = current + tubular_segments + 1;

				indices.insert(indices.end(),
				               {current, next, static_cast<uint16_t>(current + 1),
				                static_cast<uint16_t>(current + 1), next, static_cast<uint16_t>(next + 1)});
			}
		}
		return mesh_instance(vertices, indices);
	}

	// -- construct from obj_data ---------------------------------------------

	[[nodiscard]] static mesh_instance from_obj(const auto &obj) noexcept {
		return mesh_instance(
		    std::span<const vertex>{
		        reinterpret_cast<const vertex *>(obj.vertices.data()),
		        obj.vertices.size()},
		    std::span<const uint16_t>{obj.indices.data(), obj.indices.size()});
	}

	void destroy() noexcept {
		sg_destroy_buffer(bind.vertex_buffers[0]);
		sg_destroy_buffer(bind.index_buffer);
	}
};

} // namespace scene