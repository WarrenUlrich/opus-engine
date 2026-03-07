#pragma once

#include "../math/mat4.hpp"

#include <vector>

namespace scene {

class debug_grid {
public:
	debug_grid() = default;

	void init(int half_extent = 10, float spacing = 1.0f) {
		const char *vs_src = R"(
			#version 330
			layout(location=0) in vec3 position;
			layout(location=1) in vec4 color;
			uniform mat4 mvp;
			out vec4 v_color;
			void main() {
				v_color     = color;
				gl_Position = mvp * vec4(position, 1.0);
			}
		)";

		const char *fs_src = R"(
			#version 330
			in vec4 v_color;
			out vec4 frag_color;
			void main() {
				frag_color = v_color;
			}
		)";

		sg_shader_desc shd = {};
		shd.vertex_func.source = vs_src;
		shd.fragment_func.source = fs_src;
		shd.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
		shd.uniform_blocks[0].size = sizeof(math::mat4);
		shd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_MAT4, 0, "mvp"};

		sg_pipeline_desc pip = {};
		pip.shader = sg_make_shader(&shd);
		pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3;
		pip.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT4;
		pip.primitive_type = SG_PRIMITIVETYPE_LINES;
		pip.depth.write_enabled = true;
		pip.depth.compare = SG_COMPAREFUNC_LESS_EQUAL;
		pip.colors[0].blend.enabled = true;
		pip.colors[0].blend.src_factor_rgb = SG_BLENDFACTOR_SRC_ALPHA;
		pip.colors[0].blend.dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
		pip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
		pip.depth.pixel_format = SG_PIXELFORMAT_DEPTH;
		pip.sample_count = 1;
		pipeline_ = sg_make_pipeline(&pip);

		build_geometry(half_extent, spacing);
	}

	void draw(const math::mat4 &vp) const {
		sg_apply_pipeline(pipeline_);
		sg_apply_bindings(&bind_);
		sg_apply_uniforms(0, SG_RANGE(vp));
		sg_draw(0, vertex_count_, 1);
	}

	void destroy() {
		sg_destroy_buffer(bind_.vertex_buffers[0]);
		sg_destroy_pipeline(pipeline_);
	}

private:
	struct color_vertex {
		float pos[3];
		float col[4];
	};

	sg_pipeline pipeline_{};
	sg_bindings bind_{};
	int vertex_count_{0};

	void build_geometry(int half, float sp) {
		std::vector<color_vertex> verts;
		verts.reserve(static_cast<size_t>((2 * half + 1) * 4 + 6));

		const float extent = half * sp;
		const float gc[4] = {0.35f, 0.35f, 0.35f, 0.45f};

		// Grid lines parallel to X (varying Z)
		for (int i = -half; i <= half; ++i) {
			float z = i * sp;
			if (i == 0)
				continue; // axes draw over origin
			verts.push_back({{-extent, 0.0f, z}, {gc[0], gc[1], gc[2], gc[3]}});
			verts.push_back({{extent, 0.0f, z}, {gc[0], gc[1], gc[2], gc[3]}});
		}

		// Grid lines parallel to Z (varying X)
		for (int i = -half; i <= half; ++i) {
			float x = i * sp;
			if (i == 0)
				continue;
			verts.push_back({{x, 0.0f, -extent}, {gc[0], gc[1], gc[2], gc[3]}});
			verts.push_back({{x, 0.0f, extent}, {gc[0], gc[1], gc[2], gc[3]}});
		}

		// Axis lines: offset slightly above grid to avoid z-fighting
		const float y = 0.001f;
		const float axis_len = extent;

		// X axis (red, negative half dimmed)
		verts.push_back({{-axis_len, y, 0.0f}, {0.8f, 0.2f, 0.2f, 0.5f}});
		verts.push_back({{0.0f, y, 0.0f}, {0.8f, 0.2f, 0.2f, 0.5f}});
		verts.push_back({{0.0f, y, 0.0f}, {1.0f, 0.2f, 0.2f, 1.0f}});
		verts.push_back({{axis_len, y, 0.0f}, {1.0f, 0.2f, 0.2f, 1.0f}});

		// Y axis (green, negative half dimmed)
		verts.push_back({{0.0f, -axis_len, 0.0f}, {0.2f, 0.8f, 0.2f, 0.4f}});
		verts.push_back({{0.0f, 0.0f, 0.0f}, {0.2f, 0.8f, 0.2f, 0.4f}});
		verts.push_back({{0.0f, 0.0f, 0.0f}, {0.2f, 1.0f, 0.2f, 1.0f}});
		verts.push_back({{0.0f, axis_len, 0.0f}, {0.2f, 1.0f, 0.2f, 1.0f}});

		// Z axis (blue, negative half dimmed)
		verts.push_back({{0.0f, y, -axis_len}, {0.2f, 0.2f, 0.8f, 0.5f}});
		verts.push_back({{0.0f, y, 0.0f}, {0.2f, 0.2f, 0.8f, 0.5f}});
		verts.push_back({{0.0f, y, 0.0f}, {0.2f, 0.2f, 1.0f, 1.0f}});
		verts.push_back({{0.0f, y, axis_len}, {0.2f, 0.2f, 1.0f, 1.0f}});

		vertex_count_ = static_cast<int>(verts.size());

		sg_buffer_desc buf = {};
		buf.data = {verts.data(), verts.size() * sizeof(color_vertex)};
		bind_.vertex_buffers[0] = sg_make_buffer(&buf);
	}
};

} // namespace scene
