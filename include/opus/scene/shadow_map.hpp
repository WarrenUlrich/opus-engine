#pragma once

#include "../math/mat4.hpp"
#include "../math/vec3.hpp"

namespace scene {

inline constexpr int SHADOW_MAP_DEFAULT_RES = 2048;

// Directional-light shadow map.
// Renders the scene from the sun's perspective into an R32F colour
// attachment (storing gl_FragCoord.z), which the deferred light pass
// samples to evaluate PCF-filtered shadows.

class shadow_map {
public:
	shadow_map() = default;

	void init(int resolution = SHADOW_MAP_DEFAULT_RES) {
		resolution_ = resolution;

		// R32F colour image – stores light-space depth
		{
			sg_image_desc d = {};
			d.usage.color_attachment = true;
			d.width = resolution;
			d.height = resolution;
			d.pixel_format = SG_PIXELFORMAT_R32F;
			d.sample_count = 1;
			d.label = "shadow_color";
			color_img_ = sg_make_image(&d);
		}

		// Depth image for correct depth testing during the shadow pass
		{
			sg_image_desc d = {};
			d.usage.depth_stencil_attachment = true;
			d.width = resolution;
			d.height = resolution;
			d.pixel_format = SG_PIXELFORMAT_DEPTH;
			d.sample_count = 1;
			d.label = "shadow_depth";
			depth_img_ = sg_make_image(&d);
		}

		// Colour-attachment view (used in the offscreen pass)
		{
			sg_view_desc vd = {};
			vd.color_attachment.image = color_img_;
			color_att_view_ = sg_make_view(&vd);
		}

		// Depth-stencil-attachment view (used in the offscreen pass)
		{
			sg_view_desc vd = {};
			vd.depth_stencil_attachment.image = depth_img_;
			depth_att_view_ = sg_make_view(&vd);
		}

		// Texture view for sampling the shadow map in the colour pass
		{
			sg_view_desc vd = {};
			vd.texture.image = color_img_;
			shadow_tex_view_ = sg_make_view(&vd);
		}

		// Nearest-neighbour sampler (manual PCF in the fragment shader)
		{
			sg_sampler_desc sd = {};
			sd.min_filter = SG_FILTER_NEAREST;
			sd.mag_filter = SG_FILTER_NEAREST;
			sd.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
			sd.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
			sampler_ = sg_make_sampler(&sd);
		}

		// Offscreen attachments (plain struct, not a GPU resource)
		attachments_.colors[0] = color_att_view_;
		attachments_.depth_stencil = depth_att_view_;

		// Shadow-pass pipeline (front-face culling reduces acne)
		{
			const char *vs = "#version 330\n"
			                 "layout(location=0) in vec3 position;\n"
			                 "layout(location=1) in vec3 normal;\n"
			                 "uniform mat4 light_mvp;\n"
			                 "void main() { gl_Position = light_mvp * vec4(position, 1.0); }\n";

			const char *fs = "#version 330\n"
			                 "out vec4 frag_color;\n"
			                 "void main() { frag_color = vec4(gl_FragCoord.z, 0.0, 0.0, 1.0); }\n";

			sg_shader_desc shd = {};
			shd.vertex_func.source = vs;
			shd.fragment_func.source = fs;
			shd.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
			shd.uniform_blocks[0].size = sizeof(math::mat4);
			shd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_MAT4, 0, "light_mvp"};

			sg_pipeline_desc pip = {};
			pip.shader = sg_make_shader(&shd);
			pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3;
			pip.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT3;
			pip.index_type = SG_INDEXTYPE_UINT16;
			pip.cull_mode = SG_CULLMODE_FRONT;
			pip.depth.write_enabled = true;
			pip.depth.compare = SG_COMPAREFUNC_LESS_EQUAL;
			pip.colors[0].pixel_format = SG_PIXELFORMAT_R32F;
			pip.depth.pixel_format = SG_PIXELFORMAT_DEPTH;
			pip.sample_count = 1;
			pipeline_ = sg_make_pipeline(&pip);
		}

		pass_action_.colors[0].load_action = SG_LOADACTION_CLEAR;
		pass_action_.colors[0].clear_value = {1.0f, 1.0f, 1.0f, 1.0f};
		pass_action_.depth.load_action = SG_LOADACTION_CLEAR;
		pass_action_.depth.clear_value = 1.0f;
	}

	void destroy() {
		auto sd = [](auto &r, auto fn) { if (r.id) { fn(r); r = {}; } };
		sd(color_att_view_, sg_destroy_view);
		sd(depth_att_view_, sg_destroy_view);
		sd(shadow_tex_view_, sg_destroy_view);
		sd(color_img_, sg_destroy_image);
		sd(depth_img_, sg_destroy_image);
		sd(sampler_, sg_destroy_sampler);
		sd(pipeline_, sg_destroy_pipeline);
	}

	// Compute an orthographic light-space VP matrix for a directional light.
	// scene_center / scene_radius define the bounding sphere to capture.
	math::mat4 compute_light_vp(const math::vec3 &light_dir, const math::vec3 &scene_center,
	                             float scene_radius) {
		math::vec3 dir = light_dir.normalized();
		math::vec3 light_pos = scene_center - dir * scene_radius;

		math::vec3 up = {0.0f, 1.0f, 0.0f};
		if (dir.y * dir.y > 0.98f)
			up = {0.0f, 0.0f, 1.0f};

		math::mat4 view = math::mat4::look_at(light_pos, scene_center, up);
		float e = scene_radius;
		math::mat4 proj = math::mat4::ortho(-e, e, -e, e, 0.0f, 2.0f * scene_radius);
		light_vp_ = proj * view;
		return light_vp_;
	}

	[[nodiscard]] sg_pipeline pipeline() const { return pipeline_; }
	[[nodiscard]] sg_attachments attachments() const { return attachments_; }
	[[nodiscard]] sg_pass_action pass_action() const { return pass_action_; }
	[[nodiscard]] sg_view shadow_view() const { return shadow_tex_view_; }
	[[nodiscard]] sg_sampler sampler() const { return sampler_; }
	[[nodiscard]] math::mat4 light_vp() const { return light_vp_; }
	[[nodiscard]] int resolution() const { return resolution_; }

private:
	sg_image color_img_{};
	sg_image depth_img_{};
	sg_view color_att_view_{};
	sg_view depth_att_view_{};
	sg_view shadow_tex_view_{};
	sg_sampler sampler_{};
	sg_attachments attachments_{};
	sg_pipeline pipeline_{};
	sg_pass_action pass_action_{};
	math::mat4 light_vp_{};
	int resolution_{SHADOW_MAP_DEFAULT_RES};
};

} // namespace scene
