#pragma once

#include "../math/mat4.hpp"
#include "../math/vec3.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace scene {

inline constexpr int SSAO_KERNEL_SIZE = 64;
inline constexpr int SSAO_NOISE_DIM = 4; // 4x4 noise texture

class ssao {
public:
	ssao() = default;

	void init() {
		generate_kernel();
		create_noise_texture();
		create_fullscreen_vbuf();
		create_gbuffer_pipeline();
		create_ssao_pipeline();
		create_blur_pipeline();
	}

	// Recreate offscreen targets when the viewport changes.
	void ensure_targets(int w, int h) {
		if (w == width_ && h == height_)
			return;
		destroy_targets();
		width_ = w;
		height_ = h;
		create_targets(w, h);
	}

	// --- Pass 1: G-buffer (call from renderer) ---

	struct gbuf_uniforms {
		math::mat4 mvp;
		math::mat4 mv;
		math::mat4 normal_mv;
	};

	void begin_gbuffer_pass() {
		sg_pass pass = {};
		pass.action = gbuf_pass_action_;
		pass.attachments = gbuf_att_;
		sg_begin_pass(pass);
		sg_apply_pipeline(gbuf_pipeline_);
	}

	void draw_gbuffer_mesh(const sg_bindings &mesh_bind, int index_count,
	                       const gbuf_uniforms &uniforms) {
		sg_apply_bindings(&mesh_bind);
		sg_apply_uniforms(0, SG_RANGE(uniforms));
		sg_draw(0, index_count, 1);
	}

	void end_gbuffer_pass() { sg_end_pass(); }

	// --- Pass 2 + 3: SSAO + blur (self-contained) ---

	void compute(const math::mat4 &projection) {
		// SSAO pass
		{
			ssao_uniforms u{};
			for (int i = 0; i < SSAO_KERNEL_SIZE; ++i) {
				u.samples[i][0] = kernel_[i].x;
				u.samples[i][1] = kernel_[i].y;
				u.samples[i][2] = kernel_[i].z;
			}
			u.ssao_params[0] = float(width_);
			u.ssao_params[1] = float(height_);
			u.projection = projection;

			sg_pass pass = {};
			pass.action = ssao_pass_action_;
			pass.attachments = ssao_att_;
			sg_begin_pass(pass);
			sg_apply_pipeline(ssao_pipeline_);

			sg_bindings bind = {};
			bind.vertex_buffers[0] = fsq_vbuf_;
			bind.views[0] = gbuf_tex_view_;
			bind.views[1] = noise_tex_view_;
			bind.samplers[0] = nearest_sampler_;
			bind.samplers[1] = noise_sampler_;
			sg_apply_bindings(&bind);
			sg_apply_uniforms(0, SG_RANGE(u));
			sg_draw(0, 3, 1);

			sg_end_pass();
		}

		// Blur pass
		{
			blur_uniforms u{.texel_size = {1.0f / float(width_), 1.0f / float(height_)}};

			sg_pass pass = {};
			pass.action = blur_pass_action_;
			pass.attachments = blur_att_;
			sg_begin_pass(pass);
			sg_apply_pipeline(blur_pipeline_);

			sg_bindings bind = {};
			bind.vertex_buffers[0] = fsq_vbuf_;
			bind.views[0] = ssao_tex_view_;
			bind.samplers[0] = nearest_sampler_;
			sg_apply_bindings(&bind);
			sg_apply_uniforms(0, SG_RANGE(u));
			sg_draw(0, 3, 1);

			sg_end_pass();
		}
	}

	[[nodiscard]] sg_view result_view() const { return blur_tex_view_; }
	[[nodiscard]] sg_sampler result_sampler() const { return nearest_sampler_; }

	void destroy() {
		destroy_targets();
		auto sd = [](auto &r, auto fn) {
			if (r.id) {
				fn(r);
				r = {};
			}
		};
		sd(noise_tex_view_, sg_destroy_view);
		sd(noise_img_, sg_destroy_image);
		sd(noise_sampler_, sg_destroy_sampler);
		sd(nearest_sampler_, sg_destroy_sampler);
		sd(gbuf_pipeline_, sg_destroy_pipeline);
		sd(ssao_pipeline_, sg_destroy_pipeline);
		sd(blur_pipeline_, sg_destroy_pipeline);
		sd(fsq_vbuf_, sg_destroy_buffer);
	}

private:
	// ---------- Uniform structs ----------

	struct ssao_uniforms {
		float samples[SSAO_KERNEL_SIZE][4]; // vec4[64]
		float ssao_params[4];               // vec4
		math::mat4 projection;              // mat4
	};

	struct blur_uniforms {
		float texel_size[4]; // vec4
	};

	// ---------- Resources ----------

	// G-buffer
	sg_image gbuf_color_img_{};
	sg_image gbuf_depth_img_{};
	sg_view gbuf_color_att_view_{};
	sg_view gbuf_depth_att_view_{};
	sg_view gbuf_tex_view_{};
	sg_attachments gbuf_att_{};
	sg_pipeline gbuf_pipeline_{};
	sg_pass_action gbuf_pass_action_{};

	// SSAO
	sg_image ssao_color_img_{};
	sg_view ssao_color_att_view_{};
	sg_view ssao_tex_view_{};
	sg_attachments ssao_att_{};
	sg_pipeline ssao_pipeline_{};
	sg_pass_action ssao_pass_action_{};

	// Blur
	sg_image blur_color_img_{};
	sg_view blur_color_att_view_{};
	sg_view blur_tex_view_{};
	sg_attachments blur_att_{};
	sg_pipeline blur_pipeline_{};
	sg_pass_action blur_pass_action_{};

	// Shared
	sg_image noise_img_{};
	sg_view noise_tex_view_{};
	sg_sampler nearest_sampler_{};
	sg_sampler noise_sampler_{};
	sg_buffer fsq_vbuf_{};

	std::vector<math::vec3> kernel_;
	int width_{0};
	int height_{0};

	// ---------- Initialisation helpers ----------

	void generate_kernel() {
		kernel_.resize(SSAO_KERNEL_SIZE);

		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
		std::uniform_real_distribution<float> dist_neg(-1.0f, 1.0f);

		for (int i = 0; i < SSAO_KERNEL_SIZE; ++i) {
			// Random direction in tangent-space hemisphere (z >= 0)
			math::vec3 sample = {
			    dist_neg(rng),
			    dist_neg(rng),
			    dist01(rng), // ensure hemisphere (z positive)
			};
			sample.normalize();
			sample *= dist01(rng); // random length [0,1]

			// Accelerating interpolation: bias samples towards centre
			float scale = static_cast<float>(i) / static_cast<float>(SSAO_KERNEL_SIZE);
			scale = 0.1f + 0.9f * (scale * scale); // lerp(0.1, 1.0, scale²)
			sample *= scale;

			kernel_[i] = sample;
		}
	}

	void create_noise_texture() {
		std::mt19937 rng(73);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

		// 4x4 RGBA32F texture with random tangent-space rotation vectors
		std::vector<float> pixels(SSAO_NOISE_DIM * SSAO_NOISE_DIM * 4);
		for (int i = 0; i < SSAO_NOISE_DIM * SSAO_NOISE_DIM; ++i) {
			// Rotation around normal (z = 0 in tangent space)
			math::vec3 v = {dist(rng), dist(rng), 0.0f};
			v.normalize();
			pixels[i * 4 + 0] = v.x;
			pixels[i * 4 + 1] = v.y;
			pixels[i * 4 + 2] = v.z;
			pixels[i * 4 + 3] = 0.0f;
		}

		sg_image_desc d = {};
		d.width = SSAO_NOISE_DIM;
		d.height = SSAO_NOISE_DIM;
		d.pixel_format = SG_PIXELFORMAT_RGBA32F;
		d.data.mip_levels[0] = {pixels.data(), pixels.size() * sizeof(float)};
		d.label = "ssao_noise";
		noise_img_ = sg_make_image(&d);

		sg_view_desc vd = {};
		vd.texture.image = noise_img_;
		noise_tex_view_ = sg_make_view(&vd);

		// Nearest-neighbour samplers
		{
			sg_sampler_desc sd = {};
			sd.min_filter = SG_FILTER_NEAREST;
			sd.mag_filter = SG_FILTER_NEAREST;
			sd.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
			sd.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
			sd.label = "ssao_nearest";
			nearest_sampler_ = sg_make_sampler(&sd);
		}
		{
			sg_sampler_desc sd = {};
			sd.min_filter = SG_FILTER_NEAREST;
			sd.mag_filter = SG_FILTER_NEAREST;
			sd.wrap_u = SG_WRAP_REPEAT;
			sd.wrap_v = SG_WRAP_REPEAT;
			sd.label = "ssao_noise_smp";
			noise_sampler_ = sg_make_sampler(&sd);
		}
	}

	void create_fullscreen_vbuf() {
		// Single oversized triangle covering the entire viewport
		float verts[] = {
		    -1.0f, -1.0f, // 0
		    3.0f,  -1.0f, // 1
		    -1.0f, 3.0f,  // 2
		};
		sg_buffer_desc bd = {};
		bd.data = SG_RANGE(verts);
		bd.usage.vertex_buffer = true;
		bd.label = "ssao_fsq";
		fsq_vbuf_ = sg_make_buffer(&bd);
	}

	void create_gbuffer_pipeline() {
		static const unsigned char vs[] = {
#embed "ssao_gbuffer.vert"
		    , 0};
		static const unsigned char fs[] = {
#embed "ssao_gbuffer.frag"
		    , 0};

		sg_shader_desc shd = {};
		shd.vertex_func.source = reinterpret_cast<const char *>(vs);
		shd.fragment_func.source = reinterpret_cast<const char *>(fs);

		shd.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
		shd.uniform_blocks[0].size = sizeof(gbuf_uniforms);
		shd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_MAT4, 0, "mvp"};
		shd.uniform_blocks[0].glsl_uniforms[1] = {SG_UNIFORMTYPE_MAT4, 0, "mv"};
		shd.uniform_blocks[0].glsl_uniforms[2] = {SG_UNIFORMTYPE_MAT4, 0, "normal_mv"};

		sg_pipeline_desc pip = {};
		pip.shader = sg_make_shader(&shd);
		pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3;
		pip.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT3;
		pip.index_type = SG_INDEXTYPE_UINT16;
		pip.cull_mode = SG_CULLMODE_BACK;
		pip.depth.write_enabled = true;
		pip.depth.compare = SG_COMPAREFUNC_LESS_EQUAL;
		pip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
		pip.depth.pixel_format = SG_PIXELFORMAT_DEPTH;
		pip.sample_count = 1;
		gbuf_pipeline_ = sg_make_pipeline(&pip);

		gbuf_pass_action_.colors[0].load_action = SG_LOADACTION_CLEAR;
		gbuf_pass_action_.colors[0].clear_value = {0.0f, 0.0f, 0.0f, 0.0f};
		gbuf_pass_action_.depth.load_action = SG_LOADACTION_CLEAR;
		gbuf_pass_action_.depth.clear_value = 1.0f;
	}

	void create_ssao_pipeline() {
		static const unsigned char vs[] = {
#embed "fullscreen.vert"
		    , 0};
		static const unsigned char fs[] = {
#embed "ssao.frag"
		    , 0};

		sg_shader_desc shd = {};
		shd.vertex_func.source = reinterpret_cast<const char *>(vs);
		shd.fragment_func.source = reinterpret_cast<const char *>(fs);

		// Fragment uniform block: vec4[64] + vec4 + mat4
		shd.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.uniform_blocks[0].size = sizeof(ssao_uniforms);
		shd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, 64, "samples"};
		shd.uniform_blocks[0].glsl_uniforms[1] = {SG_UNIFORMTYPE_FLOAT4, 0, "ssao_params"};
		shd.uniform_blocks[0].glsl_uniforms[2] = {SG_UNIFORMTYPE_MAT4, 0, "projection"};

		// G-buffer texture (RGBA16F, unfilterable)
		shd.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
		shd.views[0].texture.image_type = SG_IMAGETYPE_2D;
		shd.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_UNFILTERABLE_FLOAT;

		// Noise texture (RGBA32F, unfilterable)
		shd.views[1].texture.stage = SG_SHADERSTAGE_FRAGMENT;
		shd.views[1].texture.image_type = SG_IMAGETYPE_2D;
		shd.views[1].texture.sample_type = SG_IMAGESAMPLETYPE_UNFILTERABLE_FLOAT;

		shd.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.samplers[0].sampler_type = SG_SAMPLERTYPE_NONFILTERING;

		shd.samplers[1].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.samplers[1].sampler_type = SG_SAMPLERTYPE_NONFILTERING;

		shd.texture_sampler_pairs[0].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.texture_sampler_pairs[0].view_slot = 0;
		shd.texture_sampler_pairs[0].sampler_slot = 0;
		shd.texture_sampler_pairs[0].glsl_name = "gbuffer_tex";

		shd.texture_sampler_pairs[1].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.texture_sampler_pairs[1].view_slot = 1;
		shd.texture_sampler_pairs[1].sampler_slot = 1;
		shd.texture_sampler_pairs[1].glsl_name = "noise_tex";

		sg_pipeline_desc pip = {};
		pip.shader = sg_make_shader(&shd);
		pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT2;
		pip.colors[0].pixel_format = SG_PIXELFORMAT_R32F;
		pip.depth.pixel_format = SG_PIXELFORMAT_NONE;
		pip.sample_count = 1;
		ssao_pipeline_ = sg_make_pipeline(&pip);

		ssao_pass_action_.colors[0].load_action = SG_LOADACTION_CLEAR;
		ssao_pass_action_.colors[0].clear_value = {1.0f, 1.0f, 1.0f, 1.0f};
	}

	void create_blur_pipeline() {
		static const unsigned char vs[] = {
#embed "fullscreen.vert"
		    , 0};
		static const unsigned char fs[] = {
#embed "ssao_blur.frag"
		    , 0};

		sg_shader_desc shd = {};
		shd.vertex_func.source = reinterpret_cast<const char *>(vs);
		shd.fragment_func.source = reinterpret_cast<const char *>(fs);

		shd.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.uniform_blocks[0].size = sizeof(blur_uniforms);
		shd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, 0, "texel_size"};

		shd.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
		shd.views[0].texture.image_type = SG_IMAGETYPE_2D;
		shd.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_UNFILTERABLE_FLOAT;

		shd.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.samplers[0].sampler_type = SG_SAMPLERTYPE_NONFILTERING;

		shd.texture_sampler_pairs[0].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.texture_sampler_pairs[0].view_slot = 0;
		shd.texture_sampler_pairs[0].sampler_slot = 0;
		shd.texture_sampler_pairs[0].glsl_name = "ssao_tex";

		sg_pipeline_desc pip = {};
		pip.shader = sg_make_shader(&shd);
		pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT2;
		pip.colors[0].pixel_format = SG_PIXELFORMAT_R32F;
		pip.depth.pixel_format = SG_PIXELFORMAT_NONE;
		pip.sample_count = 1;
		blur_pipeline_ = sg_make_pipeline(&pip);

		blur_pass_action_.colors[0].load_action = SG_LOADACTION_CLEAR;
		blur_pass_action_.colors[0].clear_value = {1.0f, 1.0f, 1.0f, 1.0f};
	}

	// ---------- Dynamic target management ----------

	void create_targets(int w, int h) {
		auto make_target = [&](sg_pixel_format fmt, const char *label, sg_image &img, sg_view &att_view,
		                       sg_view &tex_view) {
			sg_image_desc d = {};
			d.usage.color_attachment = true;
			d.width = w;
			d.height = h;
			d.pixel_format = fmt;
			d.sample_count = 1;
			d.label = label;
			img = sg_make_image(&d);
			sg_view_desc va = {};
			va.color_attachment.image = img;
			att_view = sg_make_view(&va);
			sg_view_desc vt = {};
			vt.texture.image = img;
			tex_view = sg_make_view(&vt);
		};

		make_target(SG_PIXELFORMAT_RGBA16F, "ssao_gbuf_color", gbuf_color_img_, gbuf_color_att_view_,
		            gbuf_tex_view_);
		{
			sg_image_desc dd = {};
			dd.usage.depth_stencil_attachment = true;
			dd.width = w;
			dd.height = h;
			dd.pixel_format = SG_PIXELFORMAT_DEPTH;
			dd.sample_count = 1;
			dd.label = "ssao_gbuf_depth";
			gbuf_depth_img_ = sg_make_image(&dd);
			sg_view_desc vda = {};
			vda.depth_stencil_attachment.image = gbuf_depth_img_;
			gbuf_depth_att_view_ = sg_make_view(&vda);
		}
		gbuf_att_ = {};
		gbuf_att_.colors[0] = gbuf_color_att_view_;
		gbuf_att_.depth_stencil = gbuf_depth_att_view_;

		make_target(SG_PIXELFORMAT_R32F, "ssao_raw", ssao_color_img_, ssao_color_att_view_,
		            ssao_tex_view_);
		ssao_att_ = {};
		ssao_att_.colors[0] = ssao_color_att_view_;

		make_target(SG_PIXELFORMAT_R32F, "ssao_blur", blur_color_img_, blur_color_att_view_,
		            blur_tex_view_);
		blur_att_ = {};
		blur_att_.colors[0] = blur_color_att_view_;
	}

	void destroy_targets() {
		auto sd = [](auto &r, auto fn) {
			if (r.id) {
				fn(r);
				r = {};
			}
		};
		sd(gbuf_color_att_view_, sg_destroy_view);
		sd(gbuf_depth_att_view_, sg_destroy_view);
		sd(gbuf_tex_view_, sg_destroy_view);
		sd(gbuf_color_img_, sg_destroy_image);
		sd(gbuf_depth_img_, sg_destroy_image);
		sd(ssao_color_att_view_, sg_destroy_view);
		sd(ssao_tex_view_, sg_destroy_view);
		sd(ssao_color_img_, sg_destroy_image);
		sd(blur_color_att_view_, sg_destroy_view);
		sd(blur_tex_view_, sg_destroy_view);
		sd(blur_color_img_, sg_destroy_image);
		width_ = 0;
		height_ = 0;
	}
};

} // namespace scene
