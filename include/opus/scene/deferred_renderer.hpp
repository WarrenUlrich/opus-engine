#pragma once

#include "../math/mat4.hpp"
#include "../math/vec3.hpp"
#ifdef OPUS_EDITOR
#include "debug_grid.hpp"
#endif
#include "shadow_map.hpp"
#include "ssao.hpp"
#include "world.hpp"

#include <cstring>

namespace scene {

// Three-pass deferred renderer:
//   1. G-buffer pass  – writes world position, normal, albedo + PBR params to MRT
//   2. Light pass     – fullscreen quad, reads G-buffer + shadow map + SSAO,
//                       evaluates Cook-Torrance BRDF against all lights
//   3. Screen-effects – tonemapping (ACES), FXAA, vignette, gamma correction

class deferred_renderer {
public:
#ifdef OPUS_EDITOR
	debug_grid grid;
#endif
	shadow_map shadows;
	ssao ao;

	void init() {	
		create_gbuffer_pipeline();
		create_light_pipeline();
		create_screen_effects_pipeline();
		create_fullscreen_vbuf();

		{
			sg_sampler_desc sd = {};
			sd.min_filter = SG_FILTER_NEAREST;
			sd.mag_filter = SG_FILTER_NEAREST;
			sd.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
			sd.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
			sd.label = "deferred_nearest";
			nearest_sampler_ = sg_make_sampler(&sd);
		}

		// Light data texture consumed by the deferred light pass
		{
			sg_image_desc d = {};
			d.width = MAX_LIGHTS;
			d.height = 4;
			d.pixel_format = SG_PIXELFORMAT_RGBA32F;
			d.usage.immutable = false;
			d.usage.stream_update = true;
			d.label = "deferred_light_data";
			light_data_img_ = sg_make_image(&d);

			sg_view_desc vd = {};
			vd.texture.image = light_data_img_;
			light_data_view_ = sg_make_view(&vd);
		}

		light_data_pixels_.resize(MAX_LIGHTS * 4 * 4, 0.0f);

#ifdef OPUS_EDITOR
		// Linear sampler for displaying the viewport texture in ImGui
		{
			sg_sampler_desc sd = {};
			sd.min_filter = SG_FILTER_LINEAR;
			sd.mag_filter = SG_FILTER_LINEAR;
			sd.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
			sd.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
			sd.label = "deferred_linear";
			linear_sampler_ = sg_make_sampler(&sd);
		}

		grid.init();
#endif
		shadows.init();
		ao.init();
	}

	void destroy() {
#ifdef OPUS_EDITOR
		grid.destroy();
#endif
		shadows.destroy();
		ao.destroy();
		destroy_targets();

		auto safe_destroy = [](auto &r, auto fn) {
			if (r.id) { fn(r); r = {}; }
		};

		safe_destroy(gbuf_pipeline_, sg_destroy_pipeline);
		safe_destroy(light_pipeline_, sg_destroy_pipeline);
		safe_destroy(screen_pipeline_, sg_destroy_pipeline);
#ifndef OPUS_EDITOR
		safe_destroy(screen_swapchain_pipeline_, sg_destroy_pipeline);
#endif
		safe_destroy(nearest_sampler_, sg_destroy_sampler);
		safe_destroy(fsq_vbuf_, sg_destroy_buffer);
		safe_destroy(light_data_view_, sg_destroy_view);
		safe_destroy(light_data_img_, sg_destroy_image);
#ifdef OPUS_EDITOR
		safe_destroy(linear_sampler_, sg_destroy_sampler);
#endif
	}

#ifdef OPUS_EDITOR
	// Expose the final rendered image for external display (e.g. ImGui viewport).
	[[nodiscard]] sg_view final_color_view() const { return final_color_tex_view_; }
	[[nodiscard]] sg_sampler linear_sampler() const { return linear_sampler_; }
#endif

	// Draw the scene into an offscreen LDR texture.
	// Pass the desired viewport dimensions (may differ from window size
	// when rendering into an ImGui panel).  Does NOT call sg_commit().
	void draw(world &w, int vp_width = 0, int vp_height = 0) {
		const int sw = vp_width  > 0 ? vp_width  : sapp_width();
		const int sh = vp_height > 0 ? vp_height : sapp_height();

		ensure_targets(sw, sh);

		// ---- Extract camera ----
		math::mat4 view = math::mat4::identity();
		math::mat4 proj = math::mat4::identity();
		math::mat4 vp = math::mat4::identity();
		math::vec3 cam_pos{};

		w.for_each_entity<transform, camera>([&](transform &t, camera &cam) {
			cam.aspect_ratio = sapp_widthf() / sapp_heightf();
			cam_pos = t.position;
			math::vec3 fwd = t.rotation.rotate({0.0f, 0.0f, -1.0f});
			math::vec3 up = t.rotation.rotate({0.0f, 1.0f, 0.0f});
			view = math::mat4::look_at(t.position, t.position + fwd, up);
			proj = cam.projection_matrix();
			vp = proj * view;
		});

		// ---- Collect lights ----
		light collected[MAX_LIGHTS];
		int num_lights = 0;
		w.for_each_entity<transform, light>([&](transform &t, light &l) {
			if (num_lights >= MAX_LIGHTS) return;
			if (l.type != light_type::directional)
				l.position = t.position;
			collected[num_lights++] = l;
		});
		upload_light_data(collected, num_lights);

		// ---- 1. Shadow pass ----
		render_shadows(w);

		// ---- 2. SSAO (view-space G-buffer -> SSAO -> blur) ----
		render_ssao(w, view, vp, proj, sw, sh);

		// ---- 3. G-buffer pass (world-space MRT) ----
		render_gbuffer(w, vp);

		// ---- 4. Light pass (fullscreen, reads G-buffer) ----
		render_light_pass(cam_pos, num_lights, sw, sh);

#ifdef OPUS_EDITOR
		// ---- 5. Grid overlay (into HDR target with depth) ----
		render_grid_overlay(vp);

		// ---- 6. Screen effects (tonemap, FXAA, vignette) → offscreen LDR ----
		render_screen_effects(sw, sh);
#else
		// ---- 5. Screen effects (tonemap, FXAA, vignette) → swapchain ----
		render_screen_effects_to_swapchain(sw, sh);
#endif
	}

private:
	static constexpr int MAX_LIGHTS = 1024;

	// ========================= Fullscreen quad =========================
	sg_buffer fsq_vbuf_{};

	// ========================= Samplers ================================
	sg_sampler nearest_sampler_{};

	// ========================= Light data ==============================
	sg_image light_data_img_{};
	sg_view light_data_view_{};
	std::vector<float> light_data_pixels_;

	// ========================= G-buffer MRT ============================
	sg_pipeline gbuf_pipeline_{};
	sg_pass_action gbuf_pass_action_{};

	// 3 colour targets + depth
	sg_image gbuf_position_img_{};
	sg_image gbuf_normal_img_{};
	sg_image gbuf_albedo_img_{};
	sg_image gbuf_depth_img_{};

	sg_view gbuf_pos_att_view_{};
	sg_view gbuf_nor_att_view_{};
	sg_view gbuf_alb_att_view_{};
	sg_view gbuf_depth_att_view_{};

	sg_view gbuf_pos_tex_view_{};
	sg_view gbuf_nor_tex_view_{};
	sg_view gbuf_alb_tex_view_{};

	sg_attachments gbuf_att_{};

	struct gbuf_vs_uniforms {
		math::mat4 mvp;
		math::mat4 model;
		math::mat4 normal_mat;
	};
	struct gbuf_fs_uniforms {
		float material[4]{};
		float albedo[4]{};
	};

	// ========================= Light pass ==============================
	sg_pipeline light_pipeline_{};
	sg_pass_action light_pass_action_{};

	sg_image light_color_img_{};
	sg_view light_color_att_view_{};
	sg_view light_color_tex_view_{};
	sg_attachments light_att_{};

	struct light_fs_uniforms {
		float camera_pos[4]{};
		float ambient[4]{};
		float shadow_params[4]{};
		math::mat4 light_vp;
	};

	// ========================= Screen effects ==========================
	sg_pipeline screen_pipeline_{};
	sg_pass_action screen_pass_action_{};

	struct screen_uniforms {
		float screen_params[4]{};
	};

#ifdef OPUS_EDITOR
	// ========================= Final LDR output ==========================
	sg_image final_color_img_{};
	sg_view  final_color_att_view_{};
	sg_view  final_color_tex_view_{};
	sg_attachments final_att_{};
	sg_sampler linear_sampler_{};
#else
	// ========================= Swapchain screen-effects pipeline ==========
	sg_pipeline screen_swapchain_pipeline_{};
#endif

#ifdef OPUS_EDITOR
	// ========================= Grid overlay ============================
	sg_attachments grid_att_{};
	sg_pass_action grid_pass_action_{};
#endif

	// ========================= Size tracking ===========================
	int width_{0};
	int height_{0};

	// ===================== Pipeline creation ===========================

	void create_gbuffer_pipeline() {
		static const unsigned char vs_src[] = {
#embed "deferred_gbuffer.vert"
		    , 0};
		static const unsigned char fs_src[] = {
#embed "deferred_gbuffer.frag"
		    , 0};

		sg_shader_desc shd = {};
		shd.vertex_func.source = reinterpret_cast<const char *>(vs_src);
		shd.fragment_func.source = reinterpret_cast<const char *>(fs_src);

		// Vertex uniforms
		shd.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
		shd.uniform_blocks[0].size = sizeof(gbuf_vs_uniforms);
		shd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_MAT4, 0, "mvp"};
		shd.uniform_blocks[0].glsl_uniforms[1] = {SG_UNIFORMTYPE_MAT4, 0, "model"};
		shd.uniform_blocks[0].glsl_uniforms[2] = {SG_UNIFORMTYPE_MAT4, 0, "normal_mat"};

		// Fragment uniforms
		shd.uniform_blocks[1].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.uniform_blocks[1].size = sizeof(gbuf_fs_uniforms);
		shd.uniform_blocks[1].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, 0, "material"};
		shd.uniform_blocks[1].glsl_uniforms[1] = {SG_UNIFORMTYPE_FLOAT4, 0, "albedo"};

		sg_pipeline_desc pip = {};
		pip.shader = sg_make_shader(&shd);
		pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3;
		pip.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT3;
		pip.index_type = SG_INDEXTYPE_UINT16;
		pip.cull_mode = SG_CULLMODE_BACK;
		pip.depth.write_enabled = true;
		pip.depth.compare = SG_COMPAREFUNC_LESS_EQUAL;
		// 3 colour attachments: all RGBA16F
		pip.color_count = 3;
		pip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
		pip.colors[1].pixel_format = SG_PIXELFORMAT_RGBA16F;
		pip.colors[2].pixel_format = SG_PIXELFORMAT_RGBA16F;
		pip.depth.pixel_format = SG_PIXELFORMAT_DEPTH;
		pip.sample_count = 1;
		gbuf_pipeline_ = sg_make_pipeline(&pip);

		for (int i = 0; i < 3; ++i) {
			gbuf_pass_action_.colors[i].load_action = SG_LOADACTION_CLEAR;
			gbuf_pass_action_.colors[i].clear_value = {0.0f, 0.0f, 0.0f, 0.0f};
		}
		gbuf_pass_action_.depth.load_action = SG_LOADACTION_CLEAR;
		gbuf_pass_action_.depth.clear_value = 1.0f;
	}

	void create_light_pipeline() {
		static const unsigned char vs_src[] = {
#embed "fullscreen.vert"
		    , 0};
		static const unsigned char fs_src[] = {
#embed "deferred_light.frag"
		    , 0};

		sg_shader_desc shd = {};
		shd.vertex_func.source = reinterpret_cast<const char *>(vs_src);
		shd.fragment_func.source = reinterpret_cast<const char *>(fs_src);

		// Fragment uniforms
		shd.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.uniform_blocks[0].size = sizeof(light_fs_uniforms);
		shd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, 0, "camera_pos"};
		shd.uniform_blocks[0].glsl_uniforms[1] = {SG_UNIFORMTYPE_FLOAT4, 0, "ambient"};
		shd.uniform_blocks[0].glsl_uniforms[2] = {SG_UNIFORMTYPE_FLOAT4, 0, "shadow_params"};
		shd.uniform_blocks[0].glsl_uniforms[3] = {SG_UNIFORMTYPE_MAT4, 0, "light_vp"};

		// G-buffer textures (3) + shadow map + SSAO + light data
		for (int i = 0; i < 6; ++i) {
			shd.views[i].texture.stage = SG_SHADERSTAGE_FRAGMENT;
			shd.views[i].texture.image_type = SG_IMAGETYPE_2D;
			shd.views[i].texture.sample_type = SG_IMAGESAMPLETYPE_UNFILTERABLE_FLOAT;
		}

		shd.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.samplers[0].sampler_type = SG_SAMPLERTYPE_NONFILTERING;

		const char *tex_names[] = {"g_position_tex", "g_normal_tex", "g_albedo_tex",
		                           "shadow_map_tex", "ssao_tex", "light_data_tex"};
		for (int i = 0; i < 6; ++i) {
			shd.texture_sampler_pairs[i].stage = SG_SHADERSTAGE_FRAGMENT;
			shd.texture_sampler_pairs[i].view_slot = i;
			shd.texture_sampler_pairs[i].sampler_slot = 0;
			shd.texture_sampler_pairs[i].glsl_name = tex_names[i];
		}

		sg_pipeline_desc pip = {};
		pip.shader = sg_make_shader(&shd);
		pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT2;
		pip.depth.write_enabled = false;
		pip.depth.compare = SG_COMPAREFUNC_ALWAYS;
		pip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
		pip.depth.pixel_format = SG_PIXELFORMAT_NONE;
		pip.sample_count = 1;
		light_pipeline_ = sg_make_pipeline(&pip);

		light_pass_action_.colors[0].load_action = SG_LOADACTION_CLEAR;
		light_pass_action_.colors[0].clear_value = {0.0f, 0.0f, 0.0f, 0.0f};
	}

	void create_screen_effects_pipeline() {
		static const unsigned char vs_src[] = {
#embed "fullscreen.vert"
		    , 0};
		static const unsigned char fs_src[] = {
#embed "screen_effects.frag"
		    , 0};

		sg_shader_desc shd = {};
		shd.vertex_func.source = reinterpret_cast<const char *>(vs_src);
		shd.fragment_func.source = reinterpret_cast<const char *>(fs_src);

		shd.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.uniform_blocks[0].size = sizeof(screen_uniforms);
		shd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, 0, "screen_params"};

		shd.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
		shd.views[0].texture.image_type = SG_IMAGETYPE_2D;
		shd.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_UNFILTERABLE_FLOAT;

		shd.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.samplers[0].sampler_type = SG_SAMPLERTYPE_NONFILTERING;

		shd.texture_sampler_pairs[0].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.texture_sampler_pairs[0].view_slot = 0;
		shd.texture_sampler_pairs[0].sampler_slot = 0;
		shd.texture_sampler_pairs[0].glsl_name = "scene_tex";

		sg_pipeline_desc pip = {};
		pip.shader = sg_make_shader(&shd);
		pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT2;
		pip.depth.write_enabled = false;
		pip.depth.compare = SG_COMPAREFUNC_ALWAYS;
		pip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA8;
		pip.depth.pixel_format = SG_PIXELFORMAT_NONE;
		pip.sample_count = 1;
		screen_pipeline_ = sg_make_pipeline(&pip);

		screen_pass_action_.colors[0].load_action = SG_LOADACTION_DONTCARE;
		screen_pass_action_.depth.load_action = SG_LOADACTION_CLEAR;
		screen_pass_action_.depth.clear_value = 1.0f;

#ifndef OPUS_EDITOR
		// Game mode: create a second pipeline that renders to the swapchain
		// (uses default pixel format / sample count to match the swapchain)
		sg_pipeline_desc swp = {};
		swp.shader = pip.shader; // reuse the same shader
		swp.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT2;
		swp.depth.write_enabled = false;
		swp.depth.compare = SG_COMPAREFUNC_ALWAYS;
		// leave pixel_format at default (0) to match swapchain
		// leave sample_count at default to match swapchain
		screen_swapchain_pipeline_ = sg_make_pipeline(&swp);
#endif
	}

	void create_fullscreen_vbuf() {
		float verts[] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
		sg_buffer_desc bd = {};
		bd.data = SG_RANGE(verts);
		bd.usage.vertex_buffer = true;
		bd.label = "deferred_fsq";
		fsq_vbuf_ = sg_make_buffer(&bd);
	}

	// ===================== Dynamic targets =============================

	void ensure_targets(int w, int h) {
		if (w == width_ && h == height_) return;
		destroy_targets();
		width_ = w;
		height_ = h;
		create_targets(w, h);
	}

	void create_targets(int w, int h) {
		auto make_color_target = [&](sg_pixel_format fmt, const char *label,
		                             sg_image &img, sg_view &att_view, sg_view &tex_view) {
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

		// G-buffer MRT (3 colour + 1 depth)
		make_color_target(SG_PIXELFORMAT_RGBA16F, "gbuf_position",
		                  gbuf_position_img_, gbuf_pos_att_view_, gbuf_pos_tex_view_);
		make_color_target(SG_PIXELFORMAT_RGBA16F, "gbuf_normal",
		                  gbuf_normal_img_, gbuf_nor_att_view_, gbuf_nor_tex_view_);
		make_color_target(SG_PIXELFORMAT_RGBA16F, "gbuf_albedo",
		                  gbuf_albedo_img_, gbuf_alb_att_view_, gbuf_alb_tex_view_);

		{
			sg_image_desc d = {};
			d.usage.depth_stencil_attachment = true;
			d.width = w;
			d.height = h;
			d.pixel_format = SG_PIXELFORMAT_DEPTH;
			d.sample_count = 1;
			d.label = "gbuf_depth";
			gbuf_depth_img_ = sg_make_image(&d);

			sg_view_desc vd = {};
			vd.depth_stencil_attachment.image = gbuf_depth_img_;
			gbuf_depth_att_view_ = sg_make_view(&vd);
		}

		gbuf_att_ = {};
		gbuf_att_.colors[0] = gbuf_pos_att_view_;
		gbuf_att_.colors[1] = gbuf_nor_att_view_;
		gbuf_att_.colors[2] = gbuf_alb_att_view_;
		gbuf_att_.depth_stencil = gbuf_depth_att_view_;

		// Light pass output (RGBA16F, HDR)
		make_color_target(SG_PIXELFORMAT_RGBA16F, "light_pass_color",
		                  light_color_img_, light_color_att_view_, light_color_tex_view_);

		light_att_ = {};
		light_att_.colors[0] = light_color_att_view_;

#ifdef OPUS_EDITOR
		// Grid overlay: light HDR target + G-buffer depth
		grid_att_ = {};
		grid_att_.colors[0] = light_color_att_view_;
		grid_att_.depth_stencil = gbuf_depth_att_view_;

		grid_pass_action_ = {};
		grid_pass_action_.colors[0].load_action = SG_LOADACTION_LOAD;
		grid_pass_action_.depth.load_action = SG_LOADACTION_LOAD;

		// Final LDR output (screen effects write here instead of swapchain)
		make_color_target(SG_PIXELFORMAT_RGBA8, "final_ldr_color",
		                  final_color_img_, final_color_att_view_, final_color_tex_view_);
		final_att_ = {};
		final_att_.colors[0] = final_color_att_view_;
#endif
	}

	void destroy_targets() {
		auto safe_view = [](sg_view &v) { if (v.id) { sg_destroy_view(v); v = {}; } };
		auto safe_img = [](sg_image &i) { if (i.id) { sg_destroy_image(i); i = {}; } };

		safe_view(gbuf_pos_att_view_);
		safe_view(gbuf_nor_att_view_);
		safe_view(gbuf_alb_att_view_);
		safe_view(gbuf_depth_att_view_);
		safe_view(gbuf_pos_tex_view_);
		safe_view(gbuf_nor_tex_view_);
		safe_view(gbuf_alb_tex_view_);
		safe_img(gbuf_position_img_);
		safe_img(gbuf_normal_img_);
		safe_img(gbuf_albedo_img_);
		safe_img(gbuf_depth_img_);

		safe_view(light_color_att_view_);
		safe_view(light_color_tex_view_);
		safe_img(light_color_img_);

#ifdef OPUS_EDITOR
		safe_view(final_color_att_view_);
		safe_view(final_color_tex_view_);
		safe_img(final_color_img_);
#endif

		width_ = 0;
		height_ = 0;
	}

	// ===================== Light data upload ===========================

	void upload_light_data(const light *lights, int n) {
		std::memset(light_data_pixels_.data(), 0, light_data_pixels_.size() * sizeof(float));

		for (int i = 0; i < n; ++i) {
			const auto &l = lights[i];
			float *row0 = &light_data_pixels_[(0 * MAX_LIGHTS + i) * 4];
			float *row1 = &light_data_pixels_[(1 * MAX_LIGHTS + i) * 4];
			float *row2 = &light_data_pixels_[(2 * MAX_LIGHTS + i) * 4];
			float *row3 = &light_data_pixels_[(3 * MAX_LIGHTS + i) * 4];

			row0[0] = l.position.x; row0[1] = l.position.y;
			row0[2] = l.position.z; row0[3] = static_cast<float>(static_cast<int>(l.type));
			row1[0] = l.direction.x; row1[1] = l.direction.y;
			row1[2] = l.direction.z; row1[3] = l.range;
			row2[0] = l.color.x; row2[1] = l.color.y;
			row2[2] = l.color.z; row2[3] = l.intensity;
			row3[0] = l.inner_cone_cos; row3[1] = l.outer_cone_cos;
		}

		sg_image_data d = {};
		d.mip_levels[0].ptr = light_data_pixels_.data();
		d.mip_levels[0].size = light_data_pixels_.size() * sizeof(float);
		sg_update_image(light_data_img_, &d);
	}

	// ===================== Render passes ===============================

	void render_shadows(world &w) {
		math::vec3 sun_dir = {0.0f, -1.0f, 0.0f};
		w.for_each_entity<transform, light>([&](transform &, light &l) {
			if (l.type == light_type::directional)
				sun_dir = l.direction;
		});

		shadows.compute_light_vp(sun_dir, {0.0f, 0.0f, 0.0f}, 20.0f);

		sg_pass pass = {};
		pass.action = shadows.pass_action();
		pass.attachments = shadows.attachments();
		sg_begin_pass(pass);
		sg_apply_pipeline(shadows.pipeline());

		w.for_each_entity<transform, mesh_instance, material>(
		    [&](transform &t, mesh_instance &mesh, material &) {
			    math::mat4 light_mvp = shadows.light_vp() * t.matrix();
			    sg_apply_bindings(&mesh.bind);
			    sg_apply_uniforms(0, SG_RANGE(light_mvp));
			    sg_draw(0, mesh.index_count, 1);
		    });

		sg_end_pass();
	}

	void render_ssao(world &w, const math::mat4 &view, const math::mat4 &vp,
	                 const math::mat4 &proj, int sw, int sh) {
		ao.ensure_targets(sw, sh);

		ao.begin_gbuffer_pass();
		w.for_each_entity<transform, mesh_instance, material>(
		    [&](transform &t, mesh_instance &mesh, material &) {
			    math::mat4 model = t.matrix();
			    math::mat4 mv = view * model;
			    math::mat4 nmv = view * t.rotation.to_mat4() *
			        math::mat4::scale({1.0f / t.scale.x, 1.0f / t.scale.y, 1.0f / t.scale.z});
			    ssao::gbuf_uniforms u{};
			    u.mvp = vp * model;
			    u.mv = mv;
			    u.normal_mv = nmv;
			    ao.draw_gbuffer_mesh(mesh.bind, mesh.index_count, u);
		    });
		ao.end_gbuffer_pass();

		ao.compute(proj);
	}

	// Pass 1: G-buffer
	void render_gbuffer(world &w, const math::mat4 &vp) {
		sg_pass pass = {};
		pass.action = gbuf_pass_action_;
		pass.attachments = gbuf_att_;
		sg_begin_pass(pass);

		sg_apply_pipeline(gbuf_pipeline_);

		w.for_each_entity<transform, mesh_instance, material>(
		    [&](transform &t, mesh_instance &mesh, material &mat) {
			    math::mat4 model = t.matrix();
			    math::mat4 normal_mat = t.rotation.to_mat4() *
			        math::mat4::scale({1.0f / t.scale.x, 1.0f / t.scale.y, 1.0f / t.scale.z});

			    gbuf_vs_uniforms vs{.mvp = vp * model, .model = model, .normal_mat = normal_mat};
			    gbuf_fs_uniforms fs{
			        .material = {mat.metallic, mat.roughness, mat.ao},
			        .albedo = {mat.albedo.x, mat.albedo.y, mat.albedo.z, mat.alpha},
			    };

			    sg_apply_bindings(&mesh.bind);
			    sg_apply_uniforms(0, SG_RANGE(vs));
			    sg_apply_uniforms(1, SG_RANGE(fs));
			    sg_draw(0, mesh.index_count, 1);
		    });

		sg_end_pass();
	}

#ifdef OPUS_EDITOR
	// Grid overlay: draw into HDR target with G-buffer depth
	void render_grid_overlay(const math::mat4 &vp) {
		sg_pass pass = {};
		pass.action = grid_pass_action_;
		pass.attachments = grid_att_;
		sg_begin_pass(pass);

		grid.draw(vp);

		sg_end_pass();
	}
#endif

	// Pass 2: Lighting (fullscreen)
	void render_light_pass(const math::vec3 &cam_pos, int num_lights, int sw, int sh) {
		sg_pass pass = {};
		pass.action = light_pass_action_;
		pass.attachments = light_att_;
		sg_begin_pass(pass);

		sg_apply_pipeline(light_pipeline_);

		light_fs_uniforms u{
		    .camera_pos = {cam_pos.x, cam_pos.y, cam_pos.z, float(num_lights)},
		    .ambient = {0.03f, 0.03f, 0.03f},
		    .shadow_params = {float(shadows.resolution()), 0.0002f, 1.0f, 0.003f},
		    .light_vp = shadows.light_vp(),
		};

		sg_bindings bind = {};
		bind.vertex_buffers[0] = fsq_vbuf_;
		bind.views[0] = gbuf_pos_tex_view_;
		bind.views[1] = gbuf_nor_tex_view_;
		bind.views[2] = gbuf_alb_tex_view_;
		bind.views[3] = shadows.shadow_view();
		bind.views[4] = ao.result_view();
		bind.views[5] = light_data_view_;
		bind.samplers[0] = nearest_sampler_;

		sg_apply_bindings(&bind);
		sg_apply_uniforms(0, SG_RANGE(u));
		sg_draw(0, 3, 1);

		sg_end_pass();
	}

#ifdef OPUS_EDITOR
	// Pass 3: Screen effects (tonemap + FXAA + vignette -> offscreen LDR)
	void render_screen_effects(int sw, int sh) {
		sg_pass pass = {};
		pass.action = screen_pass_action_;
		pass.attachments = final_att_;
		sg_begin_pass(pass);

		sg_apply_pipeline(screen_pipeline_);

		screen_uniforms u{.screen_params = {1.0f / float(sw), 1.0f / float(sh), 0.25f}};

		sg_bindings bind = {};
		bind.vertex_buffers[0] = fsq_vbuf_;
		bind.views[0] = light_color_tex_view_;
		bind.samplers[0] = nearest_sampler_;

		sg_apply_bindings(&bind);
		sg_apply_uniforms(0, SG_RANGE(u));
		sg_draw(0, 3, 1);

		sg_end_pass();
	}
#else
	// Game mode: screen effects render directly to the swapchain
	void render_screen_effects_to_swapchain(int sw, int sh) {
		sg_pass pass = {};
		pass.swapchain = sglue_swapchain();
		pass.action.colors[0].load_action = SG_LOADACTION_DONTCARE;
		sg_begin_pass(pass);

		sg_apply_pipeline(screen_swapchain_pipeline_);

		screen_uniforms u{.screen_params = {1.0f / float(sw), 1.0f / float(sh), 0.25f}};

		sg_bindings bind = {};
		bind.vertex_buffers[0] = fsq_vbuf_;
		bind.views[0] = light_color_tex_view_;
		bind.samplers[0] = nearest_sampler_;

		sg_apply_bindings(&bind);
		sg_apply_uniforms(0, SG_RANGE(u));
		sg_draw(0, 3, 1);

		sg_end_pass();
	}
#endif
};

} // namespace scene
