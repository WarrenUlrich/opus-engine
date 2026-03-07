#pragma once

#include "../ecs/ecs.hpp"
#include "../math/mat4.hpp"
#include "camera.hpp"
#include "debug_grid.hpp"
#include "light_grid.hpp"
#include "lighting.hpp"
#include "material.hpp"
#include "mesh_instance.hpp"
#include "shadow_map.hpp"
#include "ssao.hpp"
#include "tonemap.hpp"
#include "transform.hpp"

namespace scene {

using world = ecs::context<transform, camera, mesh_instance, material, light>;

class forward_plus_renderer {
public:
	light_grid tiles;
	debug_grid grid;
	shadow_map shadows;
	ssao ao;
	tonemap hdr;

	void init() {
		{
			const char *vs = "#version 330\n"
			                 "layout(location=0) in vec3 position;\n"
			                 "layout(location=1) in vec3 normal;\n"
			                 "uniform mat4 mvp;\n"
			                 "void main() { gl_Position = mvp * vec4(position, 1.0); }\n";
			const char *fs = "#version 330\n"
			                 "out vec4 frag_color;\n"
			                 "void main() { frag_color = vec4(0.0); }\n";

			sg_shader_desc shd = {};
			shd.vertex_func.source = vs;
			shd.fragment_func.source = fs;
			shd.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
			shd.uniform_blocks[0].size = sizeof(math::mat4);
			shd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_MAT4, 0, "mvp"};

			sg_pipeline_desc pip = {};
			pip.shader = sg_make_shader(&shd);
			pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3;
			pip.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT3;
			pip.index_type = SG_INDEXTYPE_UINT16;
			pip.cull_mode = SG_CULLMODE_BACK;
			pip.depth.write_enabled = true;
			pip.depth.compare = SG_COMPAREFUNC_LESS_EQUAL;
			pip.colors[0].write_mask = SG_COLORMASK_NONE;
			pip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
			pip.depth.pixel_format = SG_PIXELFORMAT_DEPTH;
			pip.sample_count = 1;
			depth_pipeline_ = sg_make_pipeline(&pip);
		}

		{
			static const unsigned char vs_src[] = {
#embed "forward_plus.vert"
			    , 0};
			static const unsigned char fs_src[] = {
#embed "forward_plus.frag"
			    , 0};

			sg_shader_desc shd = {};
			shd.vertex_func.source = reinterpret_cast<const char *>(vs_src);
			shd.fragment_func.source = reinterpret_cast<const char *>(fs_src);

			shd.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
			shd.uniform_blocks[0].size = sizeof(vs_uniforms);
			shd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_MAT4, 0, "mvp"};
			shd.uniform_blocks[0].glsl_uniforms[1] = {SG_UNIFORMTYPE_MAT4, 0, "model"};
			shd.uniform_blocks[0].glsl_uniforms[2] = {SG_UNIFORMTYPE_MAT4, 0, "normal_mat"};
			shd.uniform_blocks[0].glsl_uniforms[3] = {SG_UNIFORMTYPE_MAT4, 0, "light_vp"};

			shd.uniform_blocks[1].stage = SG_SHADERSTAGE_FRAGMENT;
			shd.uniform_blocks[1].size = sizeof(fs_uniforms);
			shd.uniform_blocks[1].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, 0, "material"};
			shd.uniform_blocks[1].glsl_uniforms[1] = {SG_UNIFORMTYPE_FLOAT4, 0, "albedo"};
			shd.uniform_blocks[1].glsl_uniforms[2] = {SG_UNIFORMTYPE_FLOAT4, 0, "camera_pos"};
			shd.uniform_blocks[1].glsl_uniforms[3] = {SG_UNIFORMTYPE_FLOAT4, 0, "ambient"};
			shd.uniform_blocks[1].glsl_uniforms[4] = {SG_UNIFORMTYPE_FLOAT4, 0, "tile_info"};
			shd.uniform_blocks[1].glsl_uniforms[5] = {SG_UNIFORMTYPE_FLOAT4, 0, "shadow_params"};

			shd.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
			shd.views[0].texture.image_type = SG_IMAGETYPE_2D;
			shd.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_UNFILTERABLE_FLOAT;

			shd.views[1].texture.stage = SG_SHADERSTAGE_FRAGMENT;
			shd.views[1].texture.image_type = SG_IMAGETYPE_2D;
			shd.views[1].texture.sample_type = SG_IMAGESAMPLETYPE_UNFILTERABLE_FLOAT;

			shd.views[2].texture.stage = SG_SHADERSTAGE_FRAGMENT;
			shd.views[2].texture.image_type = SG_IMAGETYPE_2D;
			shd.views[2].texture.sample_type = SG_IMAGESAMPLETYPE_UNFILTERABLE_FLOAT;

			shd.views[3].texture.stage = SG_SHADERSTAGE_FRAGMENT;
			shd.views[3].texture.image_type = SG_IMAGETYPE_2D;
			shd.views[3].texture.sample_type = SG_IMAGESAMPLETYPE_UNFILTERABLE_FLOAT;

			shd.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
			shd.samplers[0].sampler_type = SG_SAMPLERTYPE_NONFILTERING;

			shd.texture_sampler_pairs[0].stage = SG_SHADERSTAGE_FRAGMENT;
			shd.texture_sampler_pairs[0].view_slot = 0;
			shd.texture_sampler_pairs[0].sampler_slot = 0;
			shd.texture_sampler_pairs[0].glsl_name = "light_data_tex";

			shd.texture_sampler_pairs[1].stage = SG_SHADERSTAGE_FRAGMENT;
			shd.texture_sampler_pairs[1].view_slot = 1;
			shd.texture_sampler_pairs[1].sampler_slot = 0;
			shd.texture_sampler_pairs[1].glsl_name = "tile_data_tex";

			shd.texture_sampler_pairs[2].stage = SG_SHADERSTAGE_FRAGMENT;
			shd.texture_sampler_pairs[2].view_slot = 2;
			shd.texture_sampler_pairs[2].sampler_slot = 0;
			shd.texture_sampler_pairs[2].glsl_name = "shadow_map_tex";

			shd.texture_sampler_pairs[3].stage = SG_SHADERSTAGE_FRAGMENT;
			shd.texture_sampler_pairs[3].view_slot = 3;
			shd.texture_sampler_pairs[3].sampler_slot = 0;
			shd.texture_sampler_pairs[3].glsl_name = "ssao_tex";

			sg_pipeline_desc pip = {};
			pip.shader = sg_make_shader(&shd);
			pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3;
			pip.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT3;
			pip.index_type = SG_INDEXTYPE_UINT16;
			pip.cull_mode = SG_CULLMODE_BACK;
			pip.depth.write_enabled = false;
			pip.depth.compare = SG_COMPAREFUNC_LESS_EQUAL;
			pip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA16F;
			pip.depth.pixel_format = SG_PIXELFORMAT_DEPTH;
			pip.sample_count = 1;
			color_pipeline_ = sg_make_pipeline(&pip);
		}

		tiles.init();
		grid.init();
		shadows.init();
		ao.init();
		hdr.init();
	}

	void destroy() {
		tiles.destroy();
		grid.destroy();
		shadows.destroy();
		ao.destroy();
		hdr.destroy();
		if (depth_pipeline_.id)
			sg_destroy_pipeline(depth_pipeline_);
		if (color_pipeline_.id)
			sg_destroy_pipeline(color_pipeline_);
	}

	void draw(world &w) {
		const int sw = sapp_width();
		const int sh = sapp_height();

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

		light collected[FP_MAX_LIGHTS];
		int num_lights = 0;
		w.for_each_entity<transform, light>([&](transform &t, light &l) {
			if (num_lights >= FP_MAX_LIGHTS)
				return;
			if (l.type != light_type::directional)
				l.position = t.position;
			collected[num_lights++] = l;
		});

		tiles.cull(view, proj, collected, num_lights, sw, sh);

		render_shadows(w, vp);
		render_ssao(w, view, vp, proj, sw, sh);
		render_scene(w, vp, cam_pos, sw, sh);
		hdr.composite_to_swapchain();
		sg_commit();
	}

private:
	sg_pipeline depth_pipeline_{};
	sg_pipeline color_pipeline_{};

	struct vs_uniforms {
		math::mat4 mvp;
		math::mat4 model;
		math::mat4 normal_mat;
		math::mat4 light_vp;
	};

	struct fs_uniforms {
		float material[4]{};
		float albedo[4]{};
		float camera_pos[4]{};
		float ambient[4]{};
		float tile_info[4]{};
		float shadow_params[4]{};
	};

	void render_shadows(world &w, const math::mat4 &vp) {
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

	void render_scene(world &w, const math::mat4 &vp, const math::vec3 &cam_pos, int sw, int sh) {
		hdr.ensure_targets(sw, sh);

		sg_pass pass = {};
		pass.action = hdr.scene_pass_action();
		pass.attachments = hdr.scene_attachments();
		sg_begin_pass(pass);

		sg_apply_pipeline(depth_pipeline_);
		w.for_each_entity<transform, mesh_instance, material>(
		    [&](transform &t, mesh_instance &mesh, material &) {
			    math::mat4 mvp = vp * t.matrix();
			    sg_apply_bindings(&mesh.bind);
			    sg_apply_uniforms(0, SG_RANGE(mvp));
			    sg_draw(0, mesh.index_count, 1);
		    });

		sg_apply_pipeline(color_pipeline_);
		w.for_each_entity<transform, mesh_instance, material>(
		    [&](transform &t, mesh_instance &mesh, material &mat) {
			    math::mat4 model = t.matrix();
			    math::mat4 normal_mat = t.rotation.to_mat4() *
			        math::mat4::scale({1.0f / t.scale.x, 1.0f / t.scale.y, 1.0f / t.scale.z});

			    vs_uniforms vs{};
			    vs.mvp = vp * model;
			    vs.model = model;
			    vs.normal_mat = normal_mat;
			    vs.light_vp = shadows.light_vp();

			    fs_uniforms fs{};
			    fs.material[0] = mat.metallic;
			    fs.material[1] = mat.roughness;
			    fs.material[2] = mat.ao;
			    fs.albedo[0] = mat.albedo.x;
			    fs.albedo[1] = mat.albedo.y;
			    fs.albedo[2] = mat.albedo.z;
			    fs.albedo[3] = mat.alpha;
			    fs.camera_pos[0] = cam_pos.x;
			    fs.camera_pos[1] = cam_pos.y;
			    fs.camera_pos[2] = cam_pos.z;
			    fs.camera_pos[3] = static_cast<float>(tiles.num_lights());
			    fs.ambient[0] = 0.03f;
			    fs.ambient[1] = 0.03f;
			    fs.ambient[2] = 0.03f;
			    fs.tile_info[0] = static_cast<float>(tiles.num_tiles_x());
			    fs.tile_info[1] = static_cast<float>(FP_TILE_SIZE);
			    fs.tile_info[2] = static_cast<float>(sw);
			    fs.tile_info[3] = static_cast<float>(sh);
			    fs.shadow_params[0] = static_cast<float>(shadows.resolution());
			    fs.shadow_params[1] = 0.0002f;
			    fs.shadow_params[2] = 1.0f;
			    fs.shadow_params[3] = 0.003f;

			    sg_bindings bind = {};
			    bind.vertex_buffers[0] = mesh.bind.vertex_buffers[0];
			    bind.index_buffer = mesh.bind.index_buffer;
			    bind.views[0] = tiles.light_data_view();
			    bind.views[1] = tiles.tile_data_view();
			    bind.views[2] = shadows.shadow_view();
			    bind.views[3] = ao.result_view();
			    bind.samplers[0] = tiles.sampler();

			    sg_apply_bindings(&bind);
			    sg_apply_uniforms(0, SG_RANGE(vs));
			    sg_apply_uniforms(1, SG_RANGE(fs));
			    sg_draw(0, mesh.index_count, 1);
		    });

		grid.draw(vp);
		sg_end_pass();
	}
};

} // namespace scene
