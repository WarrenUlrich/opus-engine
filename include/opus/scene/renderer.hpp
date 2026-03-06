#pragma once

#include "../ecs/ecs.hpp"
#include "../math/mat4.hpp"
#include "camera.hpp"
#include "debug_grid.hpp"
#include "light_grid.hpp"
#include "lighting.hpp"
#include "mesh_instance.hpp"
#include "transform.hpp"

namespace scene {

// ─── World Type ──────────────────────────────────────────────────────────────
//
// The canonical ECS context for a scene. Register every component type here.
// Adding a new component type is a one-line change — just append to this list.

using world = ecs::context<transform, camera, mesh_instance, material, light>;

// ─── Forward Renderer (legacy, 8-light maximum) ─────────────────────────────
//
// Single-pass forward PBR rendering system that queries the ECS each frame:
//   1. Finds the first entity with (transform, camera)  → view/projection
//   2. Collects all entities with (transform, light)    → lighting_environment
//   3. Draws all entities with (transform, mesh, material) via PBR pipeline
//
// Owns only GPU pipeline state — no scene data.

class forward_renderer {
public:
	sg_pipeline pipeline{};
	sg_pass_action pass_action{};
	lighting_environment env;
	debug_grid grid;

	forward_renderer() = default;

	/// One-time GPU pipeline setup. Call after sg_setup().
	void init() {
		static const unsigned char vs_src[] = {
#embed "forward.vert"
		    , 0};

		static const unsigned char fs_src[] = {
#embed "forward.frag"
		    , 0};

		sg_shader_desc shd = {};
		shd.vertex_func.source = reinterpret_cast<const char *>(vs_src);
		shd.fragment_func.source = reinterpret_cast<const char *>(fs_src);

		// Block 0 — Vertex: MVP + Model + Normal matrix
		shd.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
		shd.uniform_blocks[0].size = sizeof(vs_uniforms);
		shd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_MAT4, 0, "mvp"};
		shd.uniform_blocks[0].glsl_uniforms[1] = {SG_UNIFORMTYPE_MAT4, 0, "model"};
		shd.uniform_blocks[0].glsl_uniforms[2] = {SG_UNIFORMTYPE_MAT4, 0, "normal_mat"};

		// Block 1 — Fragment: Lighting environment (packed gpu_data)
		shd.uniform_blocks[1].stage = SG_SHADERSTAGE_FRAGMENT;
		shd.uniform_blocks[1].size = sizeof(lighting_environment::gpu_data);
		shd.uniform_blocks[1].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, 0, "material"};
		shd.uniform_blocks[1].glsl_uniforms[1] = {SG_UNIFORMTYPE_FLOAT4, 0, "albedo"};
		shd.uniform_blocks[1].glsl_uniforms[2] = {SG_UNIFORMTYPE_FLOAT4, 0, "camera_pos"};
		shd.uniform_blocks[1].glsl_uniforms[3] = {SG_UNIFORMTYPE_FLOAT4, 0, "ambient"};
		shd.uniform_blocks[1].glsl_uniforms[4] = {SG_UNIFORMTYPE_FLOAT4, MAX_LIGHTS, "light_pos_type"};
		shd.uniform_blocks[1].glsl_uniforms[5] = {SG_UNIFORMTYPE_FLOAT4, MAX_LIGHTS, "light_dir_range"};
		shd.uniform_blocks[1].glsl_uniforms[6] = {SG_UNIFORMTYPE_FLOAT4, MAX_LIGHTS, "light_color_intensity"};
		shd.uniform_blocks[1].glsl_uniforms[7] = {SG_UNIFORMTYPE_FLOAT4, MAX_LIGHTS, "light_params"};

		sg_pipeline_desc pip = {};
		pip.shader = sg_make_shader(&shd);
		pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3; // position
		pip.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT3; // normal
		pip.index_type = SG_INDEXTYPE_UINT16;
		pip.cull_mode = SG_CULLMODE_BACK;
		pip.depth.write_enabled = true;
		pip.depth.compare = SG_COMPAREFUNC_LESS_EQUAL;
		pipeline = sg_make_pipeline(&pip);

		pass_action.colors[0].load_action = SG_LOADACTION_CLEAR;
		pass_action.colors[0].clear_value = {0.01f, 0.01f, 0.02f, 1.0f};

		grid.init();
	}

	/// Render one frame by querying the ECS world.
	void draw(world &w) {
		// ── 1. Find the active camera ──
		math::mat4 vp = math::mat4::identity();
		math::vec3 cam_pos{};

		w.for_each_entity<transform, camera>([&](transform &t, camera &cam) {
			cam.aspect_ratio = sapp_widthf() / sapp_heightf();

			// Derive view from the ECS transform
			cam_pos = t.position;
			math::mat4 view = math::mat4::look_at(t.position, t.position + cam.forward, cam.up);
			vp = cam.projection_matrix() * view;
		});

		// ── 2. Collect lights from ECS into the environment ──
		env.clear();
		w.for_each_entity<transform, light>([&](transform &t, light &l) {
			// Sync the light's position from the ECS transform for point/spot
			if (l.type != light_type::directional) {
				l.position = t.position;
			}
			env.add(l);
		});

		// ── 3. Draw all renderable entities ──
		sg_pass pass = {};
		pass.action = pass_action;
		pass.swapchain = sglue_swapchain();

		sg_begin_pass(pass);
		sg_apply_pipeline(pipeline);

		w.for_each_entity<transform, mesh_instance, material>(
		    [&](transform &t, mesh_instance &mesh, material &mat) {
			    math::mat4 model_mat = t.matrix();
			    math::mat4 normal_mat = t.rotation.to_mat4();

			    vs_uniforms vs{};
			    vs.mvp = vp * model_mat;
			    vs.model = model_mat;
			    vs.normal_mat = normal_mat;

			    auto fs = env.pack(cam_pos, mat);

			    sg_apply_bindings(&mesh.bind);
			    sg_apply_uniforms(0, SG_RANGE(vs));
			    sg_apply_uniforms(1, SG_RANGE(fs));
			    sg_draw(0, mesh.index_count, 1);
		    });

		// ── 4. Debug grid overlay ──
		grid.draw(vp);

		sg_end_pass();
		sg_commit();
	}

private:
	struct vs_uniforms {
		math::mat4 mvp;
		math::mat4 model;
		math::mat4 normal_mat;
	};
};

// ─── Forward+ Renderer (tiled, up to 1024 lights) ───────────────────────────
//
// Two-pass tiled forward PBR renderer:
//
//   Pass 1 — Depth prepass:
//     Writes depth only (no color). Populates the depth buffer so
//     the GPU's early-Z hardware can reject occluded fragments in pass 2.
//
//   Pass 2 — Color pass:
//     Fragment shader determines which screen tile it belongs to,
//     reads the tile's light list from a texture (CPU-populated),
//     and evaluates only those lights.
//
//   CPU (between query & draw):
//     Projects each light's bounding sphere to screen space,
//     bins lights into 16×16 pixel tiles, and uploads the
//     per-tile lists + light properties as GPU textures.
//
// This scales to 1000+ lights because each fragment pays only for
// the lights whose range overlaps its tile (~5-20 on average).

class forward_plus_renderer {
public:
	sg_pipeline depth_pipeline{};
	sg_pipeline color_pipeline{};
	sg_pass_action pass_action{};
	light_grid tiles;
	debug_grid grid;

	forward_plus_renderer() = default;

	/// One-time GPU pipeline setup.  Call after sg_setup().
	void init() {
		// ── Depth prepass shader (inline — trivially small) ──────────
		const char *depth_vs =
		    "#version 330\n"
		    "layout(location=0) in vec3 position;\n"
		    "layout(location=1) in vec3 normal;\n"
		    "uniform mat4 mvp;\n"
		    "void main() { gl_Position = mvp * vec4(position, 1.0); }\n";
		const char *depth_fs =
		    "#version 330\n"
		    "out vec4 frag_color;\n"
		    "void main() { frag_color = vec4(0.0); }\n";

		{
			sg_shader_desc shd = {};
			shd.vertex_func.source = depth_vs;
			shd.fragment_func.source = depth_fs;
			shd.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
			shd.uniform_blocks[0].size = sizeof(math::mat4);
			shd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_MAT4, 0, "mvp"};

			sg_pipeline_desc pip = {};
			pip.shader = sg_make_shader(&shd);
			pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3; // position
			pip.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT3; // normal (ignored but in layout)
			pip.index_type = SG_INDEXTYPE_UINT16;
			pip.cull_mode = SG_CULLMODE_BACK;
			pip.depth.write_enabled = true;
			pip.depth.compare = SG_COMPAREFUNC_LESS_EQUAL;
			pip.colors[0].write_mask = SG_COLORMASK_NONE;
			depth_pipeline = sg_make_pipeline(&pip);
		}

		// ── Color pass shader (Forward+ PBR from file) ──────────────
		static const unsigned char vs_src[] = {
#embed "forward_plus.vert"
		    , 0};

		static const unsigned char fs_src[] = {
#embed "forward_plus.frag"
		    , 0};

		{
			sg_shader_desc shd = {};
			shd.vertex_func.source = reinterpret_cast<const char *>(vs_src);
			shd.fragment_func.source = reinterpret_cast<const char *>(fs_src);

			// UB 0 — Vertex: MVP + Model + Normal matrix
			shd.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
			shd.uniform_blocks[0].size = sizeof(vs_uniforms);
			shd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_MAT4, 0, "mvp"};
			shd.uniform_blocks[0].glsl_uniforms[1] = {SG_UNIFORMTYPE_MAT4, 0, "model"};
			shd.uniform_blocks[0].glsl_uniforms[2] = {SG_UNIFORMTYPE_MAT4, 0, "normal_mat"};

			// UB 1 — Fragment: material + camera + tile info
			shd.uniform_blocks[1].stage = SG_SHADERSTAGE_FRAGMENT;
			shd.uniform_blocks[1].size = sizeof(fs_uniforms);
			shd.uniform_blocks[1].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, 0, "material"};
			shd.uniform_blocks[1].glsl_uniforms[1] = {SG_UNIFORMTYPE_FLOAT4, 0, "albedo"};
			shd.uniform_blocks[1].glsl_uniforms[2] = {SG_UNIFORMTYPE_FLOAT4, 0, "camera_pos"};
			shd.uniform_blocks[1].glsl_uniforms[3] = {SG_UNIFORMTYPE_FLOAT4, 0, "ambient"};
			shd.uniform_blocks[1].glsl_uniforms[4] = {SG_UNIFORMTYPE_FLOAT4, 0, "tile_info"};

			// View 0 — Fragment: Light data texture (RGBA32F)
			shd.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
			shd.views[0].texture.image_type = SG_IMAGETYPE_2D;
			shd.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_UNFILTERABLE_FLOAT;

			// View 1 — Fragment: Tile data texture (RGBA32F)
			shd.views[1].texture.stage = SG_SHADERSTAGE_FRAGMENT;
			shd.views[1].texture.image_type = SG_IMAGETYPE_2D;
			shd.views[1].texture.sample_type = SG_IMAGESAMPLETYPE_UNFILTERABLE_FLOAT;

			// Sampler 0 — Nearest (required for unfilterable-float)
			shd.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
			shd.samplers[0].sampler_type = SG_SAMPLERTYPE_NONFILTERING;

			// Texture-sampler pairs (bind textures to GLSL sampler2D names)
			shd.texture_sampler_pairs[0].stage = SG_SHADERSTAGE_FRAGMENT;
			shd.texture_sampler_pairs[0].view_slot = 0;
			shd.texture_sampler_pairs[0].sampler_slot = 0;
			shd.texture_sampler_pairs[0].glsl_name = "light_data_tex";

			shd.texture_sampler_pairs[1].stage = SG_SHADERSTAGE_FRAGMENT;
			shd.texture_sampler_pairs[1].view_slot = 1;
			shd.texture_sampler_pairs[1].sampler_slot = 0;
			shd.texture_sampler_pairs[1].glsl_name = "tile_data_tex";

			sg_pipeline_desc pip = {};
			pip.shader = sg_make_shader(&shd);
			pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3; // position
			pip.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT3; // normal
			pip.index_type = SG_INDEXTYPE_UINT16;
			pip.cull_mode = SG_CULLMODE_BACK;
			pip.depth.write_enabled = false;                       // already written by depth pass
			pip.depth.compare = SG_COMPAREFUNC_LESS_EQUAL;
			color_pipeline = sg_make_pipeline(&pip);
		}

		// ── Pass action ──
		pass_action.colors[0].load_action = SG_LOADACTION_CLEAR;
		pass_action.colors[0].clear_value = {0.01f, 0.01f, 0.02f, 1.0f};

		// ── Sub-systems ──
		tiles.init();
		grid.init();
	}

	/// Render one frame.  Queries the ECS world for cameras, lights,
	/// and renderable entities.
	void draw(world &w) {
		const int screen_w = sapp_width();
		const int screen_h = sapp_height();

		// ── 1. Find camera ──────────────────────────────────────────
		math::mat4 view_mat = math::mat4::identity();
		math::mat4 proj_mat = math::mat4::identity();
		math::mat4 vp = math::mat4::identity();
		math::vec3 cam_pos{};

		w.for_each_entity<transform, camera>([&](transform &t, camera &cam) {
			cam.aspect_ratio = sapp_widthf() / sapp_heightf();
			cam_pos = t.position;
			view_mat = math::mat4::look_at(t.position, t.position + cam.forward, cam.up);
			proj_mat = cam.projection_matrix();
			vp = proj_mat * view_mat;
		});

		// ── 2. Collect lights from ECS ──────────────────────────────
		light collected[FP_MAX_LIGHTS];
		int num_lights = 0;

		w.for_each_entity<transform, light>([&](transform &t, light &l) {
			if (num_lights >= FP_MAX_LIGHTS) return;

			// Sync position from ECS transform
			if (l.type != light_type::directional)
				l.position = t.position;

			collected[num_lights++] = l;
		});

		// ── 3. CPU tile culling ─────────────────────────────────────
		tiles.cull(view_mat, proj_mat, collected, num_lights, screen_w, screen_h);

		// ── 4. Begin render pass ────────────────────────────────────
		sg_pass pass = {};
		pass.action = pass_action;
		pass.swapchain = sglue_swapchain();
		sg_begin_pass(pass);

		// ── 5. Depth prepass (writes depth, no color) ───────────────
		sg_apply_pipeline(depth_pipeline);

		w.for_each_entity<transform, mesh_instance, material>(
		    [&](transform &t, mesh_instance &mesh, material &) {
			    math::mat4 mvp = vp * t.matrix();
			    sg_apply_bindings(&mesh.bind);
			    sg_apply_uniforms(0, SG_RANGE(mvp));
			    sg_draw(0, mesh.index_count, 1);
		    });

		// ── 6. Color pass (Forward+ PBR) ────────────────────────────
		sg_apply_pipeline(color_pipeline);

		w.for_each_entity<transform, mesh_instance, material>(
		    [&](transform &t, mesh_instance &mesh, material &mat) {
			    math::mat4 model_mat = t.matrix();
			    math::mat4 normal_mat = t.rotation.to_mat4();

			    vs_uniforms vs{};
			    vs.mvp = vp * model_mat;
			    vs.model = model_mat;
			    vs.normal_mat = normal_mat;

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

			    // Bind mesh buffers + light textures + sampler
			    sg_bindings bind = {};
			    bind.vertex_buffers[0] = mesh.bind.vertex_buffers[0];
			    bind.index_buffer = mesh.bind.index_buffer;
			    bind.views[0] = tiles.light_data_view();
			    bind.views[1] = tiles.tile_data_view();
			    bind.samplers[0] = tiles.sampler();

			    sg_apply_bindings(&bind);
			    sg_apply_uniforms(0, SG_RANGE(vs));
			    sg_apply_uniforms(1, SG_RANGE(fs));
			    sg_draw(0, mesh.index_count, 1);
		    });

		// ── 7. Debug grid overlay ───────────────────────────────────
		grid.draw(vp);

		sg_end_pass();
		sg_commit();
	}

private:
	struct vs_uniforms {
		math::mat4 mvp;
		math::mat4 model;
		math::mat4 normal_mat;
	};

	struct fs_uniforms {
		float material[4]{};     // metallic, roughness, ao, 0
		float albedo[4]{};       // r, g, b, alpha
		float camera_pos[4]{};   // x, y, z, num_lights
		float ambient[4]{};      // r, g, b, 0
		float tile_info[4]{};    // num_tiles_x, tile_size, 0, 0
	};
};

} // namespace scene
