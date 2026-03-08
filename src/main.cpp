// sokol declarations only – implementations live in sokol_imgui_impl.cpp
#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"

#ifdef OPUS_EDITOR
#include "imgui.h"
#include "sokol_imgui.h"
#include <opus/editor/editor.hpp>
#endif

#include <opus/opus.hpp>

#include <cmath>

static struct {
	scene::world world;
	scene::deferred_renderer renderer;
	scripting::script_runtime runtime;
	float time{};
#ifdef OPUS_EDITOR
	editor::editor_layer editor;
#endif
} state;

void init(void) {
	sg_desc desc = {};
	desc.environment = sglue_environment();
	desc.logger.func = slog_func;
	sg_setup(&desc);

#ifdef OPUS_EDITOR
	simgui_desc_t simgui_desc = {};
	simgui_desc.logger.func = slog_func;
	simgui_setup(&simgui_desc);

	state.editor.init();
#endif

	state.renderer.init();

	auto &w = state.world;

	// Camera
	math::vec3 eye = {0.0f, 3.0f, 6.0f};
	math::vec3 target = {0.0f, 1.0f, 0.0f};
	auto cam = w.new_entity();
	w.add_component(cam, scene::entity_name{"Camera"});
	w.add_component(cam, scene::transform{.position = eye,
	                                      .rotation = math::quat::look_rotation(
	                                          (target - eye).normalized())});
	w.add_component(cam, scene::camera{});

	// Sun
	auto sun = w.new_entity();
	w.add_component(sun, scene::entity_name{"Sun"});
	w.add_component(sun, scene::transform{});
	w.add_component(sun,
	                scene::light::directional({-0.4f, -1.0f, -0.3f}, {1.0f, 0.97f, 0.92f}, 1.5f));

	// Floor
	auto floor_e = w.new_entity();
	w.add_component(floor_e, scene::entity_name{"Floor"});
	w.add_component(floor_e, scene::transform{.position = {0.0f, 0.0f, 0.0f}});
	w.add_component(floor_e, scene::mesh_instance::plane(20.0f, 20.0f, 1, 1));
	w.add_component(floor_e, scene::material::ceramic({0.2f, 0.2f, 0.22f}));

	// Torus
	auto torus = w.new_entity();
	w.add_component(torus, scene::entity_name{"Torus"});
	w.add_component(torus, scene::transform{.position = {0.0f, 1.2f, 0.0f}});
	w.add_component(torus, scene::mesh_instance::torus(1.0f, 0.4f, 64, 16));
	w.add_component(torus, scene::material::plastic());

	// 3 point lights
	constexpr float TAU = 3.14159265f * 2.0f;
	math::vec3 colors[] = {{1.0f, 0.3f, 0.2f}, {0.2f, 1.0f, 0.3f}, {0.2f, 0.4f, 1.0f}};
	for (int i = 0; i < 3; ++i) {
		float angle = TAU * i / 3.0f;
		char name[32];
		snprintf(name, sizeof(name), "Point Light %d", i);
		auto e = w.new_entity();
		w.add_component(e, scene::entity_name{name});
		w.add_component(
		    e, scene::transform{.position = {3.0f * std::cos(angle), 2.0f, 3.0f * std::sin(angle)}});
		w.add_component(e, scene::light::point({}, colors[i], 4.0f, 8.0f));
	}

	// Load the hot-reloadable game DLL
	if (state.runtime.load("./libopus-game.so")) {
#ifdef OPUS_EDITOR
		state.editor.log("Game DLL loaded successfully");
#endif
	} else {
#ifdef OPUS_EDITOR
		state.editor.log("Game DLL not found — build the opus-game target");
#endif
	}

#ifndef OPUS_EDITOR
	// Game mode: start scripts immediately
	state.runtime.start_all_scripts(w, state.time);
#endif
}

void frame(void) {
	float dt = (float)sapp_frame_duration();
	state.time += dt;
	auto &w = state.world;

#ifdef OPUS_EDITOR
	// ---- Play / Stop edge triggers from editor ----
	if (state.editor.play_requested) {
		state.editor.play_requested = false;
		state.editor.is_playing = true;
		state.runtime.start_all_scripts(w, state.time);
		state.editor.log("Play started");
	}
	if (state.editor.stop_requested) {
		state.editor.stop_requested = false;
		state.editor.is_playing = false;
		state.runtime.destroy_all_scripts(w);
		state.editor.log("Play stopped");
	}

	// ---- Hot-reload check (polls file mtime) ----
	if (state.runtime.hot_reload(w, state.editor.is_playing, state.time)) {
		state.editor.log("Game DLL hot-reloaded");
	}

	// ---- Run scripts ----
	if (state.editor.is_playing) {
		state.runtime.update_all_scripts(w, dt, state.time);
	}

	// ---- Render scene into offscreen texture ----
	state.renderer.draw(state.world, state.editor.viewport_width, state.editor.viewport_height);

	// ---- Editor UI ----
	simgui_frame_desc_t simgui_frame = {};
	simgui_frame.width = sapp_width();
	simgui_frame.height = sapp_height();
	simgui_frame.delta_time = sapp_frame_duration();
	simgui_frame.dpi_scale = sapp_dpi_scale();
	simgui_new_frame(&simgui_frame);

	auto scripts = state.runtime.script_names();
	state.editor.draw(state.renderer.final_color_view(), state.renderer.linear_sampler(),
	                  state.world, scripts);

	sg_pass pass = {};
	pass.swapchain = sglue_swapchain();
	pass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
	pass.action.colors[0].clear_value = {0.0f, 0.0f, 0.0f, 1.0f};
	sg_begin_pass(pass);
	simgui_render();
	sg_end_pass();
#else
	// ---- Game mode: hot-reload + run scripts every frame ----
	state.runtime.hot_reload(w, true, state.time);
	state.runtime.update_all_scripts(w, dt, state.time);

	// ---- Render scene directly to swapchain ----
	state.renderer.draw(state.world);
#endif

	sg_commit();
}

void cleanup(void) {
	state.runtime.destroy_all_scripts(state.world);
	state.world.for_each_entity<scene::mesh_instance>(
	    [](scene::mesh_instance &mesh) { mesh.destroy(); });
	state.renderer.destroy();
#ifdef OPUS_EDITOR
	simgui_shutdown();
#endif
	sg_shutdown();
}

void event(const sapp_event *ev) {
#ifdef OPUS_EDITOR
	simgui_handle_event(ev);
#else
	(void)ev;
#endif
}

sapp_desc sokol_main(int, char *[]) {
	sapp_desc desc = {};
	desc.init_cb = init;
	desc.frame_cb = frame;
	desc.cleanup_cb = cleanup;
	desc.event_cb = event;
	desc.width = 1280;
	desc.height = 800;
#ifdef OPUS_EDITOR
	desc.sample_count = 4;
	desc.window_title = "Opus Editor";
#else
	desc.sample_count = 1;
	desc.window_title = "Opus";
#endif
	desc.icon.sokol_default = true;
	desc.logger.func = slog_func;
	return desc;
}
