#define SOKOL_IMPL
#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"

#include <opus/opus.hpp>

#include <cmath>
#include <cstdlib>

static struct {
	scene::world world;
	scene::forward_plus_renderer renderer;
	float time{};

	ecs::entity_id cam_entity;
	ecs::entity_id sphere_gold;
	ecs::entity_id sphere_copper;
	ecs::entity_id sphere_plastic;
} state;

void init(void) {
	sg_desc desc = {};
	desc.environment = sglue_environment();
	desc.logger.func = slog_func;
	sg_setup(&desc);

	state.renderer.init();

	auto &w = state.world;

	math::vec3 eye = {0.0f, 3.0f, 12.0f};
	math::vec3 target = {0.0f, 0.0f, 0.0f};
	state.cam_entity = w.new_entity();
	w.add_component(state.cam_entity, scene::transform{
	    .position = eye,
	    .rotation = math::quat::look_rotation((target - eye).normalized())
	});
	w.add_component(state.cam_entity, scene::camera{});

	auto sun = w.new_entity();
	w.add_component(sun, scene::transform{});
	w.add_component(sun, scene::light::directional(
	                         {-0.5f, -1.0f, -0.3f}, {1.0f, 0.95f, 0.9f}, 1.5f));

	// 5x5 grid of point lights
	std::srand(42);
	for (int z = -2; z <= 2; ++z) {
		for (int x = -2; x <= 2; ++x) {
			auto e = w.new_entity();

			float px = x * 2.5f;
			float py = 0.5f + 0.5f * ((float)(std::rand() % 100) / 100.0f);
			float pz = z * 2.5f;
			w.add_component(e, scene::transform{.position = {px, py, pz}});

			float r = 0.3f + 0.7f * ((float)(std::rand() % 100) / 100.0f);
			float g = 0.3f + 0.7f * ((float)(std::rand() % 100) / 100.0f);
			float b = 0.3f + 0.7f * ((float)(std::rand() % 100) / 100.0f);

			w.add_component(e, scene::light::point({}, {r, g, b}, 2.5f, 5.0f));
		}
	}

	state.sphere_gold = w.new_entity();
	w.add_component(state.sphere_gold, scene::transform{.position = {0.0f, 0.8f, 0.0f}});
	w.add_component(state.sphere_gold, scene::mesh_instance::sphere(0.8f, 64, 32));
	w.add_component(state.sphere_gold, scene::material::gold());

	state.sphere_copper = w.new_entity();
	w.add_component(state.sphere_copper, scene::transform{.position = {3.0f, 0.8f, 0.0f}});
	w.add_component(state.sphere_copper, scene::mesh_instance::sphere(0.8f, 64, 32));
	w.add_component(state.sphere_copper, scene::material::copper());

	state.sphere_plastic = w.new_entity();
	w.add_component(state.sphere_plastic, scene::transform{.position = {-3.0f, 0.8f, 0.0f}});
	w.add_component(state.sphere_plastic, scene::mesh_instance::torus(0.6f, 0.25f, 64, 32));
	w.add_component(state.sphere_plastic, scene::material::plastic({0.9f, 0.1f, 0.1f}));

	auto silver = w.new_entity();
	w.add_component(silver, scene::transform{.position = {0.0f, 0.8f, -3.0f}});
	w.add_component(silver, scene::mesh_instance::sphere(0.8f, 64, 32));
	w.add_component(silver, scene::material::silver());

	auto ceramic = w.new_entity();
	w.add_component(ceramic, scene::transform{.position = {-2.0f, 0.8f, 3.0f}});
	w.add_component(ceramic, scene::mesh_instance::sphere(0.8f, 64, 32));
	w.add_component(ceramic, scene::material::ceramic());

	auto floor = w.new_entity();
	w.add_component(floor, scene::transform{.position = {0.0f, 0.0f, 0.0f}});
	w.add_component(floor, scene::mesh_instance::plane(30.0f, 30.0f));
	w.add_component(floor, scene::material::plastic({0.4f, 0.4f, 0.4f}));
}

void frame(void) {
	float dt = (float)sapp_frame_duration();
	state.time += dt;

	auto &w = state.world;

	auto spin = [&](ecs::entity_id id, float speed, math::vec3 axis) {
		if (auto *t = w.try_get_component<scene::transform>(id)) {
			t->rotation = math::quat::angle_axis(speed * dt, axis) * t->rotation;
		}
	};

	spin(state.sphere_gold, 0.4f, {0.0f, 1.0f, 0.0f});
	spin(state.sphere_copper, 0.6f, {1.0f, 1.0f, 0.0f});
	spin(state.sphere_plastic, 0.3f, {0.0f, 1.0f, 1.0f});

	if (auto *cam_t = w.try_get_component<scene::transform>(state.cam_entity)) {
		float r = 14.0f;
		cam_t->position = {
		    r * std::sin(state.time * 0.2f),
		    3.0f + 1.5f * std::sin(state.time * 0.15f),
		    r * std::cos(state.time * 0.2f),
		};
		math::vec3 target = {0.0f, 0.0f, 0.0f};
		cam_t->rotation = math::quat::look_rotation(
		    (target - cam_t->position).normalized());
	}

	state.renderer.draw(state.world);
}

void cleanup(void) {
	state.world.for_each_entity<scene::mesh_instance>([](scene::mesh_instance &mesh) {
		mesh.destroy();
	});
	state.renderer.destroy();
	sg_shutdown();
}

sapp_desc sokol_main(int argc, char *argv[]) {
	(void)argc;
	(void)argv;
	sapp_desc desc = {};
	desc.init_cb = init;
	desc.frame_cb = frame;
	desc.cleanup_cb = cleanup;
	desc.width = 1024;
	desc.height = 768;
	desc.sample_count = 4;
	desc.window_title = "Forward+ PBR";
	desc.icon.sokol_default = true;
	desc.logger.func = slog_func;
	return desc;
}
