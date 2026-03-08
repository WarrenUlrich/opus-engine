#pragma once

#include <opus/scripting/script.hpp>

// Rotates the attached entity around the Y axis every frame.
class rotate_script : public scripting::script {
public:
	float speed{0.1f}; // radians per second

	void on_update(scripting::script_context &ctx) override {
		if (auto *t = ctx.world->try_get_component<scene::transform>(ctx.entity)) {
			t->rotation =
			    math::quat::angle_axis(speed * ctx.dt, {0.0f, 1.0f, 0.0f}) * t->rotation;
		}
	}
};
