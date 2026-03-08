#pragma once

#include <opus/scripting/script.hpp>
#include <cmath>

// Bobs the entity up and down along the Y axis using a sine wave.
class bounce_script : public scripting::script {
public:
	float amplitude{0.1f};
	float frequency{2.0f};

	void on_start(scripting::script_context &ctx) override {
		if (auto *t = ctx.world->try_get_component<scene::transform>(ctx.entity)) {
			base_y_ = t->position.y;
		}
	}

	void on_update(scripting::script_context &ctx) override {
		if (auto *t = ctx.world->try_get_component<scene::transform>(ctx.entity)) {
			t->position.y = base_y_ + amplitude * std::sin(ctx.time * frequency);
		}
	}

private:
	float base_y_{0.0f};
};
