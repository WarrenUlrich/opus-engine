#pragma once

#include "../ecs/ecs.hpp"
#include "../scene/world.hpp"

namespace scripting {

// Passed to every script callback so scripts can inspect/mutate the world.
struct script_context {
	ecs::entity_id entity;
	scene::world *world;
	float dt;
	float time;
};

// Base class for all hot-reloadable scripts.
// Game DLL scripts derive from this and override the virtual callbacks.
class script {
public:
	virtual ~script() = default;

	// Called once when play mode starts (or when the script is first attached during play).
	virtual void on_start(script_context &ctx) { (void)ctx; }

	// Called every frame while play mode is active.
	virtual void on_update(script_context &ctx) { (void)ctx; }

	// Called when play mode stops, or when the script component is removed.
	virtual void on_destroy(script_context &ctx) { (void)ctx; }
};

} // namespace scripting
