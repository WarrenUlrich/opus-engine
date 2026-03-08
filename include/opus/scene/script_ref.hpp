#pragma once

#include <cstring>

namespace scripting {
class script;
}

namespace scene {

// Lightweight component that binds an entity to a named script from the game DLL.
// The `instance` pointer is managed by the script_runtime (created on play, destroyed on stop).
struct script_ref {
	char script_name[64]{};
	scripting::script *instance{nullptr};
	bool started{false};

	script_ref() = default;
	explicit script_ref(const char *name) {
		std::strncpy(script_name, name, sizeof(script_name) - 1);
		script_name[sizeof(script_name) - 1] = '\0';
	}
};

} // namespace scene
