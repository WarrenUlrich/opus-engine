#pragma once

#include <cstring>

namespace scene {

struct entity_name {
	char name[64]{"Entity"};

	entity_name() = default;
	explicit entity_name(const char *n) {
		std::strncpy(name, n, sizeof(name) - 1);
		name[sizeof(name) - 1] = '\0';
	}
};

} // namespace scene
