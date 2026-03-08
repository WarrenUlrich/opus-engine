// Game DLL script registry.
// Add new scripts here and rebuild the shared library; the editor will
// hot-reload them automatically.

#include "scripts/bounce_script.hpp"
#include "scripts/rotate_script.hpp"

#include <opus/scripting/game_api.h>

static opus_script_desc s_scripts[] = {
    {"Rotate", []() -> void * { return new rotate_script(); }},
    {"Bounce", []() -> void * { return new bounce_script(); }},
};

extern "C" int opus_script_count() {
	return static_cast<int>(sizeof(s_scripts) / sizeof(s_scripts[0]));
}

extern "C" opus_script_desc *opus_script_list() {
	return s_scripts;
}
