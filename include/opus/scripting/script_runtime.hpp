#pragma once

#include "../scene/world.hpp"
#include "game_api.h"
#include "script.hpp"

#include <cstring>
#include <dlfcn.h>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace scripting {

// Loads the game shared library, watches it for changes, and manages
// the lifecycle of all active script instances attached to entities.
class script_runtime {
public:
	script_runtime() = default;
	~script_runtime() { unload(); }

	// Initial load of the game DLL.  Returns true on success.
	bool load(const std::string &path) {
		lib_path_ = path;
		return reload_lib();
	}

	// Poll the DLL file for modifications.  If a change is detected (with a
	// small grace period so the compiler finishes writing), the DLL is reloaded
	// and any running scripts are torn down and re-created.
	// Returns true when a hot-reload actually happened.
	bool hot_reload(scene::world &world, bool is_playing, float time) {
		if (!check_file_changed()) return false;

		// Tear down existing script instances
		destroy_all_scripts(world);

		// Reload the shared library
		if (!reload_lib()) return false;

		// If the game was playing, recreate scripts and call on_start
		if (is_playing) {
			start_all_scripts(world, time);
		}

		return true;
	}

	// --- Play-mode lifecycle -----------------------------------------------

	void start_all_scripts(scene::world &world, float time) {
		world.for_each_entity<scene::script_ref>([&](ecs::entity_id id, scene::script_ref &sr) {
			if (sr.script_name[0] == '\0') return;
			if (sr.instance) return; // already running

			sr.instance = create_script(sr.script_name);
			if (sr.instance) {
				script_context ctx{id, &world, 0.0f, time};
				sr.instance->on_start(ctx);
				sr.started = true;
			}
		});
	}

	void update_all_scripts(scene::world &world, float dt, float time) {
		world.for_each_entity<scene::script_ref>([&](ecs::entity_id id, scene::script_ref &sr) {
			if (sr.instance && sr.started) {
				script_context ctx{id, &world, dt, time};
				sr.instance->on_update(ctx);
			}
		});
	}

	void destroy_all_scripts(scene::world &world) {
		world.for_each_entity<scene::script_ref>([&](ecs::entity_id, scene::script_ref &sr) {
			if (sr.instance) {
				delete sr.instance;
				sr.instance = nullptr;
				sr.started = false;
			}
		});
	}

	// --- Query -------------------------------------------------------------

	script *create_script(const char *name) {
		if (!lib_handle_) return nullptr;

		auto count_fn = reinterpret_cast<int (*)()>(dlsym(lib_handle_, "opus_script_count"));
		auto list_fn =
		    reinterpret_cast<opus_script_desc *(*)()>(dlsym(lib_handle_, "opus_script_list"));
		if (!count_fn || !list_fn) return nullptr;

		int count = count_fn();
		opus_script_desc *descs = list_fn();

		for (int i = 0; i < count; ++i) {
			if (std::strcmp(descs[i].name, name) == 0) {
				return static_cast<script *>(descs[i].create());
			}
		}
		return nullptr;
	}

	std::vector<std::string> script_names() const {
		std::vector<std::string> names;
		if (!lib_handle_) return names;

		auto count_fn = reinterpret_cast<int (*)()>(dlsym(lib_handle_, "opus_script_count"));
		auto list_fn =
		    reinterpret_cast<opus_script_desc *(*)()>(dlsym(lib_handle_, "opus_script_list"));
		if (!count_fn || !list_fn) return names;

		int count = count_fn();
		opus_script_desc *descs = list_fn();

		for (int i = 0; i < count; ++i) {
			names.emplace_back(descs[i].name);
		}
		return names;
	}

	[[nodiscard]] bool is_loaded() const { return lib_handle_ != nullptr; }
	[[nodiscard]] const std::string &lib_path() const { return lib_path_; }

private:
	void *lib_handle_{nullptr};
	std::string lib_path_;
	time_t last_mtime_{0};
	time_t pending_mtime_{0};
	int pending_frames_{0};

	// Check whether the .so file's modification time has changed.
	// Uses a small frame-count grace period so we don't try to load
	// a partially-written file.
	bool check_file_changed() {
		if (lib_path_.empty()) return false;

		struct stat st {};
		if (stat(lib_path_.c_str(), &st) != 0) return false;

		auto mtime = st.st_mtime;
		if (mtime == last_mtime_) return false;

		if (mtime != pending_mtime_) {
			pending_mtime_ = mtime;
			pending_frames_ = 0;
			return false;
		}

		++pending_frames_;
		if (pending_frames_ < 30) return false; // ~0.5 s at 60 fps

		last_mtime_ = mtime;
		pending_mtime_ = 0;
		pending_frames_ = 0;
		return true;
	}

	bool reload_lib() {
		unload();
		lib_handle_ = dlopen(lib_path_.c_str(), RTLD_NOW);
		return lib_handle_ != nullptr;
	}

	void unload() {
		if (lib_handle_) {
			dlclose(lib_handle_);
			lib_handle_ = nullptr;
		}
	}
};

} // namespace scripting
