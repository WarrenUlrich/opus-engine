#pragma once

// C API exported by the hot-reloadable game DLL.
// The engine calls these to discover and instantiate scripts.

#ifdef __cplusplus
extern "C" {
#endif

typedef void *(*opus_script_create_fn)(void);

typedef struct opus_script_desc {
	const char *name;
	opus_script_create_fn create;
} opus_script_desc;

// Returns how many scripts are registered in this DLL.
int opus_script_count(void);

// Returns a pointer to a static array of opus_script_desc entries.
opus_script_desc *opus_script_list(void);

#ifdef __cplusplus
}
#endif
