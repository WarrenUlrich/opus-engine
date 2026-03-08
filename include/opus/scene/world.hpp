#pragma once

#include "../ecs/ecs.hpp"
#include "camera.hpp"
#include "entity_name.hpp"
#include "lighting.hpp"
#include "material.hpp"
#include "mesh_instance.hpp"
#include "script_ref.hpp"
#include "transform.hpp"

namespace scene {

using world = ecs::context<transform, camera, mesh_instance, material, light, entity_name, script_ref>;

} // namespace scene
