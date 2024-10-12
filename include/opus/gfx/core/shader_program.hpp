#pragma once

#include <string_view>

#include "../../math/vector2.hpp"
#include "../../math/vector3.hpp"
#include "../../math/matrix4x4.hpp"
#include "../../math/quaternion.hpp"

#include "render_backend.hpp"

namespace gfx {
  template<render_backend Backend>
  class shader_program;
}