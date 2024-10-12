#pragma once

#include "../render_backend.hpp"

namespace gfx {
  template<render_backend Backend, typename Uniform>
  class set_uniform_cmd;
}