#pragma once

#include "render_backend.hpp"

namespace gfx {
  // class renderer_context {
  // public:
  //   virtual bool init(const feature_set &features) noexcept = 0;

  //   virtual bool swap_buffers() noexcept = 0;

  //   virtual ~renderer_context() noexcept = default;
  // };

  template<render_backend Backend>
  class renderer_context;
}