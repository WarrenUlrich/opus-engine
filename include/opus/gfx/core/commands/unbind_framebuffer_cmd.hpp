#pragma once

#include <GL/glew.h>

#include <memory>

#include "../frame_buffer.hpp"

namespace gfx {
class unbind_framebuffer_cmd {
public:
  unbind_framebuffer_cmd() = default;

  constexpr std::string_view name() const noexcept {
    return "bind_framebuffer";
  }

  void execute() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }
};

} // namespace gfx