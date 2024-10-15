#pragma once

#include <GL/glew.h>

#include <memory>

#include "../frame_buffer.hpp"

namespace gfx {
class bind_framebuffer_cmd {
public:
  bind_framebuffer_cmd(gfx::frame_buffer *fb) : _fb(fb) {}

  bind_framebuffer_cmd(bind_framebuffer_cmd &&other) : _fb(other._fb) {}

  bind_framebuffer_cmd(const bind_framebuffer_cmd &other) = delete;

  ~bind_framebuffer_cmd() = default;

  bind_framebuffer_cmd &operator=(const bind_framebuffer_cmd &other) = delete;

  bind_framebuffer_cmd &operator=(bind_framebuffer_cmd &&other) {
    if (this != &other) {
      _fb = other._fb;
    }

    other._fb = nullptr;
    return *this;
  }

  constexpr std::string_view name() const noexcept {
    return "bind_framebuffer";
  }

  void execute() {
    if (_fb) {
      _fb->bind();
    }
  }

private:
  gfx::frame_buffer *_fb;
};

} // namespace gfx