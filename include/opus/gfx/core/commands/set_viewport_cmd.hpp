#pragma once

#include <GL/glew.h>

#include <memory>

#include "../frame_buffer.hpp"

namespace gfx {
class set_viewport_cmd {
public:
  set_viewport_cmd(GLint x, GLint y, GLsizei width, GLsizei height)
      : _x{x}, _y{y}, _width{width}, _height{height} {}

  constexpr std::string_view name() const noexcept { return "set_viewport"; }

  void execute() {
    glViewport(_x, _y, _width, _height);
  }

private:
  GLint _x;
  GLint _y;
  GLsizei _width;
  GLsizei _height;
};

} // namespace gfx