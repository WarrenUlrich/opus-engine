#pragma once

#include <GL/glew.h>

#include <memory>

#include "../texture2d.hpp"

namespace gfx {
class bind_texture_cmd {
public:
  bind_texture_cmd(const gfx::texture2d &texture, GLenum unit) : _texture(texture), _unit(unit) {}

  constexpr std::string_view name() const noexcept {
    return "bind_texture";
  }

  void execute() { 
    _texture.bind(GL_TEXTURE0 + _unit);
  }

private:
  const gfx::texture2d &_texture;
  GLenum _unit;
};

} // namespace gfx