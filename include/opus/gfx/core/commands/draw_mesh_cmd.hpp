#pragma once

#include <GL/glew.h>

#include <memory>

#include "../mesh.hpp"

namespace gfx {
class draw_mesh_cmd {
public:
  draw_mesh_cmd(gfx::mesh &mesh) : _mesh{mesh} {}

  constexpr std::string_view name() const noexcept { return "set_uniform"; }
  
  void execute() {
    _mesh.buffer.bind();

    glDrawArrays(GL_TRIANGLES, 0, _mesh.vertex_count);

    _mesh.buffer.unbind();
  }

private:
  gfx::mesh &_mesh;
};

} // namespace gfx