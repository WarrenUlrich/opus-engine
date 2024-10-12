#pragma once

#include <GL/glew.h>

#include <memory>

#include "../../core/commands/draw_mesh_cmd.hpp"
#include "../../core/mesh.hpp"

#include "../opengl_vbuffer.hpp"

namespace gfx {
template <> class draw_mesh_cmd<render_backend::opengl> {
public:
  using mesh_type = gfx::mesh<render_backend::opengl>;

  draw_mesh_cmd(mesh_type &mesh) : _mesh{mesh} {}

  void execute() {
    _mesh.buffer.bind();

    glDrawArrays(GL_TRIANGLES, 0, _mesh.vertex_count);

    _mesh.buffer.unbind();
  }

private:
  mesh_type &_mesh;
};

} // namespace gfx