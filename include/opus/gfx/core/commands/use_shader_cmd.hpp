#pragma once

#include <GL/glew.h>

#include <memory>

#include "../shader_program.hpp"

namespace gfx {
class use_shader_cmd {
public:
  use_shader_cmd(gfx::shader_program &shader) : _shader{shader} {}

  constexpr std::string_view name() const noexcept { return "use_shader"; }

  void execute() { _shader.use(); }

private:
  gfx::shader_program &_shader;
};

} // namespace gfx