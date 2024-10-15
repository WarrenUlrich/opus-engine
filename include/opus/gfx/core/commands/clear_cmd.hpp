#pragma once

#include <GL/glew.h>

#include <memory>

namespace gfx {
class clear_cmd {
public:
  clear_cmd(GLbitfield mask) : _mask(mask) {}

  constexpr std::string_view name() const noexcept { return "clear"; }

  void execute() { glClear(_mask); }

private:
  GLbitfield _mask;
};

} // namespace gfx