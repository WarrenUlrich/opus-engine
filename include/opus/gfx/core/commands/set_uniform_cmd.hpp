#pragma once

#include <memory>

#include "../../core/commands/set_uniform_cmd.hpp"

#include "../shader_program.hpp"

namespace gfx {
template <typename Uniform> class set_uniform_cmd {
public:
  using uniform_type = Uniform;

  constexpr set_uniform_cmd(std::string_view name, const uniform_type &uniform,
                            shader_program &shader) noexcept
      : _name(name), _uniform(uniform), _shader(shader) {}

  constexpr set_uniform_cmd(const set_uniform_cmd &other)
      : set_uniform_cmd(other._name, other._uniform, other._shader) {}

  constexpr set_uniform_cmd(set_uniform_cmd &&other) noexcept
      : set_uniform_cmd(std::move(other._name), std::move(other._uniform),
                        other._shader) {}

  constexpr set_uniform_cmd &operator=(const set_uniform_cmd &other) {
    if (this != &other) {
      _name = other._name;
      _uniform = other._uniform;
      _shader = other._shader;
    }
    return *this;
  }

  constexpr set_uniform_cmd &operator=(set_uniform_cmd &&other) noexcept {
    if (this != &other) {
      _name = std::move(other._name);
      _uniform = std::move(other._uniform);
      _shader = other._shader;
    }
    return *this;
  }

  constexpr std::string_view name() const noexcept { return "set_uniform"; }

  bool execute() noexcept {
    _shader.set_uniform(_name, _uniform);
    return true;
  }

private:
  std::string_view _name;
  uniform_type _uniform;
  shader_program &_shader;
};

template <typename Uniform>
set_uniform_cmd(std::string_view, const Uniform &,
                shader_program &) -> set_uniform_cmd<Uniform>;

} // namespace gfx