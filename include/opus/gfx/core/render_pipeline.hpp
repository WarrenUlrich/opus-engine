#pragma once

#include <GL/glew.h>

#include <memory>
#include <vector>

#include "../../math/math.hpp"

#include "commands/bind_framebuffer_cmd.hpp"
#include "commands/clear_cmd.hpp"
#include "commands/draw_mesh_cmd.hpp"
#include "commands/set_uniform_cmd.hpp"
#include "commands/set_viewport_cmd.hpp"
#include "commands/unbind_framebuffer_cmd.hpp"
#include "commands/use_shader_cmd.hpp"
#include "commands/bind_texture_cmd.hpp"

namespace gfx {
namespace {
template <typename A, typename Variant> struct is_in_variant;

template <typename T, typename... Ts>
struct is_in_variant<T, std::variant<Ts...>>
    : public std::disjunction<std::is_same<T, Ts>...> {};
} // namespace

class render_pipeline {
public:
  using command_variant =
      std::variant<bind_framebuffer_cmd, clear_cmd, set_viewport_cmd,
                   unbind_framebuffer_cmd, use_shader_cmd, bind_texture_cmd,
                   set_uniform_cmd<float>, set_uniform_cmd<int>,
                   set_uniform_cmd<math::vector3<float>>,
                   set_uniform_cmd<math::matrix4x4<float>>, draw_mesh_cmd>;

  bool configure(const feature_set &features) noexcept {
    // TODO:  configure features
    return true;
  }

  bool begin_frame() noexcept {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    _commands.clear();
    return true;
  }

  template <typename Command>
    requires is_in_variant<Command, command_variant>::value
  bool submit(Command &&cmd) {
    _commands.emplace_back(std::move(cmd));
    return true;
  }

  bool end_frame() noexcept {
    for (auto &command : _commands) {
      std::visit([](auto &&cmd) { cmd.execute(); }, command);
    }
    return true;
  }

  bool present() noexcept { return true; }

private:
  std::vector<command_variant> _commands;
};
} // namespace gfx