#pragma once

#include <GL/glew.h>

#include <memory>
#include <vector>

#include "../core/render_pipeline.hpp"

#include "commands/set_uniform_cmd.hpp"
#include "commands/draw_mesh_cmd.hpp"

namespace gfx {
namespace {
template <typename A, typename Variant> struct is_in_variant;

template <typename A, typename... Ts>
struct is_in_variant<A, std::variant<Ts...>>
    : std::disjunction<std::is_same<A, Ts>...> {};
} // namespace

class opengl_pipeline : public render_pipeline {
public:
  bool configure(const feature_set &features) noexcept override {
    // TODO: configure features
    return true;
  }

  bool begin_frame() noexcept override {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    _commands.clear();
    return true;
  }

  bool end_frame() noexcept override {
    for (const auto &command : _commands) {
      command->execute();
    }

    return true;
  }

  bool submit(std::unique_ptr<render_command> command) noexcept override {
    _commands.emplace_back(std::move(command));
    return true;
  }

  bool present() noexcept override { return true; }

  ~opengl_pipeline() override = default;

private:
  std::vector<std::unique_ptr<render_command>> _commands;
};

template <> class render_pipeline2<render_backend::opengl> {
public:
  template <typename Uniform>
  using set_uniform_cmd = gfx::set_uniform_cmd<render_backend::opengl, Uniform>;

  using draw_mesh_cmd = gfx::draw_mesh_cmd<render_backend::opengl>;

  using command_variant =
      std::variant<set_uniform_cmd<float>, set_uniform_cmd<int>,
                   set_uniform_cmd<math::vector3<float>>,
                   set_uniform_cmd<math::matrix4x4<float>>,
                   draw_mesh_cmd>;

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
    for (auto & command : _commands) {
      std::visit([](auto &&cmd) { cmd.execute(); }, command);
    }
    return true; 
  }

  bool present() noexcept { return true; }

private:
  std::vector<command_variant> _commands;
};
} // namespace gfx