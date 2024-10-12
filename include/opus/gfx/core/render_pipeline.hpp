#pragma once

#include <memory>

#include "render_command.hpp"
#include "feature_set.hpp"

#include "render_backend.hpp"

namespace gfx {
class render_pipeline {
public:
  virtual bool configure(const feature_set &features) noexcept = 0;

  virtual bool begin_frame() noexcept = 0;

  virtual bool end_frame() noexcept = 0;

  virtual bool submit(std::unique_ptr<render_command> command) noexcept = 0;

  virtual bool present() noexcept = 0;

  virtual ~render_pipeline() noexcept = default;
};

template<render_backend Backend>
class render_pipeline2;

} // namespace gfx