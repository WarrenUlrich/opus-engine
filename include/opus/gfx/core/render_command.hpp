#pragma once

#include <string_view>

namespace gfx {
  class render_command {
  public:
    virtual constexpr std::string_view name() const noexcept = 0;
    
    virtual bool execute() const noexcept = 0;
  };
}