#pragma once
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <webgpu/webgpu_cpp.h>

#include "texture.hpp"

namespace scene3d {
class material {
public:
  friend class forward_renderer;
};
} // namespace scene3d