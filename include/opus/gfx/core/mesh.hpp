#pragma once

#include <memory>

#include "vbuffer.hpp"

#include "../../asset/obj_model.hpp"

namespace gfx {
class mesh {
public:
  vbuffer buffer;
  std::size_t vertex_count;

  mesh(vbuffer &&buffer, std::size_t vertex_count)
      : buffer{std::move(buffer)}, vertex_count{vertex_count} {}
};
} // namespace gfx