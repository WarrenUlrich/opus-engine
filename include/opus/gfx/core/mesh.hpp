#pragma once

#include <memory>

#include "vbuffer.hpp"
#include "render_backend.hpp"

#include "../../asset/obj_model.hpp"

namespace gfx {
template <render_backend Backend>
class mesh {
public:
  using vbuffer_type = vbuffer<Backend>;

  vbuffer_type buffer;
  std::size_t vertex_count;

  mesh(vbuffer<Backend> &&buffer, std::size_t vertex_count)
      : buffer{std::move(buffer)}, vertex_count{vertex_count} {}

  // mesh(const asset::obj_model &model
};
} // namespace gfx