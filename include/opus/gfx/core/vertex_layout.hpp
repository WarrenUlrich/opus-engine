#pragma once

#include <cstdint>
#include <vector>

namespace gfx {

enum class vertex_attribute_semantic : std::uint8_t {
  position,
  color0,
  normal,
  tex_coord0
};

enum class vertex_attribute_type : std::uint8_t { float32, int32, uint8 };

class vertex_attribute {
public:
  vertex_attribute_semantic semantic;
  vertex_attribute_type type;
  std::uint8_t num_components;
  bool normalized;
  std::uint32_t offset;
};

class vertex_layout {
public:
  vertex_layout() : _stride(0) {}

  void add_attribute(vertex_attribute &&attribute) {
    auto type_size = _calc_type_size(attribute);
    attribute.offset = _stride;

    _stride += type_size * attribute.num_components;
    _attributes.emplace_back(std::move(attribute));
  }

  // Accessor methods
  const std::vector<vertex_attribute> &get_attributes() const {
    return _attributes;
  }

  std::uint32_t get_stride() const { return _stride; }

private:
  std::vector<vertex_attribute> _attributes;
  std::uint32_t _stride;

  std::uint32_t _calc_type_size(const vertex_attribute &attribute) const {
    switch (attribute.type) {
    case vertex_attribute_type::float32:
      return 4;
    case vertex_attribute_type::int32:
      return 4;
    case vertex_attribute_type::uint8:
      return 1;
    default:
      return 0; // Or handle error appropriately
    }
  }

  friend class vertex_layout_builder;
};

class vertex_layout_builder {
public:
  vertex_layout_builder() = default;

  vertex_layout_builder &with_attribute(vertex_attribute_semantic semantic,
                                        vertex_attribute_type type,
                                        std::uint8_t num_components,
                                        bool normalized = false) {
    vertex_attribute attribute;
    attribute.semantic = semantic;
    attribute.type = type;
    attribute.num_components = num_components;
    attribute.normalized = normalized;
    attribute.offset = 0; // Will be set in add_attribute

    _layout.add_attribute(std::move(attribute));
    return *this;
  }

  vertex_layout build() { return std::move(_layout); }

private:
  vertex_layout _layout;
};

} // namespace gfx
