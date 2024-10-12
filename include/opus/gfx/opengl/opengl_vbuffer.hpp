#pragma once

#include <GL/glew.h>
#include <cstddef>
#include <vector>

#include "../core/vbuffer.hpp"
#include "../core/vertex_layout.hpp"

namespace gfx {
template <> class vbuffer<render_backend::opengl> {
public:
  // Constructor
  // - data: Pointer to the vertex data.
  // - size: Size of the vertex data in bytes.
  // - layout: The vertex layout specifying the attributes.
  vbuffer(const void *data, std::size_t size, const vertex_layout &layout)
      : _layout(layout) {
    // Generate and bind VAO
    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);

    // Generate and bind VBO
    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);

    // Set up vertex attribute pointers based on the layout
    const auto &attributes = _layout.get_attributes();
    std::size_t stride = _layout.get_stride();

    for (std::size_t i = 0; i < attributes.size(); ++i) {
      const auto &attr = attributes[i];
      glEnableVertexAttribArray(static_cast<GLuint>(i));
      glVertexAttribPointer(static_cast<GLuint>(i),   // Attribute index
                            attr.num_components,      // Number of components
                            _convert_type(attr.type), // Data type
                            attr.normalized ? GL_TRUE : GL_FALSE, // Normalized?
                            static_cast<GLsizei>(stride),         // Stride
                            reinterpret_cast<void *>(
                                static_cast<uintptr_t>(attr.offset)) // Offset
      );
    }

    // Unbind the VAO and VBO
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  vbuffer(const vbuffer &other) = delete;

  vbuffer(vbuffer &&other) : _vao(other._vao), _vbo(other._vbo) {
    other._vao = 0;
    other._vbo = 0;
  }

  // Destructor
  ~vbuffer() {
    if (_vbo) {
      glDeleteBuffers(1, &_vbo);
    }
    if (_vao) {
      glDeleteVertexArrays(1, &_vao);
    }
  }

  // Bind the VAO
  void bind() const { glBindVertexArray(_vao); }

  // Unbind the VAO
  void unbind() const { glBindVertexArray(0); }

private:
  GLuint _vao = 0;
  GLuint _vbo = 0;
  vertex_layout _layout;

  // Helper function to convert vertex_attribute_type to OpenGL enum
  static GLenum _convert_type(vertex_attribute_type type) {
    switch (type) {
    case vertex_attribute_type::float32:
      return GL_FLOAT;
    case vertex_attribute_type::int32:
      return GL_INT;
    case vertex_attribute_type::uint8:
      return GL_UNSIGNED_BYTE;
    default:
      return GL_FLOAT; // Default to GL_FLOAT
    }
  }
};
} // namespace gfx
