#pragma once

#include <GL/glew.h>
#include <iostream>
#include <optional>

namespace gfx {

class texture2d {
public:
  texture2d() = default;

  static std::optional<texture2d>
  create_empty(int width, int height, GLenum internalFormat = GL_RGBA,
               GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE,
               GLenum minFilter = GL_LINEAR, GLenum magFilter = GL_LINEAR,
               GLenum wrapS = GL_CLAMP_TO_EDGE,
               GLenum wrapT = GL_CLAMP_TO_EDGE) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    if (textureID == 0) {
      std::cerr << "Failed to generate texture." << std::endl;
      return std::nullopt;
    }

    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format,
                 type, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);

    glBindTexture(GL_TEXTURE_2D, 0);

    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
      std::cerr << "OpenGL error: " << error << std::endl;
      glDeleteTextures(1, &textureID);
      return std::nullopt;
    }

    return texture2d(textureID, width, height);
  }

  ~texture2d() {
    if (_texture_id != 0) {
      glDeleteTextures(1, &_texture_id);
    }
  }

  texture2d(const texture2d &) = delete;
  texture2d &operator=(const texture2d &) = delete;

  texture2d(texture2d &&other) noexcept
      : _texture_id(other._texture_id), _width(other._width),
        _height(other._height) {
    other._texture_id = 0;
  }

  texture2d &operator=(texture2d &&other) noexcept {
    if (this != &other) {
      if (_texture_id != 0) {
        glDeleteTextures(1, &_texture_id);
      }
      _texture_id = other._texture_id;
      _width = other._width;
      _height = other._height;
      other._texture_id = 0;
    }
    return *this;
  }

  void bind(GLenum textureUnit = GL_TEXTURE0) const {
    glActiveTexture(textureUnit);
    glBindTexture(GL_TEXTURE_2D, _texture_id);
  }

  void unbind(GLenum textureUnit = GL_TEXTURE0) const {
    glActiveTexture(textureUnit);
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  GLuint id() const { return _texture_id; }

  int width() const { return _width; }

  int height() const { return _height; }

private:
  texture2d(GLuint textureID, int width, int height)
      : _texture_id(textureID), _width(width), _height(height) {}

  GLuint _texture_id = 0;
  int _width = 0;
  int _height = 0;
};

} // namespace gfx
