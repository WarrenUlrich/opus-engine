#pragma once

#include <GL/glew.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "../../math/matrix4x4.hpp"
#include "../../math/vector2.hpp"
#include "../../math/vector3.hpp"

#include "texture2d.hpp"

namespace gfx {
class shader_program {
public:
  shader_program(const std::string &vertexSource,
                 const std::string &fragmentSource) {
    if (!glewInitAndCheck()) {
      throw std::runtime_error("Failed to initialize GLEW.");
    }

    // Compile shaders
    GLuint vertexShader = _compile_shader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = _compile_shader(GL_FRAGMENT_SHADER, fragmentSource);

    if (vertexShader == 0 || fragmentShader == 0) {
      throw std::runtime_error("Shader compilation failed.");
    }

    // Create and link the shader program
    _program = glCreateProgram();
    if (!_program) {
      throw std::runtime_error("Failed to create OpenGL program.");
    }

    glAttachShader(_program, vertexShader);
    glAttachShader(_program, fragmentShader);
    glLinkProgram(_program);

    // Check for linking errors
    if (!_check_link_status()) {
      glDeleteShader(vertexShader);
      glDeleteShader(fragmentShader);
      throw std::runtime_error("Shader program linking failed.");
    }

    // Clean up shaders as they are no longer needed
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
  }

  ~shader_program() {
    if (_program) {
      glDeleteProgram(_program);
    }
  }

  void use() const noexcept {
    if (_program) {
      glUseProgram(_program);
    }
  }

  bool set_uniform(const std::string_view &name, int value) noexcept {
    GLint location = _get_uniform_location(name);
    if (location != -1) {
      glUniform1i(location, value);
      return true;
    }
    return false;
  }

  bool set_uniform(const std::string_view &name, float value) noexcept {
    GLint location = _get_uniform_location(name);
    if (location != -1) {
      glUniform1f(location, value);
      return true;
    }
    return false;
  }

  bool set_uniform(const std::string_view &name, bool value) noexcept {
    GLint location = _get_uniform_location(name);
    if (location != -1) {
      glUniform1i(location, value ? 1 : 0); // Use 1 for true, 0 for false
      return true;
    }
    return false;
  }

  bool set_uniform(const std::string_view &name,
                   const math::vector2<float> &value) noexcept {
    GLint location = _get_uniform_location(name);
    if (location != -1) {
      glUniform2f(location, value.x, value.y); // Use glUniform2f for vec2
      return true;
    }
    return false;
  }

  bool set_uniform(const std::string_view &name,
                   const math::vector3<float> &value) noexcept {
    GLint location = _get_uniform_location(name);
    if (location != -1) {
      glUniform3f(location, value.x, value.y,
                  value.z); // Use glUniform3f for vec3
      return true;
    }
    return false;
  }

  bool set_uniform(const std::string_view &name,
                   const math::matrix4x4<float> &value) noexcept {
    GLint location = _get_uniform_location(name);
    if (location != -1) {
      glUniformMatrix4fv(location, 1, GL_TRUE, &value.m[0][0]);
      return true;
    }
    return false;
  }

  bool set_uniform(const std::string_view &name, const gfx::texture2d &texture,
                   int textureUnit) noexcept {
    // Bind the texture to the specified texture unit
    texture.bind(GL_TEXTURE0 + textureUnit);

    // Set the uniform to the corresponding texture unit index
    GLint location = _get_uniform_location(name);
    if (location != -1) {
      glUniform1i(location, textureUnit);
      return true;
    }
    return false;
  }

  GLuint get_program_id() const noexcept { return _program; }

private:
  GLuint _program = 0;
  mutable std::unordered_map<std::string_view, GLint> _uniform_cache;

  bool glewInitAndCheck() const {
    static bool initialized = false;
    if (!initialized) {
      GLenum err = glewInit();
      if (err != GLEW_OK) {
        std::cerr << "GLEW Initialization Error: " << glewGetErrorString(err)
                  << std::endl;
        return false;
      }
      initialized = true;
    }
    return true;
  }

  GLuint _compile_shader(GLenum type, const std::string &source) {
    GLuint shader = glCreateShader(type);
    if (!shader) {
      std::cerr
          << "ERROR::SHADER::CREATION_FAILED: Could not create shader of type "
          << type << std::endl;
      return 0;
    }

    const char *source_cstr = source.c_str();
    glShaderSource(shader, 1, &source_cstr, nullptr);
    glCompileShader(shader);

    // Check shader compilation status
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      char infoLog[1024];
      glGetShaderInfoLog(shader, sizeof(infoLog), nullptr, infoLog);
      std::cerr << "ERROR::SHADER::COMPILATION_FAILED for "
                << (type == GL_VERTEX_SHADER ? "Vertex Shader"
                                             : "Fragment Shader")
                << "\n"
                << infoLog << "\nShader Source:\n"
                << source << std::endl;
      glDeleteShader(shader);
      return 0;
    }

    return shader;
  }

  bool _check_link_status() {
    GLint success;
    glGetProgramiv(_program, GL_LINK_STATUS, &success);
    if (!success) {
      char infoLog[1024];
      glGetProgramInfoLog(_program, sizeof(infoLog), nullptr, infoLog);
      std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"
                << infoLog << std::endl;
      glDeleteProgram(_program);
      _program = 0;
      return false;
    }
    return true;
  }

  GLint _get_uniform_location(const std::string_view &name) const {
    if (_uniform_cache.find(name) != _uniform_cache.end()) {
      return _uniform_cache[name];
    }

    GLint location = glGetUniformLocation(_program, name.data());
    if (location == -1) {
      std::cerr << "Warning: Uniform '" << name
                << "' does not exist in shader program." << std::endl;
    } else {
      _uniform_cache[name] = location;
    }

    return location;
  }
};
} // namespace gfx
