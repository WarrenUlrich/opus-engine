#pragma once

#include <GL/glew.h>
#include <iostream>
#include <optional>
#include <vector>

#include "texture2d.hpp"

namespace gfx {

class frame_buffer {
public:
  // Constructor
  frame_buffer() {
    glGenFramebuffers(1, &_fbo);
    if (_fbo == 0) {
      std::cerr << "Failed to generate framebuffer." << std::endl;
    }
  }

  // Destructor
  ~frame_buffer() {
    // Ensure that an OpenGL context is current before deleting OpenGL resources
    if (_fbo != 0) {
      if (_rbo != 0) {
        glDeleteRenderbuffers(1, &_rbo);
      }
      glDeleteFramebuffers(1, &_fbo);
    }
  }

  // Deleted copy constructor and assignment operator
  frame_buffer(const frame_buffer &) = delete;
  frame_buffer &operator=(const frame_buffer &) = delete;

  // Move constructor
  frame_buffer(frame_buffer &&other) noexcept
      : _fbo(other._fbo), _rbo(other._rbo),
        _attachments(std::move(other._attachments)) {
    other._fbo = 0;
    other._rbo = 0;
  }

  // Move assignment operator
  frame_buffer &operator=(frame_buffer &&other) noexcept {
    if (this != &other) {
        if (_rbo != 0) {
          glDeleteRenderbuffers(1, &_rbo);
        }
        if (_fbo != 0) {
          glDeleteFramebuffers(1, &_fbo);
        }

      _fbo = other._fbo;
      _rbo = other._rbo;
      _attachments = std::move(other._attachments);
      other._fbo = 0;
      other._rbo = 0;
    }
    return *this;
  }

  // Bind the framebuffer
  void bind() const {
    if (_fbo != 0) {
      glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
    } else {
      std::cerr << "Attempted to bind an uninitialized framebuffer." << std::endl;
    }
  }

  // Unbind the framebuffer
  void unbind() const { glBindFramebuffer(GL_FRAMEBUFFER, 0); }

  // Attach a texture to the framebuffer
  bool attach_texture(GLenum attachment, const texture2d &texture) {
    if (_fbo == 0) {
      std::cerr << "Framebuffer is not initialized." << std::endl;
      return false;
    }

    bind();
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D,
                           texture.id(), 0);
    unbind();

    // Add to attachments if it's a color attachment
    if (attachment >= GL_COLOR_ATTACHMENT0 &&
        attachment <= GL_COLOR_ATTACHMENT15) {
      _attachments.push_back(attachment);
    }

    return true;
  }

  // Attach a renderbuffer (for depth/stencil)
  bool attach_renderbuffer(GLenum attachment, GLenum format, int width,
                           int height) {
    if (_fbo == 0) {
      std::cerr << "Framebuffer is not initialized." << std::endl;
      return false;
    }

    if (_rbo == 0) {
      glGenRenderbuffers(1, &_rbo);
      if (_rbo == 0) {
        std::cerr << "Failed to generate renderbuffer." << std::endl;
        return false;
      }
    }

    glBindRenderbuffer(GL_RENDERBUFFER, _rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, format, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    bind();
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER,
                              _rbo);
    unbind();

    return true;
  }

  // Set the list of draw buffers
  void set_draw_buffers() {
    if (_fbo == 0) {
      std::cerr << "Framebuffer is not initialized." << std::endl;
      return;
    }

    bind();
    if (_attachments.empty()) {
      // No color attachments
      glDrawBuffer(GL_NONE);
      glReadBuffer(GL_NONE);
    } else {
      glDrawBuffers(static_cast<GLsizei>(_attachments.size()),
                    _attachments.data());
    }
    unbind();
  }

  // Check if the framebuffer is complete
  bool is_complete() const {
    if (_fbo == 0) {
      std::cerr << "Framebuffer is not initialized." << std::endl;
      return false;
    }

    bind();
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    unbind();
    if (status != GL_FRAMEBUFFER_COMPLETE) {
      std::cerr << "Framebuffer is not complete: " << status << std::endl;
      return false;
    }
    return true;
  }

  // Get the framebuffer ID
  GLuint id() const { return _fbo; }

private:
  GLuint _fbo = 0;
  GLuint _rbo = 0; // Renderbuffer ID
  std::vector<GLenum> _attachments;
};

} // namespace gfx
