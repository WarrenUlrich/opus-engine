#pragma once

#include <GL/gl.h>
#include <GL/glx.h>
#include <X11/X.h>
#include <X11/Xlib.h>

#include <array>
#include <print>

#include "../core/renderer_context.hpp"

namespace gfx {
template <> class renderer_context<render_backend::opengl> {
public:
  renderer_context(Display *display, Window window) noexcept
      : _display(display), _window(window) {}

  bool init(const feature_set &features) noexcept {
    auto attributes = std::array<GLint, 5>{GLX_RGBA, GLX_DEPTH_SIZE, 24,
                                           GLX_DOUBLEBUFFER, None};

    const auto visualInfo =
        glXChooseVisual(_display, DefaultScreen(_display), attributes.data());

    if (!visualInfo) {
      std::println("Failed to get an OpenGL visual");
      return false;
    }

    _context = glXCreateContext(_display, visualInfo, nullptr, GL_TRUE);
    if (!_context) {
      std::println("Failed to create an OpenGL context");
      return false;
    }

    if (!glXMakeCurrent(_display, _window, _context)) {
      std::println("Failed to make the OpenGL context current");
      return false;
    }

    if (features.get_feature("multisampling")) {
      glEnable(GL_MULTISAMPLE);
    }

    return true;
  }

  bool swap_buffers() noexcept {
    glXSwapBuffers(_display, _window);
    return true;
  }

  ~renderer_context() {
    if (_context) {
      glXMakeCurrent(_display, None,
                     nullptr);               // Unbind the context
      glXDestroyContext(_display, _context); // Destroy the OpenGL context
      _context = nullptr;
    }
  }

private:
  Display *_display;
  Window _window;
  GLXContext _context;
};
} // namespace gfx