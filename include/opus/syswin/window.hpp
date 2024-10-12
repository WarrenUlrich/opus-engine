#pragma once

#include <X11/Xlib.h>

#include <optional>
#include <string_view>

#include "../gfx/core/render_backend.hpp"
#include "../gfx/core/renderer_context.hpp"

namespace syswin {
template<gfx::render_backend Backend>
class window {
public:
  using context_type = gfx::renderer_context<Backend>;

  static std::optional<window> create(std::string_view title, int width,
                                      int height) {
    const auto display = XOpenDisplay(nullptr);
    if (display == nullptr)
      return std::nullopt;

    const auto screen = DefaultScreen(display);
    const auto root = RootWindow(display, screen);

    const auto win = XCreateSimpleWindow(display, root, 0, 0, width, height, 1,
                                         BlackPixel(display, screen),
                                         WhitePixel(display, screen));

    if (XSelectInput(display, win, ExposureMask | KeyPressMask) == False)
      return std::nullopt;

    if (XMapWindow(display, win) == False)
      return std::nullopt;

    if (XStoreName(display, win, title.data()) == False)
      return std::nullopt;

    
    return window(display, win);
  }

private:
  window(Display *display, Window window)
      : _display(display), _window(window) {}

  Display *_display;
  Window _window;
};
} // namespace syswin