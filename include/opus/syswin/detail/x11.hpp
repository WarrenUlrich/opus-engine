#pragma once

#include <X11/Xlib.h>

namespace detail::x11 {
using display = Display;
using window = Window;

using event = XEvent;
using key_event = XKeyEvent;
using motion_event = XMotionEvent;
using button_event = XButtonEvent;

} // namespace detail::x11