#pragma once

#include "../math/matrix4x4.hpp"
#include "../math/vector3.hpp"

namespace scene3d {

template <typename Numeric = float> class camera {
public:
  using numeric_type = Numeric;
  using vec_type = math::vector3<numeric_type>;
  using mat_type = math::matrix4x4<numeric_type>;

  enum class projection_mode { perspective, orthographic };

private:
  numeric_type fov;
  numeric_type aspect_ratio;
  numeric_type near_plane;
  numeric_type far_plane;
  projection_mode proj_mode;

  // Orthographic parameters
  numeric_type ortho_left;
  numeric_type ortho_right;
  numeric_type ortho_top;
  numeric_type ortho_bottom;

  mat_type projection_matrix;

public:
  camera()
      : fov(60.0f), aspect_ratio(16.0f / 9.0f), near_plane(0.1f),
        far_plane(1000.0f), proj_mode(projection_mode::perspective),
        ortho_left(-1.0f), ortho_right(1.0f), ortho_top(1.0f),
        ortho_bottom(-1.0f) {
    update_projection_matrix();
  }

  numeric_type get_fov() const { return fov; }

  void set_fov(numeric_type new_fov) {
    fov = new_fov;
    update_projection_matrix();
  }

  numeric_type get_aspect_ratio() const { return aspect_ratio; }

  void set_aspect_ratio(numeric_type new_aspect_ratio) {
    aspect_ratio = new_aspect_ratio;
    update_projection_matrix();
  }

  numeric_type get_near_plane() const { return near_plane; }

  void set_near_plane(numeric_type new_near_plane) {
    near_plane = new_near_plane;
    update_projection_matrix();
  }

  numeric_type get_far_plane() const { return far_plane; }

  void set_far_plane(numeric_type new_far_plane) {
    far_plane = new_far_plane;
    update_projection_matrix();
  }

  projection_mode get_projection_mode() const { return proj_mode; }

  void set_projection_mode(projection_mode mode) {
    proj_mode = mode;
    update_projection_matrix();
  }

  const mat_type &get_projection_matrix() const { return projection_matrix; }

  void update_projection_matrix() {
    if (proj_mode == projection_mode::perspective) {
      projection_matrix =
          mat_type::perspective(fov, aspect_ratio, near_plane, far_plane);
    } else {
      projection_matrix =
          mat_type::orthographic(ortho_left, ortho_right, ortho_bottom,
                                 ortho_top, near_plane, far_plane);
    }
  }

  void set_orthographic_bounds(numeric_type left, numeric_type right,
                               numeric_type bottom, numeric_type top) {
    ortho_left = left;
    ortho_right = right;
    ortho_bottom = bottom;
    ortho_top = top;
    if (proj_mode == projection_mode::orthographic) {
      update_projection_matrix();
    }
  }
};
} // namespace scene
