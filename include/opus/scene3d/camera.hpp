#pragma once

#include "transform.hpp"

#include "../math/matrix4x4.hpp"
#include "../math/vector3.hpp"

namespace scene3d {

template <typename Numeric = float> class camera {
public:
  using numeric_type = Numeric;
  using vec_type = math::vector3<numeric_type>;
  using mat_type = math::matrix4x4<numeric_type>;

  enum class projection_mode { perspective, orthographic };

  camera()
      : _fov(60.0f), _aspect_ratio(16.0f / 9.0f), _near_plane(0.1f),
        _far_plane(1000.0f), _proj_mode(projection_mode::perspective),
        _ortho_left(-1.0f), _ortho_right(1.0f), _ortho_top(1.0f),
        _ortho_bottom(-1.0f) {
    update_projection_matrix();
  }

  numeric_type get_fov() const { return _fov; }

  void set_fov(numeric_type new_fov) {
    _fov = new_fov;
    update_projection_matrix();
  }

  numeric_type get_aspect_ratio() const { return _aspect_ratio; }

  void set_aspect_ratio(numeric_type new_aspect_ratio) {
    _aspect_ratio = new_aspect_ratio;
    update_projection_matrix();
  }

  numeric_type get_near_plane() const { return _near_plane; }

  void set_near_plane(numeric_type new_near_plane) {
    _near_plane = new_near_plane;
    update_projection_matrix();
  }

  numeric_type get_far_plane() const { return _far_plane; }

  void set_far_plane(numeric_type new_far_plane) {
    _far_plane = new_far_plane;
    update_projection_matrix();
  }

  projection_mode get_projection_mode() const { return _proj_mode; }

  void set_projection_mode(projection_mode mode) {
    _proj_mode = mode;
    update_projection_matrix();
  }

  const mat_type &get_projection_matrix() const { return _projection_matrix; }

  void update_projection_matrix() {
    if (_proj_mode == projection_mode::perspective) {
      _projection_matrix =
          mat_type::perspective(_fov, _aspect_ratio, _near_plane, _far_plane);
    } else {
      _projection_matrix =
          mat_type::orthographic(_ortho_left, _ortho_right, _ortho_bottom,
                                 _ortho_top, _near_plane, _far_plane);
    }
  }

  void set_orthographic_bounds(numeric_type left, numeric_type right,
                               numeric_type bottom, numeric_type top) {
    _ortho_left = left;
    _ortho_right = right;
    _ortho_bottom = bottom;
    _ortho_top = top;
    if (_proj_mode == projection_mode::orthographic) {
      update_projection_matrix();
    }
  }

  bool operator==(const camera &other) const {
    return _fov == other._fov && _aspect_ratio == other._aspect_ratio &&
           _near_plane == other._near_plane && _far_plane == other._far_plane &&
           _proj_mode == other._proj_mode && _ortho_left == other._ortho_left &&
           _ortho_right == other._ortho_right &&
           _ortho_top == other._ortho_top &&
           _ortho_bottom == other._ortho_bottom &&
           _projection_matrix == other._projection_matrix;
  }

  bool operator!=(const camera &other) const { return !(*this == other); }

  math::matrix4x4<float>
  get_view_matrix(const scene3d::transform<float> &xform) const {
    // Calculate view matrix as inverse of the camera's world transform
    // First get the forward, right, and up vectors from the rotation quaternion
    auto forward =
        xform.rotation.rotate(math::vector3<float>(0.0f, 0.0f, 1.0f));
    auto right = xform.rotation.rotate(math::vector3<float>(1.0f, 0.0f, 0.0f));
    auto up = xform.rotation.rotate(math::vector3<float>(0.0f, 1.0f, 0.0f));

    // Build view matrix
    math::matrix4x4<float> view;
    // Set rotation part of view matrix (transposed basis vectors)
    view(0, 0)= right.x;
    view(0, 1) = up.x;
    view(0, 2) = forward.x;
    view(1, 0) = right.y;
    view(1, 1) = up.y;
    view(1, 2) = forward.y;
    view(2, 0) = right.z;
    view(2, 1) = up.z;
    view(2, 2) = forward.z;

    // Set translation part (negative dot product with position)
    view(3, 0) = -right.dot(xform.position);
    view(3, 1) = -up.dot(xform.position);
    view(3, 2) = -forward.dot(xform.position);
    view(3, 3) = 1.0f;

    return view;
  }

private:
  numeric_type _fov;
  numeric_type _aspect_ratio;
  numeric_type _near_plane;
  numeric_type _far_plane;
  projection_mode _proj_mode;

  // Orthographic parameters
  numeric_type _ortho_left;
  numeric_type _ortho_right;
  numeric_type _ortho_top;
  numeric_type _ortho_bottom;

  mat_type _projection_matrix;
};
} // namespace scene3d
