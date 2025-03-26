#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <format>
#include <limits>
#include <numbers>

#include "quaternion.hpp"
#include "vector3.hpp"

namespace math {

/**
 * @class matrix4x4
 * @brief A 4x4 matrix class primarily used for 3D transformations such as
 * translation, scaling, rotation, and projections.
 *
 * The `matrix4x4` class represents a 4x4 matrix, which is essential in computer
 * graphics for transforming 3D objects (e.g., translating, rotating, scaling,
 * and projecting them). This class supports common matrix operations like
 * multiplication and transposition, as well as generating specialized
 * transformation matrices.
 *
 * A 4x4 matrix is often used in homogeneous coordinates to enable translations
 * along with other linear transformations. The class offers both basic and
 * advanced transformations, like generating rotation matrices around specific
 * axes or creating projection matrices for 3D rendering. It also provides
 * utility functions to convert between matrices and quaternions.
 *
 * @tparam Numeric The underlying numeric type (e.g., float or double). Supports
 * both floating-point and integral types, although floating-point types are
 * more suitable for 3D graphics.
 */
template <typename Numeric = float>
  requires std::is_floating_point_v<Numeric> || std::is_integral_v<Numeric>
class matrix4x4 {
public:
  using numeric_type = Numeric;
  using mat_type = matrix4x4<numeric_type>;
  using vec_type = vector3<numeric_type>;

  constexpr static bool is_double_precision =
      std::is_same_v<numeric_type, double>;

  // Static matrices
  consteval static mat_type zero() noexcept { return mat_type{}; }

  /**
   * @brief Creates an identity matrix.
   *
   * The identity matrix has 1s along the diagonal and 0s elsewhere. It
   * represents a "no transformation" matrix and serves as the neutral element
   * for matrix multiplication.
   *
   * @return A 4x4 identity matrix.
   */
  consteval static mat_type identity() noexcept {
    mat_type result{};
    for (int i = 0; i < 4; ++i) {
      result(i, i) = static_cast<numeric_type>(1);
    }
    return result;
  }

  numeric_type m[16];

  // Constructors
  constexpr matrix4x4() noexcept {
    for (int i = 0; i < 16; ++i)
      m[i] = static_cast<numeric_type>(0);
  }

  constexpr matrix4x4(const mat_type &other) noexcept {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        (*this)(i, j) = other(i, j);
  }

  constexpr mat_type &operator=(const mat_type &other) noexcept {
    if (this != &other) {
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          (*this)(i, j) = other(i, j);
    }
    return *this;
  }

  constexpr numeric_type &operator()(std::size_t row,
                                     std::size_t col) noexcept {
    return m[col * 4 + row];
  }

  constexpr const numeric_type &operator()(std::size_t row,
                                           std::size_t col) const noexcept {
    return m[col * 4 + row];
  }

  // Comparison operators
  constexpr bool operator==(const mat_type &other) const noexcept {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        if ((*this)(i, j) != other(i, j))
          return false;
    return true;
  }

  constexpr bool operator!=(const mat_type &other) const noexcept {
    return !(*this == other);
  }

  // Arithmetic operators
  constexpr mat_type operator+(const mat_type &other) const noexcept {
    mat_type result;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        result(i, j) = (*this)(i, j) + other(i, j);
    return result;
  }

  constexpr mat_type operator-(const mat_type &other) const noexcept {
    mat_type result;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        result.m[i][j] = m[i][j] - other.m[i][j];
    return result;
  }

  constexpr mat_type operator*(const mat_type &other) const noexcept {
    mat_type result{};
    for (int row = 0; row < 4; ++row)
      for (int col = 0; col < 4; ++col)
        for (int k = 0; k < 4; ++k)
          result(row, col) += (*this)(row, k) * other(k, col);
    return result;
  }

  constexpr mat_type operator*(numeric_type scalar) const noexcept {
    mat_type result;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        result.m[i][j] = m[i][j] * scalar;
    return result;
  }

  constexpr mat_type &operator+=(const mat_type &other) noexcept {
    *this = *this + other;
    return *this;
  }

  constexpr mat_type &operator-=(const mat_type &other) noexcept {
    *this = *this - other;
    return *this;
  }

  constexpr mat_type &operator*=(const mat_type &other) noexcept {
    *this = *this * other;
    return *this;
  }

  constexpr mat_type &operator*=(numeric_type scalar) noexcept {
    *this = *this * scalar;
    return *this;
  }

  // Vector transformation
  constexpr vec_type operator*(const vec_type &vec) const noexcept {
    numeric_type x = (*this)(0, 0) * vec.x + (*this)(0, 1) * vec.y +
                     (*this)(0, 2) * vec.z + (*this)(0, 3);
    numeric_type y = (*this)(1, 0) * vec.x + (*this)(1, 1) * vec.y +
                     (*this)(1, 2) * vec.z + (*this)(1, 3);
    numeric_type z = (*this)(2, 0) * vec.x + (*this)(2, 1) * vec.y +
                     (*this)(2, 2) * vec.z + (*this)(2, 3);
    numeric_type w = (*this)(3, 0) * vec.x + (*this)(3, 1) * vec.y +
                     (*this)(3, 2) * vec.z + (*this)(3, 3);

    if (w != static_cast<numeric_type>(0) &&
        w != static_cast<numeric_type>(1)) {
      x /= w;
      y /= w;
      z /= w;
    }

    return vec_type{x, y, z};
  }

  // Transpose
  constexpr mat_type transpose() const noexcept {
    mat_type result;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        result(j, i) = (*this)(i, j);
    return result;
  }

  // Determinant (Not fully implemented due to complexity)
  numeric_type determinant() const noexcept {
    // Using cofactor expansion along the first row
    return (*this)(0, 0) * minor(0, 0) - (*this)(0, 1) * minor(0, 1) +
           (*this)(0, 2) * minor(0, 2) - (*this)(0, 3) * minor(0, 3);
  }

  numeric_type minor(std::size_t row, std::size_t col) const noexcept {
    // Create arrays to store the indices of the 3x3 submatrix
    std::size_t rows[3], cols[3];

    // Fill the arrays with indices, skipping the specified row and column
    for (std::size_t i = 0, r = 0; i < 4; ++i) {
      if (i != row) {
        rows[r++] = i;
      }
    }

    for (std::size_t j = 0, c = 0; j < 4; ++j) {
      if (j != col) {
        cols[c++] = j;
      }
    }

    // Get the elements of the 3x3 submatrix
    numeric_type a = (*this)(rows[0], cols[0]);
    numeric_type b = (*this)(rows[0], cols[1]);
    numeric_type c = (*this)(rows[0], cols[2]);
    numeric_type d = (*this)(rows[1], cols[0]);
    numeric_type e = (*this)(rows[1], cols[1]);
    numeric_type f = (*this)(rows[1], cols[2]);
    numeric_type g = (*this)(rows[2], cols[0]);
    numeric_type h = (*this)(rows[2], cols[1]);
    numeric_type i = (*this)(rows[2], cols[2]);

    // Calculate the determinant of the 3x3 submatrix
    // Det = a(ei - fh) - b(di - fg) + c(dh - eg)
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  }

  /**
   * @brief Calculates the inverse of the matrix.
   *
   * The inverse of a matrix A is another matrix A⁻¹ such that A * A⁻¹ = I,
   * where I is the identity matrix. This is calculated using the adjugate
   * method (transpose of cofactor matrix divided by determinant).
   *
   * @return The inverse of the matrix, or identity matrix if not invertible.
   */
  mat_type inverse() const noexcept {
    numeric_type det = determinant();

    // Cannot invert a matrix with zero determinant
    if (std::abs(det) < std::numeric_limits<numeric_type>::epsilon()) {
      return mat_type::identity();
    }

    mat_type result;

    // Calculate the cofactor matrix (using minors with appropriate sign)
    for (std::size_t i = 0; i < 4; ++i) {
      for (std::size_t j = 0; j < 4; ++j) {
        // Cofactor sign: (-1)^(i+j)
        numeric_type sign = ((i + j) % 2 == 0) ? 1 : -1;
        // Note: transpose by swapping i and j in result
        result(j, i) = sign * minor(i, j) / det;
      }
    }

    return result;
  }

  /**
   * @brief Creates a translation matrix.
   *
   * A translation matrix is used to move (translate) objects in 3D space.
   * It shifts the position of an object along the X, Y, and Z axes based on
   * the provided offset. This matrix plays a key role in transformations
   * such as moving objects, characters, or cameras in a 3D scene.
   *
   * This function constructs a 4x4 translation matrix based on the given
   * offset vector. When this matrix is multiplied by a point or object’s
   * position, the object is moved (translated) by the specified amount along
   * each axis.
   *
   * @param offset A 3D vector (`vec_type`) representing how much to translate
   *               along each axis:
   *               - `offset.x`: Amount to translate along the X-axis.
   *               - `offset.y`: Amount to translate along the Y-axis.
   *               - `offset.z`: Amount to translate along the Z-axis.
   *
   * @return A 4x4 translation matrix, where:
   *         - The main diagonal is filled with 1s (identity matrix).
   *         - The last column contains the X, Y, and Z offsets, controlling
   *           how much to move along each axis.
   */
  constexpr static mat_type translation(const vec_type &offset) noexcept {
    mat_type result = identity();
    result(0, 3) = offset.x;
    result(1, 3) = offset.y;
    result(2, 3) = offset.z;
    return result;
  }

  constexpr static mat_type scaling(const vec_type &factors) noexcept {
    mat_type result = identity();
    result(0, 0) = factors.x;
    result(1, 1) = factors.y;
    result(2, 2) = factors.z;
    return result;
  }

  constexpr static mat_type rotation_x(numeric_type angle) noexcept {
    mat_type result = identity();
    numeric_type c = std::cos(angle);
    numeric_type s = std::sin(angle);
    result(1, 1) = c;
    result(1, 2) = -s;
    result(2, 1) = s;
    result(2, 2) = c;
    return result;
  }

  constexpr static mat_type rotation_y(numeric_type angle) noexcept {
    mat_type result = identity();
    numeric_type c = std::cos(angle);
    numeric_type s = std::sin(angle);
    result(0, 0) = c;
    result(0, 2) = s;
    result(2, 0) = -s;
    result(2, 2) = c;
    return result;
  }

  constexpr static mat_type rotation_z(numeric_type angle) noexcept {
    mat_type result = identity();
    numeric_type c = std::cos(angle);
    numeric_type s = std::sin(angle);
    result.m[0][0] = c;
    result.m[0][1] = -s;
    result.m[1][0] = s;
    result.m[1][1] = c;
    return result;
  }

  constexpr static mat_type rotation_axis(const vec_type &axis,
                                          numeric_type angle) noexcept {
    mat_type result = identity();
    vec_type n = axis.normalized();
    numeric_type c = std::cos(angle);
    numeric_type s = std::sin(angle);
    numeric_type t = static_cast<numeric_type>(1) - c;

    result.m[0][0] = c + n.x * n.x * t;
    result.m[0][1] = n.x * n.y * t - n.z * s;
    result.m[0][2] = n.x * n.z * t + n.y * s;

    result.m[1][0] = n.y * n.x * t + n.z * s;
    result.m[1][1] = c + n.y * n.y * t;
    result.m[1][2] = n.y * n.z * t - n.x * s;

    result.m[2][0] = n.z * n.x * t - n.y * s;
    result.m[2][1] = n.z * n.y * t + n.x * s;
    result.m[2][2] = c + n.z * n.z * t;

    return result;
  }

  /**
   * @brief Creates a perspective projection matrix.
   *
   * A perspective projection matrix simulates how the human eye perceives
   * the world, where objects farther away appear smaller. This matrix is
   * essential in 3D rendering, mapping 3D points to 2D screen coordinates
   * while maintaining a sense of depth.
   *
   * The matrix ensures that objects between the near and far clipping planes
   * are visible, while everything outside these planes is clipped (not
   * rendered). It also scales the scene correctly based on the aspect ratio to
   * prevent distortion on non-square screens.
   *
   * @param fov The field of view (FOV) angle in radians, defining how wide
   *            the camera's viewing cone is. Larger values give a wide-angle
   * view, while smaller values zoom in.
   * @param aspect The aspect ratio of the display (width / height). This
   * ensures the scene does not look stretched or compressed.
   * @param near The near clipping plane. Objects closer than this distance
   *             to the camera will not be rendered.
   * @param far The far clipping plane. Objects farther than this distance
   *            from the camera will also not be rendered.
   *
   * @return A 4x4 perspective projection matrix, where:
   *         - The X and Y axes are scaled based on the FOV and aspect ratio.
   *         - The Z axis is transformed to map the depth correctly between
   *           the near and far planes.
   *         - A perspective divide is applied, making distant objects appear
   * smaller.
   */
  constexpr static mat_type perspective(numeric_type fov, numeric_type aspect,
                                        numeric_type near,
                                        numeric_type far) noexcept {
    numeric_type f = static_cast<numeric_type>(1) /
                     std::tan(fov / static_cast<numeric_type>(2));
    mat_type result = zero();

    result(0, 0) = f / aspect;
    result(1, 1) = f;
    result(2, 2) = (far + near) / (near - far);
    result(2, 3) = (2 * far * near) / (near - far);
    result(3, 2) = -1;

    return result;
  }

  constexpr static mat_type orthographic(numeric_type left, numeric_type right,
                                         numeric_type bottom, numeric_type top,
                                         numeric_type near,
                                         numeric_type far) noexcept {
    mat_type result = zero();

    result(0, 0) = static_cast<numeric_type>(2) / (right - left);
    result(1, 1) = static_cast<numeric_type>(2) / (top - bottom);
    result(2, 2) = static_cast<numeric_type>(2) / (near - far);

    result(0, 3) = (left + right) / (left - right);
    result(1, 3) = (bottom + top) / (bottom - top);
    result(2, 3) = (far + near) / (near - far);

    result(3, 3) = static_cast<numeric_type>(1);

    return result;
  }

  constexpr static mat_type
  from_quaternion(const quaternion<numeric_type> &q) noexcept {
    mat_type result = identity();
    numeric_type w = q.w;
    numeric_type x = q.x;
    numeric_type y = q.y;
    numeric_type z = q.z;

    numeric_type xx = x * x;
    numeric_type yy = y * y;
    numeric_type zz = z * z;
    numeric_type xy = x * y;
    numeric_type xz = x * z;
    numeric_type yz = y * z;
    numeric_type wx = w * x;
    numeric_type wy = w * y;
    numeric_type wz = w * z;

    result(0, 0) =
        static_cast<numeric_type>(1) - static_cast<numeric_type>(2) * (yy + zz);
    result(0, 1) = static_cast<numeric_type>(2) * (xy - wz);
    result(0, 2) = static_cast<numeric_type>(2) * (xz + wy);
    result(0, 3) = static_cast<numeric_type>(0);

    result(1, 0) = static_cast<numeric_type>(2) * (xy + wz);
    result(1, 1) =
        static_cast<numeric_type>(1) - static_cast<numeric_type>(2) * (xx + zz);
    result(1, 2) = static_cast<numeric_type>(2) * (yz - wx);
    result(1, 3) = static_cast<numeric_type>(0);

    result(2, 0) = static_cast<numeric_type>(2) * (xz - wy);
    result(2, 1) = static_cast<numeric_type>(2) * (yz + wx);
    result(2, 2) =
        static_cast<numeric_type>(1) - static_cast<numeric_type>(2) * (xx + yy);
    result(2, 3) = static_cast<numeric_type>(0);

    result(3, 0) = static_cast<numeric_type>(0);
    result(3, 1) = static_cast<numeric_type>(0);
    result(3, 2) = static_cast<numeric_type>(0);
    result(3, 3) = static_cast<numeric_type>(1);

    return result;
  }

  /**
   * @brief Converts this rotation matrix to a quaternion.
   *
   * This method extracts a quaternion from a 3D rotation matrix using a
   * numerically stable algorithm that works well even in edge cases. The matrix
   * should be a proper rotation matrix (orthogonal with determinant 1) for
   * correct results.
   *
   * @return A quaternion representing the same rotation as this matrix.
   *
   * @note If the input matrix is not a proper rotation matrix, the results may
   * be incorrect.
   */
  constexpr quaternion<numeric_type> to_quaternion() const noexcept {
    // Get the trace of the rotation matrix (sum of diagonal elements)
    numeric_type trace = (*this)(0, 0) + (*this)(1, 1) + (*this)(2, 2);

    if (trace > 0) {
      // If the trace is positive, use the standard method
      numeric_type s = static_cast<numeric_type>(0.5) / std::sqrt(trace + 1.0);
      return quaternion<numeric_type>(static_cast<numeric_type>(0.25) / s,
                                      ((*this)(2, 1) - (*this)(1, 2)) * s,
                                      ((*this)(0, 2) - (*this)(2, 0)) * s,
                                      ((*this)(1, 0) - (*this)(0, 1)) * s);
    } else {
      // If the trace is not positive, find the largest diagonal element
      // and use a specific formula based on which one it is
      if ((*this)(0, 0) > (*this)(1, 1) && (*this)(0, 0) > (*this)(2, 2)) {
        // x is the largest component
        numeric_type s =
            static_cast<numeric_type>(2.0) *
            std::sqrt(1.0 + (*this)(0, 0) - (*this)(1, 1) - (*this)(2, 2));
        return quaternion<numeric_type>(((*this)(2, 1) - (*this)(1, 2)) / s,
                                        static_cast<numeric_type>(0.25) * s,
                                        ((*this)(0, 1) + (*this)(1, 0)) / s,
                                        ((*this)(0, 2) + (*this)(2, 0)) / s);
      } else if ((*this)(1, 1) > (*this)(2, 2)) {
        // y is the largest component
        numeric_type s =
            static_cast<numeric_type>(2.0) *
            std::sqrt(1.0 + (*this)(1, 1) - (*this)(0, 0) - (*this)(2, 2));
        return quaternion<numeric_type>(((*this)(0, 2) - (*this)(2, 0)) / s,
                                        ((*this)(0, 1) + (*this)(1, 0)) / s,
                                        static_cast<numeric_type>(0.25) * s,
                                        ((*this)(1, 2) + (*this)(2, 1)) / s);
      } else {
        // z is the largest component
        numeric_type s =
            static_cast<numeric_type>(2.0) *
            std::sqrt(1.0 + (*this)(2, 2) - (*this)(0, 0) - (*this)(1, 1));
        return quaternion<numeric_type>(((*this)(1, 0) - (*this)(0, 1)) / s,
                                        ((*this)(0, 2) + (*this)(2, 0)) / s,
                                        ((*this)(1, 2) + (*this)(2, 1)) / s,
                                        static_cast<numeric_type>(0.25) * s);
      }
    }
  }

  constexpr static mat_type look_at(const vec_type &eye, const vec_type &target,
                                    const vec_type &up = vec_type(0, 1, 0)) noexcept {
    vec_type forward =
        (eye - target)
            .normalized(); // Forward vector (camera's "backwards" direction)
    vec_type right = up.cross(forward).normalized(); // Right vector
    vec_type camera_up =
        forward.cross(right); // Recomputed up vector to ensure orthogonality

    mat_type result = identity();

    // Set rotation part of the matrix (top-left 3x3 part)
    result(0, 0) = right.x;
    result(1, 0) = right.y;
    result(2, 0) = right.z;
    result(0, 1) = camera_up.x;
    result(1, 1) = camera_up.y;
    result(2, 1) = camera_up.z;
    result(0, 2) = forward.x;
    result(1, 2) = forward.y;
    result(2, 2) = forward.z;
    // Set translation part of the matrix (top-right 3x1 part)
    result(3, 0) = -right.dot(eye);
    result(3, 1) = -camera_up.dot(eye);
    result(3, 2) = -forward.dot(eye);

    return result;
  }

  /**
   * @brief Sets the values of an entire column in the matrix.
   *
   * This method allows setting all four values of a specified column at once,
   * which is particularly useful when constructing transformation matrices
   * where columns often represent specific components (e.g., basis vectors
   * or translation components).
   *
   * @param col The index of the column to set (0-3).
   * @param values A 4-element array containing the values to set.
   * @return A reference to this matrix for method chaining.
   */
  constexpr mat_type &set_column(std::size_t col,
                                 const numeric_type values[4]) noexcept {
    for (std::size_t row = 0; row < 4; ++row) {
      (*this)(row, col) = values[row];
    }
    return *this;
  }

  /**
   * @brief Sets the values of an entire column in the matrix using a vector3
   * and w component.
   *
   * This overload is convenient for setting the first three columns (which
   * often represent basis vectors in transformation matrices) or the fourth
   * column (which represents translation in homogeneous coordinates).
   *
   * @param col The index of the column to set (0-3).
   * @param vec The 3D vector containing the first three values.
   * @param w The fourth component (default is 0 for basis vectors, 1 for the
   * translation column).
   * @return A reference to this matrix for method chaining.
   */
  constexpr mat_type &set_column(std::size_t col, const vec_type &vec,
                                 numeric_type w = 0) noexcept {
    (*this)(0, col) = vec.x;
    (*this)(1, col) = vec.y;
    (*this)(2, col) = vec.z;
    (*this)(3, col) = w;
    return *this;
  }
};

} // namespace math

template <std::floating_point Float>
struct std::formatter<math::matrix4x4<Float>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const math::matrix4x4<Float> &mat,
              std::format_context &ctx) const {
    auto out = ctx.out();
    out = std::format_to(out, "[\n");
    for (int i = 0; i < 4; ++i) {
      out = std::format_to(out, "  [");
      for (int j = 0; j < 4; ++j) {
        out = std::format_to(out, "{: .4f}", mat.m[i][j]);
        if (j < 3)
          out = std::format_to(out, ", ");
      }
      out = std::format_to(out, "]");
      if (i < 3)
        out = std::format_to(out, ",\n");
    }
    out = std::format_to(out, "\n]");
    return out;
  }
};
