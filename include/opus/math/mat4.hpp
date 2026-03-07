#pragma once

#include "vec3.hpp"

#include <cmath>
#include <cstddef>

namespace math {

class mat4 {
public:
	// Column-major memory layout: m[column * 4 + row]
	float m[16]{0.0f};

	constexpr mat4() noexcept = default;

	[[nodiscard]] static constexpr mat4 identity() noexcept {
		mat4 res;
		res.m[0] = 1.0f;
		res.m[5] = 1.0f;
		res.m[10] = 1.0f;
		res.m[15] = 1.0f;
		return res;
	}

	constexpr float &operator[](std::size_t i) noexcept { return m[i]; }
	constexpr const float &operator[](std::size_t i) const noexcept { return m[i]; }

	// Matrix-Matrix Multiplication (Hidden Friend for fast compilation & RVO)
	[[nodiscard]] friend constexpr mat4 operator*(const mat4 &lhs, const mat4 &rhs) noexcept {
		mat4 res;
		for (int c = 0; c < 4; ++c) {
			for (int r = 0; r < 4; ++r) {
				res.m[c * 4 + r] =
				    lhs.m[0 * 4 + r] * rhs.m[c * 4 + 0] + lhs.m[1 * 4 + r] * rhs.m[c * 4 + 1] +
				    lhs.m[2 * 4 + r] * rhs.m[c * 4 + 2] + lhs.m[3 * 4 + r] * rhs.m[c * 4 + 3];
			}
		}
		return res;
	}

	constexpr mat4 &operator*=(const mat4 &rhs) noexcept {
		*this = *this * rhs;
		return *this;
	}

	[[nodiscard]] static constexpr mat4 translate(const vec3 &v) noexcept {
		mat4 res = identity();
		res.m[12] = v.x;
		res.m[13] = v.y;
		res.m[14] = v.z;
		return res;
	}

	[[nodiscard]] static constexpr mat4 scale(const vec3 &v) noexcept {
		mat4 res = identity();
		res.m[0] = v.x;
		res.m[5] = v.y;
		res.m[10] = v.z;
		return res;
	}

	// Uses Rodrigues' rotation formula. Axis must be normalized.
	[[nodiscard]] static inline mat4 rotate(float angle_rad, const vec3 &axis) noexcept {
		mat4 res = identity();
		const float c = std::cos(angle_rad);
		const float s = std::sin(angle_rad);
		const float omc = 1.0f - c;

		const float x = axis.x, y = axis.y, z = axis.z;

		res.m[0] = x * x * omc + c;
		res.m[1] = y * x * omc + z * s;
		res.m[2] = x * z * omc - y * s;

		res.m[4] = x * y * omc - z * s;
		res.m[5] = y * y * omc + c;
		res.m[6] = y * z * omc + x * s;

		res.m[8] = x * z * omc + y * s;
		res.m[9] = y * z * omc - x * s;
		res.m[10] = z * z * omc + c;

		return res;
	}

	[[nodiscard]] constexpr mat4 transposed() const noexcept {
		mat4 res;
		for (int c = 0; c < 4; ++c)
			for (int r = 0; r < 4; ++r)
				res.m[r * 4 + c] = m[c * 4 + r];
		return res;
	}

	// Right-handed perspective projection.
	[[nodiscard]] static inline mat4 perspective(float fov_y_rad, float aspect, float n,
	                                             float f) noexcept {
		mat4 res;
		const float tan_half_fov = std::tan(fov_y_rad * 0.5f);

		res.m[0] = 1.0f / (aspect * tan_half_fov);
		res.m[5] = 1.0f / tan_half_fov;
		res.m[10] = -(f + n) / (f - n);
		res.m[11] = -1.0f;
		res.m[14] = -(2.0f * f * n) / (f - n);

		return res;
	}

	// Right-handed orthographic projection.
	[[nodiscard]] static constexpr mat4 ortho(float left, float right, float bottom, float top,
	                                          float near_val, float far_val) noexcept {
		mat4 res;
		res.m[0] = 2.0f / (right - left);
		res.m[5] = 2.0f / (top - bottom);
		res.m[10] = -2.0f / (far_val - near_val);
		res.m[12] = -(right + left) / (right - left);
		res.m[13] = -(top + bottom) / (top - bottom);
		res.m[14] = -(far_val + near_val) / (far_val - near_val);
		res.m[15] = 1.0f;
		return res;
	}

	// Right-handed view matrix.
	[[nodiscard]] static inline mat4 look_at(const vec3 &eye, const vec3 &center,
	                                         const vec3 &up) noexcept {
		const vec3 f = (center - eye).normalized();
		const vec3 s = cross(f, up).normalized();
		const vec3 u = cross(s, f);

		mat4 res = identity();
		res.m[0] = s.x;
		res.m[4] = s.y;
		res.m[8] = s.z;
		res.m[1] = u.x;
		res.m[5] = u.y;
		res.m[9] = u.z;
		res.m[2] = -f.x;
		res.m[6] = -f.y;
		res.m[10] = -f.z;

		res.m[12] = -dot(s, eye);
		res.m[13] = -dot(u, eye);
		res.m[14] = dot(f, eye);

		return res;
	}
};

} // namespace math