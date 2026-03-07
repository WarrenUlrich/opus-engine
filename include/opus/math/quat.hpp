#pragma once

#include "mat4.hpp"
#include "vec3.hpp"

#include <cmath>

namespace math {

class quat {
public:
	float x{0.0f}, y{0.0f}, z{0.0f}, w{1.0f};

	constexpr quat() noexcept = default;
	constexpr quat(float _x, float _y, float _z, float _w) noexcept : x{_x}, y{_y}, z{_z}, w{_w} {}

	[[nodiscard]] static inline quat angle_axis(float angle_rad, const vec3 &axis) noexcept {
		const float half = angle_rad * 0.5f;
		const float s = std::sin(half);
		return {axis.x * s, axis.y * s, axis.z * s, std::cos(half)};
	}

	// Builds a rotation from a look direction (camera convention: -Z = forward).
	// Extracts quaternion from the orthonormal basis via Shepperd's method.
	[[nodiscard]] static inline quat
	look_rotation(const vec3 &forward, const vec3 &world_up = {0.0f, 1.0f, 0.0f}) noexcept {
		vec3 f = forward.normalized();
		vec3 r = cross(f, world_up).normalized();
		vec3 u = cross(r, f);

		// Rotation matrix columns: col0=right, col1=up, col2=-forward
		float m00 = r.x, m01 = u.x, m02 = -f.x;
		float m10 = r.y, m11 = u.y, m12 = -f.y;
		float m20 = r.z, m21 = u.z, m22 = -f.z;

		float trace = m00 + m11 + m22;
		if (trace > 0.0f) {
			float s = 2.0f * std::sqrt(trace + 1.0f);
			return {(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s * 0.25f};
		} else if (m00 > m11 && m00 > m22) {
			float s = 2.0f * std::sqrt(1.0f + m00 - m11 - m22);
			return {s * 0.25f, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s};
		} else if (m11 > m22) {
			float s = 2.0f * std::sqrt(1.0f + m11 - m00 - m22);
			return {(m01 + m10) / s, s * 0.25f, (m12 + m21) / s, (m02 - m20) / s};
		} else {
			float s = 2.0f * std::sqrt(1.0f + m22 - m00 - m11);
			return {(m02 + m20) / s, (m12 + m21) / s, s * 0.25f, (m10 - m01) / s};
		}
	}

	[[nodiscard]] friend constexpr quat operator*(const quat &q1, const quat &q2) noexcept {
		return {q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
		        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
		        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
		        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z};
	}

	// Rotates a vector by this quaternion (q * v * q^-1, optimized form).
	[[nodiscard]] constexpr vec3 rotate(const vec3 &v) const noexcept {
		vec3 u{x, y, z};
		return u * (2.0f * dot(u, v)) + v * (w * w - dot(u, u)) + cross(u, v) * (2.0f * w);
	}

	[[nodiscard]] constexpr mat4 to_mat4() const noexcept {
		mat4 res = mat4::identity();
		const float xx = x * x, yy = y * y, zz = z * z;
		const float xy = x * y, xz = x * z, yz = y * z;
		const float wx = w * x, wy = w * y, wz = w * z;

		res[0] = 1.0f - 2.0f * (yy + zz);
		res[1] = 2.0f * (xy + wz);
		res[2] = 2.0f * (xz - wy);

		res[4] = 2.0f * (xy - wz);
		res[5] = 1.0f - 2.0f * (xx + zz);
		res[6] = 2.0f * (yz + wx);

		res[8] = 2.0f * (xz + wy);
		res[9] = 2.0f * (yz - wx);
		res[10] = 1.0f - 2.0f * (xx + yy);

		return res;
	}
};

} // namespace math