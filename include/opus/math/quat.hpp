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
		const float half_angle = angle_rad * 0.5f;
		const float s = std::sin(half_angle);
		return {axis.x * s, axis.y * s, axis.z * s, std::cos(half_angle)};
	}

	[[nodiscard]] friend constexpr quat operator*(const quat &q1, const quat &q2) noexcept {
		return {q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
		        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
		        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
		        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z};
	}

	// Fast conversion to standard rotation matrix
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