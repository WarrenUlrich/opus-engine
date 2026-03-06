#pragma once

#include <cmath>
#include <compare>

namespace math {

class vec3 {
public:
	float x{0.0f}, y{0.0f}, z{0.0f};

	constexpr vec3() noexcept = default;
	constexpr vec3(float _x, float _y, float _z) noexcept : x{_x}, y{_y}, z{_z} {}
	explicit constexpr vec3(float scalar) noexcept : x{scalar}, y{scalar}, z{scalar} {}

	[[nodiscard]] constexpr auto operator<=>(const vec3 &) const = default;

	[[nodiscard]] constexpr vec3 operator-() const noexcept { return {-x, -y, -z}; }

	constexpr vec3 &operator+=(const vec3 &v) noexcept {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	constexpr vec3 &operator-=(const vec3 &v) noexcept {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	constexpr vec3 &operator*=(const vec3 &v) noexcept {
		x *= v.x;
		y *= v.y;
		z *= v.z;
		return *this;
	}

	constexpr vec3 &operator*=(float s) noexcept {
		x *= s;
		y *= s;
		z *= s;
		return *this;
	}

	constexpr vec3 &operator/=(float s) noexcept {
		const float inv = 1.0f / s;
		x *= inv;
		y *= inv;
		z *= inv;
		return *this;
	}

	[[nodiscard]] friend constexpr vec3 operator+(vec3 lhs, const vec3 &rhs) noexcept {
		return lhs += rhs;
	}
	[[nodiscard]] friend constexpr vec3 operator-(vec3 lhs, const vec3 &rhs) noexcept {
		return lhs -= rhs;
	}
	[[nodiscard]] friend constexpr vec3 operator*(vec3 lhs, const vec3 &rhs) noexcept {
		return lhs *= rhs;
	}
	[[nodiscard]] friend constexpr vec3 operator*(vec3 lhs, float s) noexcept {
		return lhs *= s;
	}
	[[nodiscard]] friend constexpr vec3 operator*(float s, vec3 rhs) noexcept {
		return rhs *= s;
	}
	[[nodiscard]] friend constexpr vec3 operator/(vec3 lhs, float s) noexcept {
		return lhs /= s;
	}

	[[nodiscard]] constexpr float length_squared() const noexcept { return x * x + y * y + z * z; }

	[[nodiscard]] inline float length() const noexcept { return std::sqrt(length_squared()); }

	[[nodiscard]] inline vec3 normalized() const noexcept {
		const float len_sq = length_squared();
		if (len_sq == 0.0f) [[unlikely]]
			return *this;
		const float inv_len = 1.0f / std::sqrt(len_sq);
		return *this * inv_len;
	}

	inline void normalize() noexcept { *this = normalized(); }
};

[[nodiscard]] constexpr float dot(const vec3 &a, const vec3 &b) noexcept {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] constexpr vec3 cross(const vec3 &a, const vec3 &b) noexcept {
	return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

[[nodiscard]] constexpr vec3 lerp(const vec3 &a, const vec3 &b, float t) noexcept {
	return a + (b - a) * t;
}

// Reflects an incident vector 'i' off a surface with normal 'n'.
// 'n' must be normalized. 'i' points towards the surface.
// Essential for specular highlights (Phong/Blinn-Phong) and environment mapping.
[[nodiscard]] constexpr vec3 reflect(const vec3 &i, const vec3 &n) noexcept {
	return i - n * (2.0f * dot(i, n));
}

// Computes the refraction vector based on Snell's law.
// 'eta' is the relative index of refraction (eta_i / eta_t).
// Returns a zero vector on total internal reflection.
[[nodiscard]] inline vec3 refract(const vec3 &i, const vec3 &n, float eta) noexcept {
	const float cos_theta_i = dot(-i, n);
	const float sin2_theta_t = eta * eta * (1.0f - cos_theta_i * cos_theta_i);

	if (sin2_theta_t > 1.0f) [[unlikely]] {
		return {0.0f, 0.0f, 0.0f};
	}

	const float cos_theta_t = std::sqrt(1.0f - sin2_theta_t);
	return i * eta + n * (eta * cos_theta_i - cos_theta_t);
}

} // namespace math