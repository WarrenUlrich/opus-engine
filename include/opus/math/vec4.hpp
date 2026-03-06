#pragma once

#include "vec3.hpp"
#include "mat4.hpp"

#include <compare>

namespace math {

class vec4 {
public:
    float x{0.0f}, y{0.0f}, z{0.0f}, w{0.0f};

    constexpr vec4() noexcept = default;
    constexpr vec4(float _x, float _y, float _z, float _w) noexcept : x{_x}, y{_y}, z{_z}, w{_w} {}
    explicit constexpr vec4(float scalar) noexcept : x{scalar}, y{scalar}, z{scalar}, w{scalar} {}
    
    // Homogenous coordinate constructor (crucial for MVP multiplication)
    constexpr vec4(const vec3& v, float _w) noexcept : x{v.x}, y{v.y}, z{v.z}, w{_w} {}

    [[nodiscard]] constexpr auto operator<=>(const vec4&) const = default;

    [[nodiscard]] constexpr vec4 operator-() const noexcept {
        return {-x, -y, -z, -w};
    }

    constexpr vec4& operator+=(const vec4& v) noexcept {
        x += v.x; y += v.y; z += v.z; w += v.w;
        return *this;
    }

    constexpr vec4& operator-=(const vec4& v) noexcept {
        x -= v.x; y -= v.y; z -= v.z; w -= v.w;
        return *this;
    }

    constexpr vec4& operator*=(float s) noexcept {
        x *= s; y *= s; z *= s; w *= s;
        return *this;
    }

    constexpr vec4& operator/=(float s) noexcept {
        const float inv = 1.0f / s; 
        x *= inv; y *= inv; z *= inv; w *= inv;
        return *this;
    }

    [[nodiscard]] friend constexpr vec4 operator+(vec4 lhs, const vec4& rhs) noexcept { return lhs += rhs; }
    [[nodiscard]] friend constexpr vec4 operator-(vec4 lhs, const vec4& rhs) noexcept { return lhs -= rhs; }
    [[nodiscard]] friend constexpr vec4 operator*(vec4 lhs, float s) noexcept { return lhs *= s; }
    [[nodiscard]] friend constexpr vec4 operator*(float s, vec4 rhs) noexcept { return rhs *= s; }
    [[nodiscard]] friend constexpr vec4 operator/(vec4 lhs, float s) noexcept { return lhs /= s; }
};

[[nodiscard]] constexpr float dot(const vec4& a, const vec4& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// Mat4 x Vec4 Multiplication (Hidden inside mat4 conceptually, but written here for logical flow)
// If you want to keep headers clean, add this to mat4.h instead.
[[nodiscard]] inline constexpr vec4 operator*(const mat4& m, const vec4& v) noexcept {
    return {
        m[0] * v.x + m[4] * v.y + m[8]  * v.z + m[12] * v.w,
        m[1] * v.x + m[5] * v.y + m[9]  * v.z + m[13] * v.w,
        m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w,
        m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w
    };
}

} // namespace math