#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D scene_tex;
uniform vec4 screen_params;   // {1/width, 1/height, vignette_strength, 0}

// ACES filmic tonemapping (Narkowicz 2015)
vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// FXAA-style luminance-weighted edge detection for cheap anti-aliasing
vec3 fxaa_pass(vec2 uv, vec2 texel) {
    vec3 center = texture(scene_tex, uv).rgb;

    vec3 nw = texture(scene_tex, uv + vec2(-texel.x, -texel.y)).rgb;
    vec3 ne = texture(scene_tex, uv + vec2( texel.x, -texel.y)).rgb;
    vec3 sw = texture(scene_tex, uv + vec2(-texel.x,  texel.y)).rgb;
    vec3 se = texture(scene_tex, uv + vec2( texel.x,  texel.y)).rgb;

    vec3 luma_coeff = vec3(0.299, 0.587, 0.114);
    float luma_c  = dot(center, luma_coeff);
    float luma_nw = dot(nw, luma_coeff);
    float luma_ne = dot(ne, luma_coeff);
    float luma_sw = dot(sw, luma_coeff);
    float luma_se = dot(se, luma_coeff);

    float luma_min = min(luma_c, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    float luma_max = max(luma_c, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));
    float luma_range = luma_max - luma_min;

    // Skip if contrast is low
    if (luma_range < max(0.0312, luma_max * 0.125))
        return center;

    float dir_x = -((luma_nw + luma_ne) - (luma_sw + luma_se));
    float dir_y =  ((luma_nw + luma_sw) - (luma_ne + luma_se));
    float dir_reduce = max((luma_nw + luma_ne + luma_sw + luma_se) * 0.03125, 0.0078125);
    float rcp_dir_min = 1.0 / (min(abs(dir_x), abs(dir_y)) + dir_reduce);
    vec2 dir = clamp(vec2(dir_x, dir_y) * rcp_dir_min, -8.0, 8.0) * texel;

    vec3 result_a = 0.5 * (
        texture(scene_tex, uv + dir * (1.0/3.0 - 0.5)).rgb +
        texture(scene_tex, uv + dir * (2.0/3.0 - 0.5)).rgb
    );
    vec3 result_b = result_a * 0.5 + 0.25 * (
        texture(scene_tex, uv + dir * -0.5).rgb +
        texture(scene_tex, uv + dir *  0.5).rgb
    );

    float luma_b = dot(result_b, luma_coeff);
    if (luma_b < luma_min || luma_b > luma_max)
        return result_a;
    return result_b;
}

void main() {
    vec2 texel = screen_params.xy;
    float vignette_strength = screen_params.z;

    // FXAA
    vec3 color = fxaa_pass(v_uv, texel);

    // No FXAA (for comparison)
    // vec3 color = texture(scene_tex, v_uv).rgb;

    // ACES tonemapping
    color = aces_tonemap(color);

    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    // Vignette
    vec2 vig_uv = v_uv * 2.0 - 1.0;
    float vig = 1.0 - dot(vig_uv, vig_uv) * vignette_strength;
    color *= clamp(vig, 0.0, 1.0);

    frag_color = vec4(color, 1.0);
}
