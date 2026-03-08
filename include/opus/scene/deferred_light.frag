#version 330

const float PI = 3.14159265359;

in vec2 v_uv;
out vec4 frag_color;

// G-buffer textures
uniform sampler2D g_position_tex;
uniform sampler2D g_normal_tex;
uniform sampler2D g_albedo_tex;

// Shadow & SSAO
uniform sampler2D shadow_map_tex;
uniform sampler2D ssao_tex;

// Light data texture
uniform sampler2D light_data_tex;

uniform vec4 camera_pos;      // {x, y, z, num_lights}
uniform vec4 ambient;         // {r, g, b, 0}
uniform vec4 shadow_params;   // {map_resolution, min_bias, strength, max_bias}
uniform mat4 light_vp;        // light-space view-projection

// Normal Distribution (GGX / Trowbridge-Reitz)
float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a    = roughness * roughness;
    float a2   = a * a;
    float NdH  = max(dot(N, H), 0.0);
    float d    = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Geometry (Schlick-GGX + Smith)
float geometry_schlick(float NdV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdV / (NdV * (1.0 - k) + k);
}

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometry_schlick(max(dot(N, V), 0.0), roughness)
         * geometry_schlick(max(dot(N, L), 0.0), roughness);
}

// Fresnel (Schlick approximation)
vec3 fresnel_schlick(float cos_theta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Smooth windowed inverse-square attenuation
float attenuation(float dist, float range) {
    float d2      = dist * dist;
    float factor  = d2 / (range * range);
    float falloff = max(1.0 - factor * factor, 0.0);
    return (falloff * falloff) / max(d2, 0.0001);
}

// Shadow mapping with 3x3 PCF
float calc_shadow(vec3 world_pos, vec3 N, vec3 L) {
    vec4 lp  = light_vp * vec4(world_pos, 1.0);
    vec3 proj = lp.xyz / lp.w;
    proj = proj * 0.5 + 0.5;

    if (proj.x < 0.0 || proj.x > 1.0 ||
        proj.y < 0.0 || proj.y > 1.0 || proj.z > 1.0)
        return 1.0;

    float current_depth = proj.z;
    float min_bias      = shadow_params.y;
    float strength      = shadow_params.z;
    float max_bias      = shadow_params.w;
    float bias          = max(max_bias * (1.0 - max(dot(N, L), 0.0)), min_bias);

    float shadow     = 0.0;
    float texel_size = 1.0 / shadow_params.x;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float pcf_depth = texture(shadow_map_tex,
                proj.xy + vec2(x, y) * texel_size).r;
            shadow += (current_depth - bias > pcf_depth) ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;
    return 1.0 - shadow * strength;
}

void main() {
    vec4 g_pos = texture(g_position_tex, v_uv);
    vec4 g_nor = texture(g_normal_tex, v_uv);
    vec4 g_alb = texture(g_albedo_tex, v_uv);

    vec3  world_pos = g_pos.xyz;
    float metallic  = g_pos.w;
    vec3  N         = normalize(g_nor.xyz);
    float roughness = g_nor.w;
    vec3  base      = g_alb.rgb;
    float ao        = g_alb.w;

    // Discard empty G-buffer fragments (no geometry written)
    if (dot(g_nor.xyz, g_nor.xyz) < 0.001) {
        frag_color = vec4(0.0);
        return;
    }

    vec3 V  = normalize(camera_pos.xyz - world_pos);
    vec3 F0 = mix(vec3(0.04), base, metallic);

    int num_lights = int(camera_pos.w);
    vec3 Lo = vec3(0.0);

    for (int i = 0; i < 1024; ++i) {
        if (i >= num_lights) break;

        vec4 pos_type  = texelFetch(light_data_tex, ivec2(i, 0), 0);
        vec4 dir_range = texelFetch(light_data_tex, ivec2(i, 1), 0);
        vec4 color_int = texelFetch(light_data_tex, ivec2(i, 2), 0);
        vec4 params    = texelFetch(light_data_tex, ivec2(i, 3), 0);

        int  type        = int(pos_type.w);
        vec3 light_color = color_int.xyz * color_int.w;

        vec3  L;
        float atten = 1.0;

        if (type == 0) {
            // Directional
            L = normalize(-dir_range.xyz);
            atten *= calc_shadow(world_pos, N, L);
        } else {
            vec3  to_light = pos_type.xyz - world_pos;
            float dist     = length(to_light);
            if (dist > dir_range.w) continue;
            L     = to_light / dist;
            atten = attenuation(dist, dir_range.w);

            if (type == 2) {
                // Spot cone falloff
                float theta = dot(L, normalize(-dir_range.xyz));
                float inner = params.x;
                float outer = params.y;
                atten *= clamp((theta - outer) / (inner - outer), 0.0, 1.0);
            }
        }

        vec3 H = normalize(V + L);

        float D = distribution_ggx(N, H, roughness);
        float G = geometry_smith(N, V, L, roughness);
        vec3  F = fresnel_schlick(max(dot(H, V), 0.0), F0);

        vec3  numerator   = D * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3  specular    = numerator / denominator;

        vec3  kD  = (vec3(1.0) - F) * (1.0 - metallic);
        float NdL = max(dot(N, L), 0.0);

        Lo += (kD * base / PI + specular) * light_color * atten * NdL;
    }

    // SSAO
    float ssao_val = texture(ssao_tex, v_uv).r;

    vec3 color = ambient.xyz * base * ao * ssao_val + Lo;
    frag_color = vec4(color, 1.0);
}
