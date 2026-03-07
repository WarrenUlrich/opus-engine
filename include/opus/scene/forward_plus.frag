#version 330

const int   TILE_SIZE            = 16;
const int   MAX_LIGHTS_PER_TILE  = 128;
const int   TILE_DATA_STRIDE     = 33;   // ceil((1 + MAX_LIGHTS_PER_TILE) / 4)
const float PI                   = 3.14159265359;

in vec3 v_world_pos;
in vec3 v_normal;
in vec4 v_light_pos;

out vec4 frag_color;

uniform vec4 material;        // {metallic, roughness, ao, 0}
uniform vec4 albedo;          // {r, g, b, alpha}
uniform vec4 camera_pos;      // {x, y, z, num_lights_total}
uniform vec4 ambient;         // {r, g, b, 0}
uniform vec4 tile_info;       // {num_tiles_x, TILE_SIZE, 0, 0}
uniform vec4 shadow_params;   // {map_resolution, min_bias, strength, max_bias}

// light_data_tex: RGBA32F, width=MAX_LIGHTS, height=4
//   row 0: (pos.x, pos.y, pos.z, type)
//   row 1: (dir.x, dir.y, dir.z, range)
//   row 2: (color.r, color.g, color.b, intensity)
//   row 3: (inner_cos, outer_cos, 0, 0)
//
// tile_data_tex: RGBA32F, width=(num_tiles_x * TILE_DATA_STRIDE), height=num_tiles_y
//   Each tile spans TILE_DATA_STRIDE texels. Values packed 4-per-texel:
//     value[0] = light count
//     value[1..N] = light indices

uniform sampler2D light_data_tex;
uniform sampler2D tile_data_tex;
uniform sampler2D shadow_map_tex;
uniform sampler2D ssao_tex;

float fetch_tile_value(int tile_x, int tile_y, int p) {
    int base_x = tile_x * TILE_DATA_STRIDE;
    int texel_x = base_x + p / 4;
    int channel = p - (p / 4) * 4;       // p % 4 without modulo instruction
    return texelFetch(tile_data_tex, ivec2(texel_x, tile_y), 0)[channel];
}

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

// Shadow mapping with 3x3 PCF (slope-scaled bias)
float calc_shadow(vec3 N, vec3 L) {
    vec3 proj = v_light_pos.xyz / v_light_pos.w;
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
    vec3 N = normalize(v_normal);
    vec3 V = normalize(camera_pos.xyz - v_world_pos);

    float metallic  = material.x;
    float roughness = material.y;
    float ao        = material.z;
    vec3  base      = albedo.xyz;

    // Dielectrics reflect ~4% at normal incidence; metals use albedo as F0
    vec3 F0 = mix(vec3(0.04), base, metallic);

    ivec2 tile = ivec2(gl_FragCoord.xy) / TILE_SIZE;
    int count  = int(fetch_tile_value(tile.x, tile.y, 0));
    count = min(count, MAX_LIGHTS_PER_TILE);

    vec3 Lo = vec3(0.0);

    for (int i = 0; i < MAX_LIGHTS_PER_TILE; ++i) {
        if (i >= count) break;

        int light_idx = int(fetch_tile_value(tile.x, tile.y, i + 1));

        vec4 pos_type  = texelFetch(light_data_tex, ivec2(light_idx, 0), 0);
        vec4 dir_range = texelFetch(light_data_tex, ivec2(light_idx, 1), 0);
        vec4 color_int = texelFetch(light_data_tex, ivec2(light_idx, 2), 0);
        vec4 params    = texelFetch(light_data_tex, ivec2(light_idx, 3), 0);

        int  type        = int(pos_type.w);
        vec3 light_color = color_int.xyz * color_int.w;

        vec3  L;
        float atten = 1.0;

        if (type == 0) {
            L = normalize(-dir_range.xyz);
            atten *= calc_shadow(N, L);
        } else {
            vec3  to_light = pos_type.xyz - v_world_pos;
            float dist     = length(to_light);
            if (dist > dir_range.w) continue;   // per-fragment range cull
            L              = to_light / dist;
            atten          = attenuation(dist, dir_range.w);

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

        // Energy conservation: what isn't reflected is refracted (diffuse)
        vec3  kD  = (vec3(1.0) - F) * (1.0 - metallic);
        float NdL = max(dot(N, L), 0.0);

        Lo += (kD * base / PI + specular) * light_color * atten * NdL;
    }

    // Screen-space ambient occlusion
    vec2 ssao_uv = gl_FragCoord.xy / tile_info.zw;
    float ssao_val = texture(ssao_tex, ssao_uv).r;

    vec3 color = ambient.xyz * base * ao * ssao_val + Lo;

    frag_color = vec4(color, albedo.w);
}
