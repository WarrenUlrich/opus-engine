#version 330

// ─── Forward+ (Tiled Forward) Fragment Shader ────────────────────────────────
//
// Instead of evaluating all scene lights, this shader:
//   1. Determines which screen tile this fragment belongs to
//   2. Reads the tile's light list from a texture (populated by CPU culling)
//   3. Evaluates only the lights assigned to that tile
//
// This scales to 1000+ lights because each fragment only processes the
// handful of lights whose range overlaps its screen tile.

// ─── Tiling Constants ────────────────────────────────────────────────────────

const int   TILE_SIZE            = 16;
const int   MAX_LIGHTS_PER_TILE  = 64;
const int   TILE_DATA_STRIDE     = 17;   // ceil((1 + MAX_LIGHTS_PER_TILE) / 4)
const float PI                   = 3.14159265359;

// ─── Varyings ────────────────────────────────────────────────────────────────

in vec3 v_world_pos;
in vec3 v_normal;

out vec4 frag_color;

// ─── Per-Object Uniforms ─────────────────────────────────────────────────────

uniform vec4 material;        // {metallic, roughness, ao, 0}
uniform vec4 albedo;          // {r, g, b, alpha}
uniform vec4 camera_pos;      // {x, y, z, num_lights_total}
uniform vec4 ambient;         // {r, g, b, 0}
uniform vec4 tile_info;       // {num_tiles_x, TILE_SIZE, 0, 0}

// ─── Light Data Textures (set by CPU each frame) ────────────────────────────
//
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

// ─── Tile Data Access ────────────────────────────────────────────────────────

float fetch_tile_value(int tile_x, int tile_y, int p) {
    int base_x = tile_x * TILE_DATA_STRIDE;
    int texel_x = base_x + p / 4;
    int channel = p - (p / 4) * 4;       // p % 4 without modulo instruction
    return texelFetch(tile_data_tex, ivec2(texel_x, tile_y), 0)[channel];
}

// ─── PBR: Normal Distribution (GGX / Trowbridge-Reitz) ──────────────────────

float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a    = roughness * roughness;
    float a2   = a * a;
    float NdH  = max(dot(N, H), 0.0);
    float d    = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// ─── PBR: Geometry (Schlick-GGX + Smith) ─────────────────────────────────────

float geometry_schlick(float NdV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdV / (NdV * (1.0 - k) + k);
}

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometry_schlick(max(dot(N, V), 0.0), roughness)
         * geometry_schlick(max(dot(N, L), 0.0), roughness);
}

// ─── PBR: Fresnel (Schlick Approximation) ────────────────────────────────────

vec3 fresnel_schlick(float cos_theta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// ─── Attenuation: Smooth Windowed Inverse-Square ─────────────────────────────

float attenuation(float dist, float range) {
    float d2      = dist * dist;
    float factor  = d2 / (range * range);
    float falloff = max(1.0 - factor * factor, 0.0);
    return (falloff * falloff) / max(d2, 0.0001);
}

// ─── Main ────────────────────────────────────────────────────────────────────

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(camera_pos.xyz - v_world_pos);

    float metallic  = material.x;
    float roughness = material.y;
    float ao        = material.z;
    vec3  base      = albedo.xyz;

    // Dielectrics reflect ~4% at normal incidence; metals use albedo as F0
    vec3 F0 = mix(vec3(0.04), base, metallic);

    // ── Determine tile ──
    ivec2 tile = ivec2(gl_FragCoord.xy) / TILE_SIZE;
    int count  = int(fetch_tile_value(tile.x, tile.y, 0));
    count = min(count, MAX_LIGHTS_PER_TILE);

    // ── Accumulate lighting from this tile's light list ──
    vec3 Lo = vec3(0.0);

    for (int i = 0; i < MAX_LIGHTS_PER_TILE; ++i) {
        if (i >= count) break;

        int light_idx = int(fetch_tile_value(tile.x, tile.y, i + 1));

        // Fetch light properties from data texture
        vec4 pos_type  = texelFetch(light_data_tex, ivec2(light_idx, 0), 0);
        vec4 dir_range = texelFetch(light_data_tex, ivec2(light_idx, 1), 0);
        vec4 color_int = texelFetch(light_data_tex, ivec2(light_idx, 2), 0);
        vec4 params    = texelFetch(light_data_tex, ivec2(light_idx, 3), 0);

        int  type        = int(pos_type.w);
        vec3 light_color = color_int.xyz * color_int.w;

        vec3  L;
        float atten = 1.0;

        if (type == 0) {
            // ── Directional ──
            L = normalize(-dir_range.xyz);
        } else {
            // ── Point / Spot ──
            vec3  to_light = pos_type.xyz - v_world_pos;
            float dist     = length(to_light);
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

        // Cook-Torrance specular BRDF
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

    // Ambient
    vec3 color = ambient.xyz * base * ao + Lo;

    // Reinhard tone mapping + gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    frag_color = vec4(color, albedo.w);
}
