#version 330

// ─── Constants ───────────────────────────────────────────────────────────────

const int   MAX_LIGHTS = 8;
const float PI         = 3.14159265359;

// ─── Varyings ────────────────────────────────────────────────────────────────

in vec3 v_world_pos;
in vec3 v_normal;

out vec4 frag_color;

// ─── Uniforms (maps byte-for-byte to scene::lighting_environment::gpu_data) ─

uniform vec4 material;                          // {metallic, roughness, ao, 0}
uniform vec4 albedo;                            // {r, g, b, alpha}
uniform vec4 camera_pos;                        // {x, y, z, num_lights}
uniform vec4 ambient;                           // {r, g, b, 0}
uniform vec4 light_pos_type[MAX_LIGHTS];        // {x, y, z, type}
uniform vec4 light_dir_range[MAX_LIGHTS];       // {dx, dy, dz, range}
uniform vec4 light_color_intensity[MAX_LIGHTS]; // {r, g, b, intensity}
uniform vec4 light_params[MAX_LIGHTS];          // {inner_cos, outer_cos, 0, 0}

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

    int  num_lights = int(camera_pos.w);
    vec3 Lo = vec3(0.0);

    for (int i = 0; i < MAX_LIGHTS; ++i) {
        if (i >= num_lights) break;

        int  type        = int(light_pos_type[i].w);
        vec3 light_color = light_color_intensity[i].xyz * light_color_intensity[i].w;

        vec3  L;
        float atten = 1.0;

        if (type == 0) {
            // ── Directional ──
            L = normalize(-light_dir_range[i].xyz);
        } else {
            // ── Point / Spot ──
            vec3  to_light = light_pos_type[i].xyz - v_world_pos;
            float dist     = length(to_light);
            L              = to_light / dist;
            atten          = attenuation(dist, light_dir_range[i].w);

            if (type == 2) {
                // Spot cone falloff
                float theta = dot(L, normalize(-light_dir_range[i].xyz));
                float inner = light_params[i].x;
                float outer = light_params[i].y;
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
        vec3  kD   = (vec3(1.0) - F) * (1.0 - metallic);
        float NdL  = max(dot(N, L), 0.0);

        Lo += (kD * base / PI + specular) * light_color * atten * NdL;
    }

    // Ambient
    vec3 color = ambient.xyz * base * ao + Lo;

    // Reinhard tone mapping + gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    frag_color = vec4(color, albedo.w);
}
