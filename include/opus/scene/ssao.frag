#version 330

in vec2 v_uv;
out vec4 frag_color;

const int   KERNEL_SIZE = 64;
const float RADIUS      = 0.5;
const float BIAS        = 0.025;
const float POWER       = 1.5;

uniform vec4 samples[64];      // hemisphere kernel (.xyz used)
uniform vec4 ssao_params;      // {screen_w, screen_h, 0, 0}
uniform mat4 projection;

uniform sampler2D gbuffer_tex;
uniform sampler2D noise_tex;

void main() {
    vec2 noise_scale = ssao_params.xy / 4.0;

    vec4 gbuf        = texture(gbuffer_tex, v_uv);
    vec3 normal      = normalize(gbuf.xyz);
    float lin_depth  = gbuf.w;

    // Discard sky / background fragments
    if (lin_depth <= 0.0) {
        frag_color = vec4(1.0);
        return;
    }

    // Reconstruct view-space position from linear depth + screen UV.
    //   x_ndc = P[0][0] * x_view / (-z_view)  =>  x_view = x_ndc * depth / P[0][0]
    vec2 ndc     = v_uv * 2.0 - 1.0;
    vec3 frag_pos = vec3(
        ndc.x * lin_depth / projection[0][0],
        ndc.y * lin_depth / projection[1][1],
        -lin_depth
    );

    // Random rotation from tiled noise texture (4x4, wraps)
    vec3 random_vec = normalize(texture(noise_tex, v_uv * noise_scale).xyz);

    // Gramm-Schmidt: build tangent frame around normal
    vec3 tangent   = normalize(random_vec - normal * dot(random_vec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN       = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;

    for (int i = 0; i < KERNEL_SIZE; ++i) {
        // Orient sample from tangent-space hemisphere into view space
        vec3 sample_dir = TBN * samples[i].xyz;
        vec3 sample_pos = frag_pos + sample_dir * RADIUS;

        // Project sample back to screen
        vec4 clip      = projection * vec4(sample_pos, 1.0);
        vec2 sample_uv = (clip.xy / clip.w) * 0.5 + 0.5;

        // Out-of-bounds → skip
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 ||
            sample_uv.y < 0.0 || sample_uv.y > 1.0)
            continue;

        float surface_depth = texture(gbuffer_tex, sample_uv).w;
        if (surface_depth <= 0.0) continue;     // no geometry

        float sample_z = -sample_pos.z;         // positive linear depth of sample

        // Range-aware occlusion: ignore surfaces far from current fragment
        float range_check = smoothstep(0.0, 1.0, RADIUS / abs(lin_depth - surface_depth));
        occlusion += ((surface_depth <= sample_z - BIAS) ? 1.0 : 0.0) * range_check;
    }

    occlusion = 1.0 - (occlusion / float(KERNEL_SIZE));
    occlusion = pow(clamp(occlusion, 0.0, 1.0), POWER);

    frag_color = vec4(occlusion, 0.0, 0.0, 1.0);
}
