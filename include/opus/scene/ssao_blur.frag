#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform vec4 texel_size;       // {1/w, 1/h, 0, 0}

uniform sampler2D ssao_tex;

void main() {
    float result = 0.0;

    // 5x5 box blur — simple and effective for SSAO
    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            vec2 offset = vec2(float(x), float(y)) * texel_size.xy;
            result += texture(ssao_tex, v_uv + offset).r;
        }
    }

    frag_color = vec4(result / 25.0, 0.0, 0.0, 1.0);
}
