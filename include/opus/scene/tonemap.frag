#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D scene_tex;

void main() {
    vec4 hdr = texture(scene_tex, v_uv);
    vec3 color = hdr.rgb / (hdr.rgb + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    frag_color = vec4(color, hdr.a);
}
