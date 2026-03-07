#version 330

in vec3 v_view_normal;
in float v_linear_depth;

out vec4 frag_color;

void main() {
    frag_color = vec4(normalize(v_view_normal), v_linear_depth);
}
