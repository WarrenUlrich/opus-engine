#version 330

layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;

uniform mat4 mvp;
uniform mat4 mv;
uniform mat4 normal_mv;

out vec3 v_view_normal;
out float v_linear_depth;

void main() {
    vec4 view_pos   = mv * vec4(position, 1.0);
    v_linear_depth  = -view_pos.z;
    v_view_normal   = normalize((normal_mv * vec4(normal, 0.0)).xyz);
    gl_Position     = mvp * vec4(position, 1.0);
}
