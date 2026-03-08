#version 330

in vec3 v_world_pos;
in vec3 v_normal;

// MRT outputs
layout(location=0) out vec4 gPosition;   // world-space position + metallic
layout(location=1) out vec4 gNormal;     // world-space normal   + roughness
layout(location=2) out vec4 gAlbedo;     // albedo.rgb           + ao

uniform vec4 material;   // {metallic, roughness, ao, 0}
uniform vec4 albedo;     // {r, g, b, alpha}

void main() {
    gPosition = vec4(v_world_pos, material.x);       // .w = metallic
    gNormal   = vec4(normalize(v_normal), material.y); // .w = roughness
    gAlbedo   = vec4(albedo.rgb, material.z);         // .w = ao
}
