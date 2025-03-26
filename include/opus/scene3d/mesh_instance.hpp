#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include <webgpu/webgpu_cpp.h>

#include "../math/vector2.hpp"
#include "../math/vector3.hpp"

namespace scene3d {

class mesh_instance {
public:
  struct vertex {
    math::vector3<float> position; // Vertex position
    math::vector3<float> normal;   // Normal for lighting
    math::vector2<float> uv;       // Texture coordinates
  };

  mesh_instance() = default;

  std::uint32_t get_index_count() const noexcept { return _index_count; }

  const wgpu::Buffer &get_vertex_buffer() const noexcept {
    return _vertex_buffer;
  }

  const wgpu::Buffer &get_index_buffer() const noexcept {
    return _index_buffer;
  }

  static mesh_instance create_plane(const wgpu::Device &device, float width,
                                    float depth, uint32_t width_segments = 1,
                                    uint32_t depth_segments = 1) noexcept {
    std::vector<vertex> vertices;
    std::vector<uint16_t> indices;

    float half_width = width * 0.5f;
    float half_depth = depth * 0.5f;

    // Generate vertices
    for (uint32_t z = 0; z <= depth_segments; ++z) {
      float v = static_cast<float>(z) / static_cast<float>(depth_segments);
      float pz = v * depth - half_depth;

      for (uint32_t x = 0; x <= width_segments; ++x) {
        float u = static_cast<float>(x) / static_cast<float>(width_segments);
        float px = u * width - half_width;

        // Position on the plane
        math::vector3<float> position = {px, 0.0f, pz};

        // Normal points up for a horizontal plane
        math::vector3<float> normal = {0.0f, 1.0f, 0.0f};

        // UV coordinates
        math::vector2<float> uv = {u,
                                   1.0f - v}; // Flip v for texture orientation

        vertices.push_back({position, normal, uv});
      }
    }

    // Generate indices
    for (uint32_t z = 0; z < depth_segments; ++z) {
      for (uint32_t x = 0; x < width_segments; ++x) {
        uint32_t a = z * (width_segments + 1) + x;
        uint32_t b = a + 1;
        uint32_t c = a + (width_segments + 1);
        uint32_t d = c + 1;

        // Two triangles per grid cell (CCW order)
        indices.push_back(a);
        indices.push_back(c);
        indices.push_back(b);

        indices.push_back(b);
        indices.push_back(c);
        indices.push_back(d);
      }
    }

    // Create and return buffers
    return _create_mesh_buffers(device, vertices, indices);
  }

  static mesh_instance create_cube(const wgpu::Device &device,
                                   float size) noexcept {
    std::vector<vertex> vertices = {
        // Front face (smoothed normals)
        {{-size, -size, size}, {-0.577f, -0.577f, 0.577f}, {0.0f, 0.0f}},
        {{size, -size, size}, {0.577f, -0.577f, 0.577f}, {1.0f, 0.0f}},
        {{size, size, size}, {0.577f, 0.577f, 0.577f}, {1.0f, 1.0f}},
        {{-size, size, size}, {-0.577f, 0.577f, 0.577f}, {0.0f, 1.0f}},

        // Back face (smoothed normals)
        {{size, -size, -size}, {0.577f, -0.577f, -0.577f}, {0.0f, 0.0f}},
        {{-size, -size, -size}, {-0.577f, -0.577f, -0.577f}, {1.0f, 0.0f}},
        {{-size, size, -size}, {-0.577f, 0.577f, -0.577f}, {1.0f, 1.0f}},
        {{size, size, -size}, {0.577f, 0.577f, -0.577f}, {0.0f, 1.0f}},

        // Left face (smoothed normals)
        {{-size, -size, -size}, {-0.577f, -0.577f, -0.577f}, {0.0f, 0.0f}},
        {{-size, -size, size}, {-0.577f, -0.577f, 0.577f}, {1.0f, 0.0f}},
        {{-size, size, size}, {-0.577f, 0.577f, 0.577f}, {1.0f, 1.0f}},
        {{-size, size, -size}, {-0.577f, 0.577f, -0.577f}, {0.0f, 1.0f}},

        // Right face (smoothed normals)
        {{size, -size, size}, {0.577f, -0.577f, 0.577f}, {0.0f, 0.0f}},
        {{size, -size, -size}, {0.577f, -0.577f, -0.577f}, {1.0f, 0.0f}},
        {{size, size, -size}, {0.577f, 0.577f, -0.577f}, {1.0f, 1.0f}},
        {{size, size, size}, {0.577f, 0.577f, 0.577f}, {0.0f, 1.0f}},

        // Top face (smoothed normals)
        {{-size, size, size}, {-0.577f, 0.577f, 0.577f}, {0.0f, 0.0f}},
        {{size, size, size}, {0.577f, 0.577f, 0.577f}, {1.0f, 0.0f}},
        {{size, size, -size}, {0.577f, 0.577f, -0.577f}, {1.0f, 1.0f}},
        {{-size, size, -size}, {-0.577f, 0.577f, -0.577f}, {0.0f, 1.0f}},

        // Bottom face (smoothed normals)
        {{-size, -size, -size}, {-0.577f, -0.577f, -0.577f}, {0.0f, 0.0f}},
        {{size, -size, -size}, {0.577f, -0.577f, -0.577f}, {1.0f, 0.0f}},
        {{size, -size, size}, {0.577f, -0.577f, 0.577f}, {1.0f, 1.0f}},
        {{-size, -size, size}, {-0.577f, -0.577f, 0.577f}, {0.0f, 1.0f}}};

    // Indices for 12 triangles (36 indices)
    std::vector<uint16_t> indices = {
        0,  1,  2,  2,  3,  0,  // Front face
        4,  5,  6,  6,  7,  4,  // Back face
        8,  9,  10, 10, 11, 8,  // Left face
        12, 13, 14, 14, 15, 12, // Right face
        16, 17, 18, 18, 19, 16, // Top face
        20, 21, 22, 22, 23, 20  // Bottom face
    };

    // Create vertex buffer
    wgpu::BufferDescriptor vertexBufferDesc = {};
    vertexBufferDesc.usage =
        wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst;
    vertexBufferDesc.size = sizeof(vertex) * vertices.size();
    vertexBufferDesc.mappedAtCreation = true;

    wgpu::Buffer vertexBuffer = device.CreateBuffer(&vertexBufferDesc);
    std::memcpy(vertexBuffer.GetMappedRange(), vertices.data(),
                sizeof(vertex) * vertices.size());
    vertexBuffer.Unmap();

    // Create index buffer
    wgpu::BufferDescriptor indexBufferDesc = {};
    indexBufferDesc.usage =
        wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst;
    indexBufferDesc.size = sizeof(uint16_t) * indices.size();
    indexBufferDesc.mappedAtCreation = true;

    wgpu::Buffer indexBuffer = device.CreateBuffer(&indexBufferDesc);
    std::memcpy(indexBuffer.GetMappedRange(), indices.data(),
                sizeof(uint16_t) * indices.size());
    indexBuffer.Unmap();

    // Return the mesh instance
    return mesh_instance(std::move(vertexBuffer), std::move(indexBuffer),
                         indices.size());
  }

  static mesh_instance create_sphere(const wgpu::Device &device, float radius,
                                     uint32_t latitude_count,
                                     uint32_t longitude_count) noexcept {
    std::vector<vertex> vertices;
    std::vector<uint16_t> indices;

    for (uint32_t lat = 0; lat <= latitude_count; ++lat) {
      float theta = static_cast<float>(lat) * M_PI /
                    static_cast<float>(latitude_count); // From 0 to PI
      float sinTheta = std::sin(theta);
      float cosTheta = std::cos(theta);

      for (uint32_t lon = 0; lon <= longitude_count; ++lon) {
        float phi = static_cast<float>(lon) * 2.0f * M_PI /
                    static_cast<float>(longitude_count); // From 0 to 2*PI
        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);

        // Spherical to Cartesian coordinates
        math::vector3<float> position = {radius * cosPhi * sinTheta,
                                         radius * cosTheta,
                                         radius * sinPhi * sinTheta};

        // Normal is the normalized position for a sphere
        math::vector3<float> normal = position.normalized();

        // UV coordinates for texturing
        math::vector2<float> uv = {
            static_cast<float>(lon) / static_cast<float>(longitude_count),
            static_cast<float>(lat) / static_cast<float>(latitude_count)};

        // Add vertex to the mesh
        vertices.push_back({position, normal, uv});
      }
    }

    // Generate indices for sphere triangles (using a strip-like grid structure)
    for (uint32_t lat = 0; lat < latitude_count; ++lat) {
      for (uint32_t lon = 0; lon < longitude_count; ++lon) {
        uint32_t first = lat * (longitude_count + 1) + lon;
        uint32_t second = first + longitude_count + 1;

        // Triangle 1
        indices.push_back(first);
        indices.push_back(second);
        indices.push_back(first + 1);

        // Triangle 2
        indices.push_back(second);
        indices.push_back(second + 1);
        indices.push_back(first + 1);
      }
    }

    // Create vertex buffer
    wgpu::BufferDescriptor vertexBufferDesc = {};
    vertexBufferDesc.usage =
        wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst;
    vertexBufferDesc.size = sizeof(vertex) * vertices.size();
    vertexBufferDesc.mappedAtCreation = true;

    wgpu::Buffer vertexBuffer = device.CreateBuffer(&vertexBufferDesc);
    std::memcpy(vertexBuffer.GetMappedRange(), vertices.data(),
                sizeof(vertex) * vertices.size());
    vertexBuffer.Unmap();

    // Create index buffer
    wgpu::BufferDescriptor indexBufferDesc = {};
    indexBufferDesc.usage =
        wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst;
    indexBufferDesc.size = sizeof(uint16_t) * indices.size();
    indexBufferDesc.mappedAtCreation = true;

    wgpu::Buffer indexBuffer = device.CreateBuffer(&indexBufferDesc);
    std::memcpy(indexBuffer.GetMappedRange(), indices.data(),
                sizeof(uint16_t) * indices.size());
    indexBuffer.Unmap();

    // Return the mesh instance
    return mesh_instance(std::move(vertexBuffer), std::move(indexBuffer),
                         indices.size());
  }

  static mesh_instance create_cone(const wgpu::Device &device, float radius,
                                   float height, uint32_t radial_segments = 32,
                                   bool capped = true) noexcept {
    std::vector<vertex> vertices;
    std::vector<uint16_t> indices;
    float half_height = height * 0.5f;

    // Cone apex (top point)
    math::vector3<float> apex = {0.0f, half_height, 0.0f};

    // Generate vertices for the sides
    for (uint32_t i = 0; i <= radial_segments; ++i) {
      float u = static_cast<float>(i) / static_cast<float>(radial_segments);
      float theta = u * 2.0f * M_PI;
      float cos_theta = std::cos(theta);
      float sin_theta = std::sin(theta);

      // Position at the bottom circle
      math::vector3<float> position = {radius * cos_theta, -half_height,
                                       radius * sin_theta};

      // Calculate normal for cone side
      math::vector3<float> to_apex = apex - position;
      math::vector3<float> tangent = {-sin_theta, 0.0f,
                                      cos_theta}; // Tangent to the base circle
      math::vector3<float> normal = to_apex.cross(tangent).cross(to_apex);
      normal = normal.normalized();

      // UV coordinates
      math::vector2<float> uv = {u, 0.0f}; // Bottom of the cone

      // Add base vertex
      vertices.push_back({position, normal, uv});

      // Add a corresponding apex vertex with the same normal for correct
      // shading
      vertices.push_back({apex, normal, {u, 1.0f}});
    }

    // Generate indices for the sides
    for (uint32_t i = 0; i < radial_segments; ++i) {
      uint32_t a = i * 2;       // Current base vertex
      uint32_t b = a + 1;       // Current apex reference
      uint32_t c = (i + 1) * 2; // Next base vertex

      // Triangle from current point on base to apex to next point on base (CCW
      // order)
      indices.push_back(a);
      indices.push_back(b);
      indices.push_back(c);
    }

    // Add cap at the bottom if requested
    if (capped) {
      uint32_t base_index = vertices.size();

      // Add center vertex for base cap
      math::vector3<float> bottom_center = {0.0f, -half_height, 0.0f};
      math::vector3<float> bottom_normal = {0.0f, -1.0f, 0.0f};

      // Bottom center vertex
      vertices.push_back({bottom_center, bottom_normal, {0.5f, 0.5f}});

      // Add vertices for edges of base cap
      for (uint32_t i = 0; i <= radial_segments; ++i) {
        float u = static_cast<float>(i) / static_cast<float>(radial_segments);
        float theta = u * 2.0f * M_PI;
        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);

        // Position on the circle at the bottom
        math::vector3<float> bottom_position = {
            radius * cos_theta, -half_height, radius * sin_theta};

        // UV coordinates (circular mapping)
        math::vector2<float> bottom_uv = {0.5f + 0.5f * cos_theta,
                                          0.5f + 0.5f * sin_theta};

        // Add vertex for bottom cap
        vertices.push_back({bottom_position, bottom_normal, bottom_uv});
      }

      // Generate indices for the base cap
      uint32_t bottom_center_index = base_index;
      uint32_t first_bottom_rim_index = base_index + 1;
      for (uint32_t i = 0; i < radial_segments; ++i) {
        uint32_t next_i = i + 1;

        // Bottom cap triangles (fan formation) - CCW when viewed from outside
        // (below)
        indices.push_back(bottom_center_index);
        indices.push_back(first_bottom_rim_index + i);
        indices.push_back(first_bottom_rim_index + next_i);
      }
    }

    // Create and return buffers
    return _create_mesh_buffers(device, vertices, indices);
  }

  static mesh_instance create_cylinder(const wgpu::Device &device, float radius,
                                       float height,
                                       uint32_t radial_segments = 32,
                                       bool capped = true) noexcept {
    std::vector<vertex> vertices;
    std::vector<uint16_t> indices;

    float half_height = height * 0.5f;

    // Generate vertices for the sides
    for (uint32_t i = 0; i <= radial_segments; ++i) {
      float u = static_cast<float>(i) / static_cast<float>(radial_segments);
      float theta = u * 2.0f * M_PI;

      float cos_theta = std::cos(theta);
      float sin_theta = std::sin(theta);

      // Create vertices for top and bottom of the cylinder side
      for (int j = 0; j < 2; ++j) {
        float v = j;
        float y = (v == 0) ? -half_height : half_height;

        // Position
        math::vector3<float> position = {radius * cos_theta, y,
                                         radius * sin_theta};

        // Normal (pointing outward from the cylinder)
        math::vector3<float> normal = {cos_theta, 0.0f, sin_theta};
        normal = normal.normalized();

        // UV coordinates
        math::vector2<float> uv = {u, v};

        vertices.push_back({position, normal, uv});
      }
    }

    // Generate indices for the sides
    for (uint32_t i = 0; i < radial_segments; ++i) {
      uint32_t a = i * 2;
      uint32_t b = a + 1;
      uint32_t c = (i + 1) * 2;
      uint32_t d = c + 1;

      // Two triangles per side segment (CCW order)
      indices.push_back(a);
      indices.push_back(b);
      indices.push_back(c);

      indices.push_back(b);
      indices.push_back(d);
      indices.push_back(c);
    }

    // Generate cap vertices and indices if requested
    if (capped) {
      // ---- TOP CAP ----
      // Index where top cap vertices start
      uint32_t top_cap_start = vertices.size();

      // Add top center vertex
      math::vector3<float> top_center = {0.0f, half_height, 0.0f};
      math::vector3<float> top_normal = {0.0f, 1.0f, 0.0f};
      vertices.push_back({top_center, top_normal, {0.5f, 0.5f}});

      // Add vertices for top cap rim
      for (uint32_t i = 0; i <= radial_segments; ++i) {
        float u = static_cast<float>(i) / static_cast<float>(radial_segments);
        float theta = u * 2.0f * M_PI;

        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);

        // Position on the circle at the top
        math::vector3<float> position = {radius * cos_theta, half_height,
                                         radius * sin_theta};

        // UV coordinates (circular mapping)
        math::vector2<float> uv = {0.5f + 0.5f * cos_theta,
                                   0.5f + 0.5f * sin_theta};

        vertices.push_back({position, top_normal, uv});
      }

      // Generate top cap triangles - REVERSED ORDER for correct CCW winding
      uint32_t top_center_index = top_cap_start;
      for (uint32_t i = 0; i < radial_segments; ++i) {
        indices.push_back(top_center_index);
        indices.push_back(top_cap_start + 2 + i); // Reversed these two lines
        indices.push_back(top_cap_start + 1 + i); // to fix winding order
      }

      // ---- BOTTOM CAP ----
      // Index where bottom cap vertices start
      uint32_t bottom_cap_start = vertices.size();

      // Add bottom center vertex
      math::vector3<float> bottom_center = {0.0f, -half_height, 0.0f};
      math::vector3<float> bottom_normal = {0.0f, -1.0f, 0.0f};
      vertices.push_back({bottom_center, bottom_normal, {0.5f, 0.5f}});

      // Add vertices for bottom cap rim
      for (uint32_t i = 0; i <= radial_segments; ++i) {
        float u = static_cast<float>(i) / static_cast<float>(radial_segments);
        float theta = u * 2.0f * M_PI;

        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);

        // Position on the circle at the bottom
        math::vector3<float> position = {radius * cos_theta, -half_height,
                                         radius * sin_theta};

        // UV coordinates (circular mapping)
        math::vector2<float> uv = {0.5f + 0.5f * cos_theta,
                                   0.5f + 0.5f * sin_theta};

        vertices.push_back({position, bottom_normal, uv});
      }

      uint32_t bottom_center_index = bottom_cap_start;
      for (uint32_t i = 0; i < radial_segments; ++i) {
        indices.push_back(bottom_center_index);
        indices.push_back(bottom_cap_start + 1 + i);
        indices.push_back(bottom_cap_start + 2 + i);
      }
    }

    // Create and return buffers
    return _create_mesh_buffers(device, vertices, indices);
  }

  static mesh_instance create_torus(const wgpu::Device &device, float radius,
                                    float tube_radius,
                                    uint32_t radial_segments = 32,
                                    uint32_t tubular_segments = 32) noexcept {
    std::vector<vertex> vertices;
    std::vector<uint16_t> indices;

    // Generate vertices
    for (uint32_t i = 0; i <= radial_segments; ++i) {
      float u = static_cast<float>(i) / static_cast<float>(radial_segments) *
                2.0f * M_PI;

      // Calculate the position of the center of the tube at this radial segment
      float cos_u = std::cos(u);
      float sin_u = std::sin(u);

      for (uint32_t j = 0; j <= tubular_segments; ++j) {
        float v = static_cast<float>(j) / static_cast<float>(tubular_segments) *
                  2.0f * M_PI;

        float cos_v = std::cos(v);
        float sin_v = std::sin(v);

        // Calculate position around the tube
        math::vector3<float> position = {
            (radius + tube_radius * cos_v) * cos_u, tube_radius * sin_v,
            (radius + tube_radius * cos_v) * sin_u};

        // Calculate normal
        math::vector3<float> normal = {cos_u * cos_v, sin_v, sin_u * cos_v};
        normal = normal.normalized();

        // Calculate UV coordinates
        math::vector2<float> uv = {
            static_cast<float>(i) / static_cast<float>(radial_segments),
            static_cast<float>(j) / static_cast<float>(tubular_segments)};

        vertices.push_back({position, normal, uv});
      }
    }

    // Generate indices
    for (uint32_t i = 0; i < radial_segments; ++i) {
      for (uint32_t j = 0; j < tubular_segments; ++j) {
        uint32_t a = i * (tubular_segments + 1) + j;
        uint32_t b = a + 1;
        uint32_t c = (i + 1) * (tubular_segments + 1) + j;
        uint32_t d = c + 1;

        // Corrected winding order for CCW front faces
        indices.push_back(a);
        indices.push_back(b);
        indices.push_back(c);

        indices.push_back(b);
        indices.push_back(d);
        indices.push_back(c);
      }
    }

    // Create and return buffers
    return _create_mesh_buffers(device, vertices, indices);
  }

  static mesh_instance create_capsule(const wgpu::Device &device, float radius,
                                      float height,
                                      uint32_t radial_segments = 32,
                                      uint32_t height_segments = 1,
                                      uint32_t cap_segments = 8) noexcept {
    std::vector<vertex> vertices;
    std::vector<uint16_t> indices;

    const float half_height = height * 0.5f;

    // Generate the top hemisphere vertices
    for (uint32_t lat = 0; lat <= cap_segments; ++lat) {
      // Calculate the current latitude angle (0 at the top, PI/2 at the
      // equator)
      float theta = static_cast<float>(lat) * M_PI * 0.5f /
                    static_cast<float>(cap_segments);
      float sin_theta = std::sin(theta);
      float cos_theta = std::cos(theta);

      // y-position relative to the top of the cylinder
      float y = half_height + radius * cos_theta;
      float r = radius * sin_theta; // radius at this latitude

      for (uint32_t lon = 0; lon <= radial_segments; ++lon) {
        float phi = static_cast<float>(lon) * 2.0f * M_PI /
                    static_cast<float>(radial_segments);
        float sin_phi = std::sin(phi);
        float cos_phi = std::cos(phi);

        // Position
        math::vector3<float> position = {r * cos_phi, y, r * sin_phi};

        // Normal points outward
        math::vector3<float> normal = {sin_theta * cos_phi, cos_theta,
                                       sin_theta * sin_phi};
        normal = normal.normalized();

        // UV coordinates (map to top quarter of texture)
        math::vector2<float> uv = {
            static_cast<float>(lon) / static_cast<float>(radial_segments),
            1.0f - static_cast<float>(lat) / static_cast<float>(cap_segments) *
                       0.25f};

        vertices.push_back({position, normal, uv});
      }
    }

    // Generate the middle cylinder vertices
    for (uint32_t h = 0; h <= height_segments; ++h) {
      float v = static_cast<float>(h) / static_cast<float>(height_segments);
      float y = half_height - v * height;

      for (uint32_t lon = 0; lon <= radial_segments; ++lon) {
        float u = static_cast<float>(lon) / static_cast<float>(radial_segments);
        float phi = u * 2.0f * M_PI;
        float cos_phi = std::cos(phi);
        float sin_phi = std::sin(phi);

        // Position
        math::vector3<float> position = {radius * cos_phi, y, radius * sin_phi};

        // Normal points outward horizontally
        math::vector3<float> normal = {cos_phi, 0.0f, sin_phi};
        normal = normal.normalized();

        // UV coordinates (middle half of texture)
        math::vector2<float> uv = {u, 0.75f - v * 0.5f};

        vertices.push_back({position, normal, uv});
      }
    }

    // Generate the bottom hemisphere vertices
    for (uint32_t lat = 0; lat <= cap_segments; ++lat) {
      // Calculate the current latitude angle (0 at the bottom, PI/2 at the
      // equator)
      float theta = static_cast<float>(lat) * M_PI * 0.5f /
                    static_cast<float>(cap_segments);
      float sin_theta = std::sin(theta);
      float cos_theta = std::cos(theta);

      // y-position relative to the bottom of the cylinder
      float y = -half_height - radius * cos_theta;
      float r = radius * sin_theta; // radius at this latitude

      for (uint32_t lon = 0; lon <= radial_segments; ++lon) {
        float phi = static_cast<float>(lon) * 2.0f * M_PI /
                    static_cast<float>(radial_segments);
        float sin_phi = std::sin(phi);
        float cos_phi = std::cos(phi);

        // Position
        math::vector3<float> position = {r * cos_phi, y, r * sin_phi};

        // Normal points outward
        math::vector3<float> normal = {sin_theta * cos_phi, -cos_theta,
                                       sin_theta * sin_phi};
        normal = normal.normalized();

        // UV coordinates (bottom quarter of texture)
        math::vector2<float> uv = {
            static_cast<float>(lon) / static_cast<float>(radial_segments),
            0.25f - static_cast<float>(lat) / static_cast<float>(cap_segments) *
                        0.25f};

        vertices.push_back({position, normal, uv});
      }
    }

    // Calculate starting indices for each section
    uint32_t top_hemisphere_vertices =
        (cap_segments + 1) * (radial_segments + 1);
    uint32_t cylinder_vertices = (height_segments + 1) * (radial_segments + 1);

    // Generate indices for top hemisphere
    for (uint32_t lat = 0; lat < cap_segments; ++lat) {
      uint32_t current_row = lat * (radial_segments + 1);
      uint32_t next_row = (lat + 1) * (radial_segments + 1);

      for (uint32_t lon = 0; lon < radial_segments; ++lon) {
        indices.push_back(current_row + lon);
        indices.push_back(current_row + lon + 1);
        indices.push_back(next_row + lon + 1);

        indices.push_back(current_row + lon);
        indices.push_back(next_row + lon + 1);
        indices.push_back(next_row + lon);
      }
    }

    // Generate indices for cylinder
    uint32_t cylinder_start = top_hemisphere_vertices;
    for (uint32_t h = 0; h < height_segments; ++h) {
      uint32_t current_row = cylinder_start + h * (radial_segments + 1);
      uint32_t next_row = current_row + (radial_segments + 1);

      for (uint32_t lon = 0; lon < radial_segments; ++lon) {
        // First triangle
        indices.push_back(current_row + lon);
        indices.push_back(current_row + lon + 1);
        indices.push_back(next_row + lon);

        // Second triangle
        indices.push_back(current_row + lon + 1);
        indices.push_back(next_row + lon + 1);
        indices.push_back(next_row + lon);
      }
    }

    // Generate indices for bottom hemisphere
    uint32_t bottom_hemisphere_start = cylinder_start + cylinder_vertices;
    for (uint32_t lat = 0; lat < cap_segments; ++lat) {
      uint32_t current_row =
          bottom_hemisphere_start + lat * (radial_segments + 1);
      uint32_t next_row = current_row + (radial_segments + 1);

      for (uint32_t lon = 0; lon < radial_segments; ++lon) {
        // First triangle
        indices.push_back(current_row + lon);
        indices.push_back(next_row + lon);
        indices.push_back(current_row + lon + 1);

        // Second triangle
        indices.push_back(current_row + lon + 1);
        indices.push_back(next_row + lon);
        indices.push_back(next_row + lon + 1);
      }
    }

    // Create and return buffers
    return _create_mesh_buffers(device, vertices, indices);
  }

private:
  mesh_instance(wgpu::Buffer &&vbuffer, wgpu::Buffer &&ibuffer,
                std::uint32_t index_count)
      : _vertex_buffer{std::move(vbuffer)}, _index_buffer{std::move(ibuffer)},
        _index_count{index_count} {}

  wgpu::Buffer _vertex_buffer;
  wgpu::Buffer _index_buffer;
  std::uint32_t _index_count;

  static mesh_instance
  _create_mesh_buffers(const wgpu::Device &device,
                       const std::vector<vertex> &vertices,
                       const std::vector<uint16_t> &indices) {
    // Create vertex buffer
    wgpu::BufferDescriptor vertexBufferDesc = {};
    vertexBufferDesc.usage =
        wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst;
    vertexBufferDesc.size = sizeof(vertex) * vertices.size();
    vertexBufferDesc.mappedAtCreation = true;

    wgpu::Buffer vertexBuffer = device.CreateBuffer(&vertexBufferDesc);
    std::memcpy(vertexBuffer.GetMappedRange(), vertices.data(),
                sizeof(vertex) * vertices.size());
    vertexBuffer.Unmap();

    // Create index buffer
    wgpu::BufferDescriptor indexBufferDesc = {};
    indexBufferDesc.usage =
        wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst;
    indexBufferDesc.size = sizeof(uint16_t) * indices.size();
    indexBufferDesc.mappedAtCreation = true;

    wgpu::Buffer indexBuffer = device.CreateBuffer(&indexBufferDesc);
    std::memcpy(indexBuffer.GetMappedRange(), indices.data(),
                sizeof(uint16_t) * indices.size());
    indexBuffer.Unmap();

    // Return the mesh instance
    return mesh_instance(std::move(vertexBuffer), std::move(indexBuffer),
                         indices.size());
  }
};

} // namespace scene3d
