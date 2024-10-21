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
  // Simplified vertex struct with only pos, normal, and uv
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

  // Create a cube mesh with simplified attributes
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

private:
  mesh_instance(wgpu::Buffer &&vbuffer, wgpu::Buffer &&ibuffer,
                std::uint32_t index_count)
      : _vertex_buffer{std::move(vbuffer)}, _index_buffer{std::move(ibuffer)},
        _index_count{index_count} {}

  wgpu::Buffer _vertex_buffer;
  wgpu::Buffer _index_buffer;
  std::uint32_t _index_count;
};

} // namespace scene3d
