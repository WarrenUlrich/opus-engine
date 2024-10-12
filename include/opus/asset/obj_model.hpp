#pragma once

#include <algorithm>
#include <array>
#include <fstream>
#include <optional>
#include <sstream>
#include <vector>

#include "../math/vector2.hpp"
#include "../math/vector3.hpp"

namespace asset {
class obj_model {
public:
  using vertex = math::vector3<float>;
  using texture_coord = math::vector2<float>;
  using normal = math::vector3<float>;

  struct face {
    std::vector<int> vertex_indices;
    std::vector<int> texture_indices;
    std::vector<int> normal_indices;
  };

  std::vector<vertex> vertices;
  std::vector<texture_coord> uvs;
  std::vector<normal> normals;
  std::vector<face> faces;

  std::vector<float> flatten() const noexcept {
    std::vector<float> data;
    // Estimate initial size: number of faces * triangles per face * vertices
    // per triangle * floats per vertex For positions (3) + texCoords (2) +
    // normals (3) = 8 floats per vertex
    size_t estimated_size =
        faces.size() * 3 * 8; // Assuming all faces are triangles
    data.reserve(estimated_size);

    for (const auto &face : faces) {
      // Ensure the face has at least 3 vertices to form a triangle
      if (face.vertex_indices.size() < 3) {
        continue; // Skip invalid or degenerate faces
      }

      // Triangulate the face
      for (std::size_t i = 1; i < face.vertex_indices.size() - 1; ++i) {
        std::array<int, 3> vertex_indices = {face.vertex_indices[0],
                                             face.vertex_indices[i],
                                             face.vertex_indices[i + 1]};

        std::array<int, 3> texture_indices = {
            (face.texture_indices.size() >= 1) ? face.texture_indices[0] : -1,
            (face.texture_indices.size() > i) ? face.texture_indices[i] : -1,
            (face.texture_indices.size() > i + 1) ? face.texture_indices[i + 1]
                                                  : -1};

        std::array<int, 3> normal_indices = {
            (face.normal_indices.size() >= 1) ? face.normal_indices[0] : -1,
            (face.normal_indices.size() > i) ? face.normal_indices[i] : -1,
            (face.normal_indices.size() > i + 1) ? face.normal_indices[i + 1]
                                                 : -1};

        for (int j = 0; j < 3; ++j) {
          int vert_idx = vertex_indices[j];
          // Bounds checking for vertex indices
          if (vert_idx < 0 ||
              static_cast<std::size_t>(vert_idx) >= vertices.size()) {
            // Handle invalid vertex index, skip or assign default value
            // Here, we'll skip the entire face if any vertex index is invalid
            data.clear(); // Optionally clear data or handle as needed
            return data;  // Early exit due to invalid index
          }

          const auto &vert = vertices[vert_idx];
          data.push_back(vert.x);
          data.push_back(vert.y);
          data.push_back(vert.z);

          int tex_idx = texture_indices[j];
          if (tex_idx != -1) {
            // Bounds checking for texture indices
            if (static_cast<std::size_t>(tex_idx) >= uvs.size()) {
              // Handle invalid texture index, assign default UV
              data.push_back(0.0f);
              data.push_back(0.0f);
            } else {
              const auto &uv = uvs[tex_idx];
              data.push_back(uv.x);
              data.push_back(uv.y);
            }
          } else {
            data.push_back(0.0f); // Default UV if missing
            data.push_back(0.0f);
          }

          int norm_idx = normal_indices[j];
          if (norm_idx != -1) {
            // Bounds checking for normal indices
            if (static_cast<std::size_t>(norm_idx) >= normals.size()) {
              // Handle invalid normal index, assign default normal
              data.push_back(0.0f);
              data.push_back(0.0f);
              data.push_back(0.0f);
            } else {
              const auto &norm = normals[norm_idx];
              data.push_back(norm.x);
              data.push_back(norm.y);
              data.push_back(norm.z);
            }
          } else {
            data.push_back(0.0f); // Default normal if missing
            data.push_back(0.0f);
            data.push_back(0.0f);
          }
        }
      }
    }

    return data;
  }

  static std::optional<obj_model> load_from(std::istream &stream) noexcept {
    obj_model result{};
    try {
      std::string line;
      while (std::getline(stream, line)) {
        // Trim leading whitespace
        auto first_non_space = line.find_first_not_of(" \t\r\n");
        if (first_non_space == std::string::npos)
          continue; // Skip empty lines
        if (line[first_non_space] == '#')
          continue; // Skip comments

        std::istringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "v") {
          vertex v;
          ss >> v.x >> v.y >> v.z;
          result.vertices.push_back(v);
        } else if (prefix == "vt") {
          texture_coord vt;
          ss >> vt.x >> vt.y;
          result.uvs.push_back(vt);
        } else if (prefix == "vn") { // Handle vertex normals
          normal vn;
          ss >> vn.x >> vn.y >> vn.z;
          result.normals.push_back(vn);
        } else if (prefix == "f") {
          face f;
          std::string vertexInfo;
          while (ss >> vertexInfo) {
            // Handle different face formats: v, v/vt, v//vn, v/vt/vn
            std::replace(vertexInfo.begin(), vertexInfo.end(), '/', ' ');
            std::istringstream vertexStream(vertexInfo);

            int vertexIndex = 0, textureIndex = 0, normalIndex = 0;
            vertexStream >> vertexIndex;

            if (vertexStream.peek() == ' ') {
              vertexStream.ignore();            // Ignore the space
              if (vertexStream.peek() != ' ') { // There's a texture index
                vertexStream >> textureIndex;
              }
              if (vertexStream.peek() == ' ') {
                vertexStream.ignore();            // Ignore the space
                if (vertexStream.peek() != EOF) { // There's a normal index
                  vertexStream >> normalIndex;
                }
              }
            }

            // Handle negative indices for vertices
            if (vertexIndex > 0) {
              f.vertex_indices.push_back(vertexIndex - 1);
            } else if (vertexIndex < 0) {
              f.vertex_indices.push_back(
                  static_cast<int>(result.vertices.size()) + vertexIndex);
            } else {
              f.vertex_indices.push_back(-1); // Invalid index
            }

            // Handle negative indices for texture coordinates
            if (textureIndex > 0) {
              f.texture_indices.push_back(textureIndex - 1);
            } else if (textureIndex < 0) {
              f.texture_indices.push_back(static_cast<int>(result.uvs.size()) +
                                          textureIndex);
            } else {
              f.texture_indices.push_back(-1); // No texture coordinate
            }

            // Handle negative indices for normals
            if (normalIndex > 0) {
              f.normal_indices.push_back(normalIndex - 1);
            } else if (normalIndex < 0) {
              f.normal_indices.push_back(
                  static_cast<int>(result.normals.size()) + normalIndex);
            } else {
              f.normal_indices.push_back(-1); // No normal
            }
          }

          // Ensure that the number of texture and normal indices matches vertex
          // indices Fill with -1 if necessary
          while (f.texture_indices.size() < f.vertex_indices.size()) {
            f.texture_indices.push_back(-1);
          }
          while (f.normal_indices.size() < f.vertex_indices.size()) {
            f.normal_indices.push_back(-1);
          }

          result.faces.push_back(f);
        }
        // Optionally handle other prefixes like "g", "o", "mtllib", "usemtl",
        // etc.
      }
      return result;
    } catch (...) {
      return std::nullopt;
    }
  }
};
} // namespace asset
