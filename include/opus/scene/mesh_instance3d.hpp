#pragma once

#include <memory>
#include "../gfx/core/mesh.hpp"

namespace scene {

class mesh_instance3d {
public:
  using mesh_type = gfx::mesh;

  std::shared_ptr<mesh_type> mesh;

  // Default constructor
  mesh_instance3d() = default;

  // Parameterized constructor
  mesh_instance3d(std::shared_ptr<mesh_type> mesh) : mesh(std::move(mesh)) {}

  // Copy constructor
  mesh_instance3d(const mesh_instance3d& other) = default;

  // Move constructor
  mesh_instance3d(mesh_instance3d&& other) noexcept = default;

  // Copy assignment operator
  mesh_instance3d& operator=(const mesh_instance3d& other) = default;

  // Move assignment operator
  mesh_instance3d& operator=(mesh_instance3d&& other) noexcept = default;

  // Destructor
  ~mesh_instance3d() = default;
};

} // namespace scene
