#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numbers>
#include <optional>
#include <thread>
#include <vector>

#include <webgpu/webgpu_cpp.h>
#include <webgpu/webgpu_glfw.h>

#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>

#include <GLFW/glfw3.h>

#include <opus/math/math.hpp>

#include <opus/scene3d/scene3d.hpp>

#include <opus/ecs/ecs.hpp>

using ecs_context =
    ecs::context<scene3d::transform<float>, scene3d::camera<float>,
                 scene3d::mesh_instance>;

int main(int argc, char **args) {
  glfwSetErrorCallback([](int code, const char *message) {
    std::cerr << "GLFW error: " << code << " - " << message << '\n';
  });

  if (!glfwInit()) {
    return false;
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  constexpr auto width = 1920 / 2, height = 1080;
  const auto window =
      glfwCreateWindow(width, height, "WebGPU 3D Sphere", nullptr, nullptr);

  auto renderer = scene3d::forward_renderer::create(window);
  if (!renderer) {
    std::cerr << "Failed to create application" << std::endl;
    return 1;
  }

  ecs_context ctx;

  auto camera_entity = ctx.new_entity();
  ctx.add_component(camera_entity,
                    scene3d::transform<float>{
                        .position = {0.0f, 0.0f, -15.0f},
                        .rotation = math::quaternion<float>::look_at(
                            {0.0f, 0.0f, -15.0f}, math::vector3<float>::zero)});

  ctx.add_component(camera_entity, [&]() {
    auto camera = scene3d::camera<float>();
    camera.set_fov(60.0f * (std::numbers::pi_v<float> / 180.0f));
    camera.set_aspect_ratio(static_cast<float>(width) / height);
    camera.set_far_plane(10000.0f);

    return camera;
  }());

  auto sphere1 = ctx.new_entity();
  ctx.add_component(sphere1, scene3d::transform<float>{.position = {0, -2, 0}});
  ctx.add_component(sphere1, scene3d::mesh_instance::create_torus(
                                 renderer->get_device(), 1.0f, 0.5f));

  auto sphere2 = ctx.new_entity();
  ctx.add_component(sphere2, scene3d::transform<float>{.position = {0, 0, 0}});
  ctx.add_component(sphere2, scene3d::mesh_instance::create_capsule(
                                 renderer->get_device(), 1.0f, 2.0f));

  // And add this plane creation code:
  auto plane = ctx.new_entity();
  ctx.add_component(plane,
                    scene3d::transform<float>{
                        .position = {0.0f, -5.0f, 0.0f},
                        .rotation = math::quaternion<float>::from_axis_angle(
                            math::vector3<float>(0.0f, 1.0f, 0.0f),
                            std::numbers::pi_v<float> / 2)});

  // Create a large flat plane - we'll assume there's a create_plane method
  // If not available, you'd need to create a custom plane mesh
  ctx.add_component(plane, scene3d::mesh_instance::create_plane(
                               renderer->get_device(), 20.0f, 20.0f));

  const float camera_speed = 20.0f;  // Units per second
  const float rotation_speed = 1.5f; // Radians per second

  double last_frame_time = glfwGetTime();

  float camera_yaw = 0.0f;   // Rotation around Y axis (left/right)
  float camera_pitch = 0.0f; // Rotation around X axis (up/down)

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Calculate delta time for frame-rate independent movement
    double current_time = glfwGetTime();
    float delta_time = static_cast<float>(current_time - last_frame_time);
    last_frame_time = current_time;

    // Handle camera movement with WASD keys
    auto camera_transform =
        ctx.try_get_component<scene3d::transform<float>>(camera_entity);
    auto camera = ctx.try_get_component<scene3d::camera<float>>(camera_entity);

    // Get camera orientation vectors
    auto forward = math::vector3<float>(0.0f, 1.0f, 0.0f);
    auto right = math::vector3<float>(1.0f, 0.0f, 0.0f);

    // Apply rotation if your camera has orientation
    if (camera_transform->rotation != math::quaternion<float>()) {
      forward = camera_transform->rotation.rotate(forward);
      right = camera_transform->rotation.rotate(right);
    }

    // Check for key presses and move camera
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
      camera_transform->position += forward * camera_speed * delta_time;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
      camera_transform->position -= forward * camera_speed * delta_time;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
      camera_transform->position -= right * camera_speed * delta_time;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
      camera_transform->position += right * camera_speed * delta_time;
    }

    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
      camera_pitch += rotation_speed * delta_time;
      // Clamp pitch to avoid gimbal lock (-89 to 89 degrees in radians)
      camera_pitch = std::min(std::max(camera_pitch, -1.55f), 1.55f);
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
      camera_pitch -= rotation_speed * delta_time;
      camera_pitch = std::min(std::max(camera_pitch, -1.55f), 1.55f);
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
      camera_yaw -= rotation_speed * delta_time;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
      camera_yaw += rotation_speed * delta_time;
    }

    auto yaw_rotation = math::quaternion<float>::from_axis_angle(
        math::vector3<float>(0.0f, 1.0f, 0.0f), camera_yaw);
    auto pitch_rotation = math::quaternion<float>::from_axis_angle(
        math::vector3<float>(1.0f, 0.0f, 0.0f), camera_pitch);

    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
      camera_transform->rotation = math::quaternion<float>::look_at(
          camera_transform->position, math::vector3<float>(0, 0, 0));
    }

    float angle = static_cast<float>(current_time);

    renderer->render(ctx);
  }

  return 0;
}