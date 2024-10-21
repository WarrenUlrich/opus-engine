#pragma once

#include <optional>

#include <webgpu/webgpu_cpp.h>
#include <webgpu/webgpu_glfw.h>

#include <dawn/native/DawnNative.h>

#include <GLFW/glfw3.h>

#include "../ecs/ecs.hpp"

#include "camera.hpp"
#include "mesh_instance.hpp"
#include "transform.hpp"

namespace scene3d {
class forward_renderer {
public:
  static std::optional<forward_renderer> create(GLFWwindow *window) {
    auto instance = wgpu::Instance();
    if (!_create_instance(instance))
      return std::nullopt;

    auto adapter = wgpu::Adapter();
    if (!_create_adapter(instance, adapter))
      return std::nullopt;

    auto device = wgpu::Device();
    auto queue = wgpu::Queue();
    if (!_create_device(instance, adapter, device, queue))
      return std::nullopt;

    auto surface = wgpu::Surface();
    auto texture_format = wgpu::TextureFormat::RGBA8Unorm;
    if (!_create_surface(window, instance, adapter, device, surface))
      return std::nullopt;

    return forward_renderer(window, std::move(instance), std::move(adapter),
                            std::move(device), std::move(queue),
                            std::move(surface));
  }

private:
  struct _camera_uniform {
    math::matrix4x4<float> mvp;
  };

  struct _light_uniform {
    math::vector3<float> position;
    float intensity;
    math::vector3<float> color;
    float radius;
  };

  struct _tile_uniform {
    uint32_t light_count;
    std::array<uint32_t, 256> light_indices;
  };

  // Initialization variables
  GLFWwindow *_window;
  wgpu::Instance _instance;
  wgpu::Adapter _adapter;
  wgpu::Device _device;
  wgpu::Queue _queue;
  wgpu::Surface _surface;

  int _width;
  int _height;

  wgpu::Texture _depth_texture;

  wgpu::Buffer _camera_buffer;
  wgpu::Buffer _light_buffer;
  wgpu::Buffer _tile_buffer;

  wgpu::BindGroup _bind_group;

  wgpu::ShaderModule _vertex_shader;
  wgpu::ShaderModule _fragment_shader;

  forward_renderer(GLFWwindow *window, wgpu::Instance &&instance,
                   wgpu::Adapter &&adapter, wgpu::Device &&device,
                   wgpu::Queue &&queue, wgpu::Surface &&surface)
      : _window(window), _instance(std::move(instance)),
        _adapter(std::move(adapter)), _device(std::move(device)),
        _queue(std::move(queue)), _surface(std::move(surface)) {
    glfwGetWindowSize(_window, &_width, &_height);

    _configure_surface();

    _configure_depth_texture();

    _configure_buffers();

    _configure_bind_group();

    _configure_shaders();

    _configure_pipeline();
  }

  wgpu::Buffer _create_buffer(const void *data, size_t size,
                              wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor descriptor{};
    descriptor.size = size;
    descriptor.usage = usage | wgpu::BufferUsage::CopyDst;

    auto buffer = _device.CreateBuffer(&descriptor);
    if (data) {
      _queue.WriteBuffer(buffer, 0, data, size);
    }
    return buffer;
  }

  void _configure_surface() noexcept {
    auto surface_caps = wgpu::SurfaceCapabilities();
    _surface.GetCapabilities(_adapter, &surface_caps);

    const auto surface_config = wgpu::SurfaceConfiguration{
        .device = _device,
        .format = surface_caps.formats[0],
        .width = static_cast<std::uint32_t>(_width),
        .height = static_cast<std::uint32_t>(_height),
        .presentMode = wgpu::PresentMode::Fifo,
    };

    _surface.Configure(&surface_config);
  }

  void _configure_depth_texture() noexcept {
    const auto depth_descriptor = wgpu::TextureDescriptor{
        .usage = wgpu::TextureUsage::RenderAttachment |
                 wgpu::TextureUsage::TextureBinding,
        .size = wgpu::Extent3D{.width = static_cast<uint32_t>(_width),
                               .height = static_cast<uint32_t>(_height)},
        .format = wgpu::TextureFormat::Depth24Plus,
    };

    _depth_texture = _device.CreateTexture(&depth_descriptor);
  }

  void _configure_buffers() noexcept {
    // Camera buffer
    _camera_uniform camera_data = {}; // Fill with camera data
    _camera_buffer = _create_buffer(&camera_data, sizeof(_camera_uniform),
                                    wgpu::BufferUsage::Uniform);

    std::vector<_light_uniform> lights(256);
    _light_buffer =
        _create_buffer(lights.data(), sizeof(_light_uniform) * lights.size(),
                       wgpu::BufferUsage::Storage);

    std::vector<_tile_uniform> tiles(16 * 16);
    _tile_buffer =
        _create_buffer(tiles.data(), sizeof(_tile_uniform) * tiles.size(),
                       wgpu::BufferUsage::Storage);
  }

  void _configure_bind_group() noexcept {
    const auto bgl_entries = {
        // Camera entry
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility =
                wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
            .buffer = {.type = wgpu::BufferBindingType::Uniform},
        },
        // Light entry
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Fragment,
            .buffer = {.type = wgpu::BufferBindingType::Storage},
        },
        // Tile entry
        wgpu::BindGroupLayoutEntry{
            .binding = 2,
            .visibility = wgpu::ShaderStage::Fragment,
            .buffer = {.type = wgpu::BufferBindingType::Storage},
        }};

    const auto bgl_descriptor = wgpu::BindGroupLayoutDescriptor{
        .entryCount = bgl_entries.size(), .entries = bgl_entries.begin()};

    const auto bg_entries = {
        // Camera entry
        wgpu::BindGroupEntry{.binding = 0,
                             .buffer = _camera_buffer,
                             .size = sizeof(_camera_uniform)},
        // Light entry
        wgpu::BindGroupEntry{.binding = 1,
                             .buffer = _light_buffer,
                             .size = sizeof(_light_uniform) * 256},
        // Tile entry
        wgpu::BindGroupEntry{.binding = 2,
                             .buffer = _tile_buffer,
                             .size = sizeof(_tile_uniform) * 16 * 16}};

    const auto bg_descriptor = wgpu::BindGroupDescriptor{
        .layout = _device.CreateBindGroupLayout(&bgl_descriptor),
        .entryCount = 3,
        .entries = bg_entries.begin()};

    _bind_group = _device.CreateBindGroup(&bg_descriptor);
  }

  void _configure_shaders() noexcept {
    auto vshader_desc = wgpu::ShaderModuleWGSLDescriptor{};
    vshader_desc.code = R"(
      struct Camera {
          mvp: mat4x4<f32>,
      };

      @group(0) @binding(0) var<uniform> camera: Camera;

      struct VertexInput {
          @location(0) position: vec3<f32>,
          @location(1) normal: vec3<f32>,
          @location(2) uv: vec2<f32>,
      };

      struct VertexOutput {
          @builtin(position) pos: vec4<f32>,
          @location(0) frag_pos: vec3<f32>,
          @location(1) normal: vec3<f32>,
      };

      @vertex
      fn main(input: VertexInput) -> VertexOutput {
          var output: VertexOutput;
          output.pos = camera.view_proj * vec4<f32>(input.position, 1.0);
          output.frag_pos = input.position;
          output.normal = input.normal;
          return output;
      }
    )";

    auto vshader_desc_ = wgpu::ShaderModuleDescriptor{};
    vshader_desc_.nextInChain =
        reinterpret_cast<const wgpu::ChainedStruct *>(&vshader_desc);

    _vertex_shader = _device.CreateShaderModule(&vshader_desc_);

    auto fshader_desc = wgpu::ShaderModuleWGSLDescriptor{};
    fshader_desc.code = R"(
      struct Light {
        position: vec3<f32>,
        intensity: f32,
        color: vec3<f32>,
        radius: f32,
      };

      struct TileData {
        light_count: u32,
        light_indices: array<u32>,
      };

      @group(0) @binding(1) var<storage, read> lights: array<Light>;
      @group(0) @binding(2) var<storage, read> tiles: array<TileData>;

      @fragment
      fn main(@location(0) frag_pos: vec3<f32>, @location(1) normal: vec3<f32>) -> @location(0) vec4<f32> {
          let tile_index = calculate_tile_index(frag_pos); // Tile index calculation function
          let tile = tiles[tile_index];
          
          var color: vec3<f32> = vec3<f32>(0.0);

          for (var i = 0u; i < tile.light_count; i = i + 1u) {
              let light = lights[tile.light_indices[i]];
              let light_dir = normalize(light.position - frag_pos);
              let diff = max(dot(normal, light_dir), 0.0);
              color += diff * light.intensity * light.color;
          }

          return vec4<f32>(color, 1.0);
      }
)";

    auto fshader_desc_ = wgpu::ShaderModuleDescriptor{};
    fshader_desc_.nextInChain =
        reinterpret_cast<const wgpu::ChainedStruct *>(&fshader_desc);

    _fragment_shader = _device.CreateShaderModule(&fshader_desc_);
  }

  void _configure_pipeline() noexcept {
    
  }

  static bool _create_instance(wgpu::Instance &instance) noexcept {
    dawnProcSetProcs(&dawn::native::GetProcs());

    std::vector<const char *> enableToggleNames;
    std::vector<const char *> disabledToggleNames;

    const auto toggles = wgpu::DawnTogglesDescriptor({
        .enabledToggleCount = enableToggleNames.size(),
        .enabledToggles = enableToggleNames.data(),
        .disabledToggleCount = disabledToggleNames.size(),
        .disabledToggles = disabledToggleNames.data(),
    });

    const auto instance_descriptor =
        wgpu::InstanceDescriptor{.nextInChain = &toggles,
                                 .features = wgpu::InstanceFeatures{
                                     .timedWaitAnyEnable = true,
                                 }};

    instance = wgpu::CreateInstance(&instance_descriptor);
    return instance != nullptr;
  }

  static bool _create_adapter(const wgpu::Instance &instance,
                              wgpu::Adapter &adapter) noexcept {
    const auto adapter_options = wgpu::RequestAdapterOptions{
        .powerPreference = wgpu::PowerPreference::HighPerformance,
        .backendType = wgpu::BackendType::Vulkan,
    };

    instance.WaitAny(instance.RequestAdapter(
                         &adapter_options, wgpu::CallbackMode::WaitAnyOnly,
                         [&](wgpu::RequestAdapterStatus status,
                             wgpu::Adapter adapter, const char *message) {
                           if (status != wgpu::RequestAdapterStatus::Success) {
                             std::cerr << "Failed to get adapter. " << message
                                       << '\n';
                             return;
                           }
                           adapter = std::move(adapter);
                         }),
                     UINT64_MAX);
    return adapter != nullptr;
  }

  static bool _create_device(const wgpu::Instance &instance,
                             const wgpu::Adapter &adapter, wgpu::Device &device,
                             wgpu::Queue &queue) noexcept {
    auto device_desc = wgpu::DeviceDescriptor();
    device_desc.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [](const wgpu::Device &, wgpu::DeviceLostReason reason,
           wgpu::StringView message) {
          const char *reasonName = "";
          switch (reason) {
          case wgpu::DeviceLostReason::Unknown:
            reasonName = "Unknown";
            break;
          case wgpu::DeviceLostReason::Destroyed:
            reasonName = "Destroyed";
            break;
          case wgpu::DeviceLostReason::InstanceDropped:
            reasonName = "InstanceDropped";
            break;
          case wgpu::DeviceLostReason::FailedCreation:
            reasonName = "FailedCreation";
            break;
          }
          std::cerr << "Device lost because of " << reasonName << ": "
                    << message.data;
        });

    device_desc.SetUncapturedErrorCallback([](const wgpu::Device &,
                                              wgpu::ErrorType type,
                                              wgpu::StringView message) {
      const char *errorTypeName = "";
      switch (type) {
      case wgpu::ErrorType::Validation:
        errorTypeName = "Validation";
        break;
      case wgpu::ErrorType::OutOfMemory:
        errorTypeName = "Out of memory";
        break;
      case wgpu::ErrorType::Unknown:
        errorTypeName = "Unknown";
        break;
      case wgpu::ErrorType::DeviceLost:
        errorTypeName = "Device lost";
        break;
      }
      std::cerr << errorTypeName << " error: " << message.data << '\n';
    });

    instance.WaitAny(adapter.RequestDevice(
                         &device_desc, wgpu::CallbackMode::WaitAnyOnly,
                         [&](wgpu::RequestDeviceStatus status,
                             wgpu::Device device, wgpu::StringView message) {
                           if (status != wgpu::RequestDeviceStatus::Success) {
                             std::cerr
                                 << "Failed to get a device:" << message.data
                                 << '\n';
                             return;
                           }

                           device = std::move(device);
                           queue = device.GetQueue();
                         }),
                     UINT64_MAX);

    return device != nullptr;
  }

  static bool _create_surface(GLFWwindow *window,
                              const wgpu::Instance &instance,
                              const wgpu::Adapter &adapter,
                              const wgpu::Device &device,
                              wgpu::Surface &surface) noexcept {
    surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);
    if (surface == nullptr)
      return false;

    return surface != nullptr;
  }
};
} // namespace scene3d