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
class deferred_renderer {
public:
  static std::optional<deferred_renderer> create(GLFWwindow *window) {
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
    if (!_create_surface(window, instance, adapter, device, surface,
                         texture_format))
      return std::nullopt;

    return deferred_renderer(window, std::move(instance), std::move(adapter),
                             std::move(device), std::move(queue),
                             std::move(surface), texture_format);
  }

  auto operator()(ecs::entity_id id, transform<float> &xform,
                  mesh_instance &mesh) {}

private:
  // Initialization variables
  GLFWwindow *_window;
  wgpu::Instance _instance;
  wgpu::Adapter _adapter;
  wgpu::Device _device;
  wgpu::Queue _queue;
  wgpu::Surface _surface;
  wgpu::TextureFormat _texture_format;

  // Gbuffer variables
  wgpu::Texture _gbuffer_albedo;
  wgpu::Texture _gbuffer_normals;
  wgpu::Texture _gbuffer_depth;
  std::array<wgpu::TextureView, 3> _gbuffer_views;
  wgpu::RenderPipeline _gbuffer_pipeline;

  deferred_renderer(GLFWwindow *window, wgpu::Instance &&instance,
                    wgpu::Adapter &&adapter, wgpu::Device &&device,
                    wgpu::Queue &&queue, wgpu::Surface &&surface,
                    wgpu::TextureFormat texture_format)
      : _window(window), _instance(std::move(instance)),
        _adapter(std::move(adapter)), _device(std::move(device)),
        _queue(std::move(queue)), _surface(std::move(surface)),
        _texture_format(std::move(texture_format)) {
    _create_gbuffer();
  }

  void _create_gbuffer() noexcept {
    int width, height;
    glfwGetWindowSize(_window, &width, &height);

    const auto albedo_descriptor = wgpu::TextureDescriptor{
        .usage = wgpu::TextureUsage::RenderAttachment |
                 wgpu::TextureUsage::TextureBinding,
        .size = wgpu::Extent3D{.width = static_cast<uint32_t>(width),
                               .height = static_cast<uint32_t>(height)},
        .format = wgpu::TextureFormat::BGRA8Unorm,
    };
    _gbuffer_albedo = _device.CreateTexture(&albedo_descriptor);

    const auto normals_descriptor = wgpu::TextureDescriptor{
        .usage = wgpu::TextureUsage::RenderAttachment |
                 wgpu::TextureUsage::TextureBinding,
        .size = wgpu::Extent3D{.width = static_cast<uint32_t>(width),
                               .height = static_cast<uint32_t>(height)},
        .format = wgpu::TextureFormat::RGBA16Float,
    };
    _gbuffer_normals = _device.CreateTexture(&normals_descriptor);

    const auto depth_descriptor = wgpu::TextureDescriptor{
        .usage = wgpu::TextureUsage::RenderAttachment |
                 wgpu::TextureUsage::TextureBinding,
        .size = wgpu::Extent3D{.width = static_cast<uint32_t>(width),
                               .height = static_cast<uint32_t>(height)},
        .format = wgpu::TextureFormat::Depth24Plus,
    };
    _gbuffer_depth = _device.CreateTexture(&depth_descriptor);

    _gbuffer_views[0] = _gbuffer_albedo.CreateView();
    _gbuffer_views[1] = _gbuffer_normals.CreateView();
    _gbuffer_views[2] = _gbuffer_depth.CreateView();

    const auto gbuffer_pipeline_layout = wgpu::PipelineLayoutDescriptor{};
    const auto gbuffer_descriptor = wgpu::RenderPipelineDescriptor{
      
    };

    _gbuffer_pipeline = _device.CreateRenderPipeline(&gbuffer_descriptor);
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
                              wgpu::Surface &surface,
                              wgpu::TextureFormat &format) noexcept {
    surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);
    if (surface == nullptr)
      return false;

    auto surface_caps = wgpu::SurfaceCapabilities();
    surface.GetCapabilities(adapter, &surface_caps);

    int width, height;
    glfwGetWindowSize(window, &width, &height);

    auto surface_config = wgpu::SurfaceConfiguration{
        .device = device,
        .format = surface_caps.formats[0],
        .width = static_cast<std::uint32_t>(width),
        .height = static_cast<std::uint32_t>(height),
    };

    format = surface_caps.formats[0];
    return surface != nullptr;
  }
};
} // namespace scene3d