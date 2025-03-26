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
  // Factory method to create an instance of the application
  static std::optional<forward_renderer> create(GLFWwindow *window) {
    dawnProcSetProcs(&dawn::native::GetProcs());

    forward_renderer app;
    app._window = window;

    glfwGetWindowSize(window, &app._width, &app._height);

    // Initialize WebGPU components
    if (!app._init_webgpu()) {
      return std::nullopt;
    }

    // Initialize GLFW and create a window
    if (!app._initialize_glfw()) {
      return std::nullopt;
    }

    // Create depth texture for rendering
    app._create_depth_texture();

    // Create uniform buffer
    app._create_buffers();

    // Create shader modules
    app._create_shader();

    // Set up bind group layout and bind group
    app._create_bind_group_layout();
    app._create_bind_group();

    // Set up pipeline layout and create the render pipeline
    app._create_pipeline_layout();
    app._create_pipeline();

    // Initialize some default materials
    std::vector<_material_uniform> materials(64);

    // Set up some example materials
    materials[0].albedo = math::vector3<float>{0.9f, 0.9f, 0.9f}; // White
    materials[0].metallic = 0.0f;
    materials[0].roughness = 0.5f;

    materials[1].albedo = math::vector3<float>{0.9f, 0.1f, 0.1f}; // Red
    materials[1].metallic = 0.0f;
    materials[1].roughness = 0.7f;

    materials[2].albedo = math::vector3<float>{0.1f, 0.9f, 0.1f}; // Green
    materials[2].metallic = 0.0f;
    materials[2].roughness = 1.0f;

    materials[3].albedo = math::vector3<float>{0.1f, 0.9f, 0.1f}; // Green
    materials[3].metallic = 1.0f;
    materials[3].roughness = 0.2f;

    // Update material buffer
    app._update_material_buffer(materials);

    return app;
  }

  void render(auto &ctx) {
    std::vector<_light_uniform> lights(256);

    // Clear all lights first (set to inactive)
    for (int i = 0; i < 256; ++i) {
      lights[i].type = _light_uniform::directional; // Default
      lights[i].intensity = 0.0f;                   // Turn off by default
    }

    // Create a main directional light (like sun)
    lights[0].type = _light_uniform::directional;
    lights[0].direction = math::vector3<float>{0.5f, -1.0f, 0.5f}.normalized();
    lights[0].color = math::vector3<float>{1.0f, 0.9f, 0.8f}; // Warm sunlight
    lights[0].intensity = 0.1f;

    // Blue spotlight
    lights[2].type = _light_uniform::spot;
    lights[2].position = math::vector3<float>{0.0f, 1.0f, 0.0f};
    lights[2].direction = math::vector3<float>{0.0f, -1.0f, 0.0f}.normalized();
    lights[2].color = math::vector3<float>{0.0f, 0.0f, 1.0f};
    lights[2].intensity = 200.0f;
    lights[2].radius = 15.0f;
    lights[2].cutoff_angle = 30.0f;

    // Update buffers
    update_light_buffer(lights);

    // Get the camera transform and camera
    ctx.template for_each_entity<scene3d::transform<float>,
                                 scene3d::camera<float>>(
        [&](const scene3d::transform<float> &xform,
            const scene3d::camera<float> &camera) {
          // Update the camera uniform buffer
          _camera_uniform camera_data;
          camera_data.view_proj =
              camera.get_projection_matrix() * xform.to_matrix().inverse();
          camera_data.position = xform.position;

          _queue.WriteBuffer(_camera_buffer, 0, &camera_data,
                             sizeof(_camera_uniform));
        });

    // Acquire the next texture from the surface
    wgpu::SurfaceTexture texture;
    _surface.GetCurrentTexture(&texture);

    if (!texture.texture) {
      std::cerr << "Failed to acquire next surface texture.\n";
      return;
    }

    // Create a texture view from the texture
    const auto texture_view = texture.texture.CreateView();

    // Create depth texture view
    const auto depth_texture_descriptor = wgpu::TextureViewDescriptor{};
    const auto depth_texture_view =
        _depth_texture.CreateView(&depth_texture_descriptor);

    // Create command encoder
    auto encoder_desc = wgpu::CommandEncoderDescriptor{};
    auto encoder = _device.CreateCommandEncoder(&encoder_desc);

    // Set up the render pass descriptor
    const auto color_attachment = wgpu::RenderPassColorAttachment{
        .view = texture_view,
        .resolveTarget = nullptr,
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {0.0f, 0.0f, 0.0f, 1.0f},
    };

    const auto depth_stencil_attachment =
        wgpu::RenderPassDepthStencilAttachment{
            .view = depth_texture_view,
            .depthLoadOp = wgpu::LoadOp::Clear,
            .depthStoreOp = wgpu::StoreOp::Store,
            .depthClearValue = 1.0f,
            .depthReadOnly = false,

            // Set stencil operations
            .stencilLoadOp = wgpu::LoadOp::Clear,
            .stencilStoreOp = wgpu::StoreOp::Store,
            .stencilClearValue = 0,
            .stencilReadOnly = false,
        };

    const auto render_pass_desc = wgpu::RenderPassDescriptor{
        .colorAttachmentCount = 1,
        .colorAttachments = &color_attachment,
        .depthStencilAttachment = &depth_stencil_attachment,
    };

    // Begin render pass
    auto render_pass = encoder.BeginRenderPass(&render_pass_desc);

    const size_t model_uniform_alignment = 256; // Alignment requirement
    size_t entity_index = 0;

    ctx.template for_each_entity<scene3d::transform<float>,
                                 scene3d::mesh_instance>(
        [&](const scene3d::transform<float> &xform,
            const scene3d::mesh_instance &mesh) {
          // Update model uniform buffer
          _model_uniform model_data;
          model_data.transform = xform.to_matrix();
          static uint32_t _matidx = 0;
          model_data.material_idx = ++_matidx % 3;

          // Calculate offset for this entity (must be 256-byte aligned)
          size_t model_data_offset = entity_index * model_uniform_alignment;

          // Write data to the buffer at the calculated offset
          _queue.WriteBuffer(_model_buffer, model_data_offset, &model_data,
                             sizeof(_model_uniform));

          // Set pipeline and bind group
          render_pass.SetPipeline(_pipeline);

          // Dynamic offset for this draw call
          uint32_t dynamic_offset = static_cast<uint32_t>(model_data_offset);

          render_pass.SetBindGroup(0, _bind_group_per_frame, 0, nullptr);
          render_pass.SetBindGroup(1, _bind_group_per_object, 1,
                                   &dynamic_offset);
          render_pass.SetBindGroup(2, _bind_group_shared, 0, nullptr);

          // Set vertex and index buffers
          render_pass.SetVertexBuffer(0, mesh.get_vertex_buffer());
          render_pass.SetIndexBuffer(mesh.get_index_buffer(),
                                     wgpu::IndexFormat::Uint16);

          // Draw call
          render_pass.DrawIndexed(mesh.get_index_count(), 1, 0, 0, 0);

          ++entity_index;
        });

    // End render pass
    render_pass.End();

    // Finish encoding commands
    const auto command_buffer = encoder.Finish();

    // Submit commands to the GPU queue
    _queue.Submit(1, &command_buffer);

    // Present the rendered image to the surface
    _surface.Present();
  }

  const auto &get_device() const noexcept { return _device; }

private:
  struct alignas(16) _camera_uniform {
    math::matrix4x4<float> view_proj;
    math::vector3<float> position;
  };

  struct alignas(16) _material_uniform {
    math::vector3<float> albedo;
    float metallic;
    float roughness;
  };

  struct alignas(16) _model_uniform {
    math::matrix4x4<float> transform;
    std::uint32_t material_idx;
  };

  struct alignas(16) _light_uniform {
    enum type { point = 0, directional = 1, spot = 2 };

    math::vector3<float> position;
    float intensity;
    math::vector3<float> color;
    float radius;
    math::vector3<float> direction;
    float cutoff_angle;
    type type;
  };

  struct alignas(16) _tile_uniform {
    uint32_t light_count;
    std::array<uint32_t, 256> light_indices;
  };

  forward_renderer() = default;

  // Initialize WebGPU instance, adapter, and device
  bool _init_webgpu() {
    // Set up toggles for Dawn
    std::vector<const char *> enableToggleNames;
    std::vector<const char *> disabledToggleNames;

    const auto toggles = wgpu::DawnTogglesDescriptor({
        .enabledToggleCount = enableToggleNames.size(),
        .enabledToggles = enableToggleNames.data(),
        .disabledToggleCount = disabledToggleNames.size(),
        .disabledToggles = disabledToggleNames.data(),
    });

    const auto instance_descriptor =
        wgpu::InstanceDescriptor{.nextInChain = nullptr,
                                 .features = wgpu::InstanceFeatures{
                                     .timedWaitAnyEnable = true,
                                 }};

    _instance = wgpu::CreateInstance(&instance_descriptor);

    // Request a GPU adapter
    const auto adapter_options = wgpu::RequestAdapterOptions{
        .powerPreference = wgpu::PowerPreference::HighPerformance,
        .backendType = wgpu::BackendType::Vulkan,
    };

    _instance.WaitAny(_instance.RequestAdapter(
                          &adapter_options, wgpu::CallbackMode::WaitAnyOnly,
                          [&](wgpu::RequestAdapterStatus status,
                              wgpu::Adapter adapter, const char *message) {
                            if (status != wgpu::RequestAdapterStatus::Success) {
                              std::cerr << "Failed to get adapter. " << message
                                        << '\n';
                              return;
                            }
                            _adapter = std::move(adapter);
                          }),
                      UINT64_MAX);

    if (_adapter == nullptr) {
      return false;
    }

    // Request a device from the adapter
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

    _instance.WaitAny(_adapter.RequestDevice(
                          &device_desc, wgpu::CallbackMode::WaitAnyOnly,
                          [&](wgpu::RequestDeviceStatus status,
                              wgpu::Device device, wgpu::StringView message) {
                            if (status != wgpu::RequestDeviceStatus::Success) {
                              std::cerr
                                  << "Failed to get a device:" << message.data
                                  << '\n';
                              return;
                            }

                            _device = std::move(device);
                            _queue = _device.GetQueue();
                          }),
                      UINT64_MAX);

    if (_device == nullptr) {
      return false;
    }

    return true;
  }

  // Initialize GLFW and create a window and surface
  bool _initialize_glfw() {
    // Create a surface for the window
    _surface = wgpu::glfw::CreateSurfaceForWindow(_instance, _window);

    // Configure the surface
    auto surface_caps = wgpu::SurfaceCapabilities();
    _surface.GetCapabilities(_adapter, &surface_caps);

    const auto surface_config =
        wgpu::SurfaceConfiguration{.device = _device,
                                   .format = surface_caps.formats[0],
                                   .width = static_cast<uint32_t>(_width),
                                   .height = static_cast<uint32_t>(_height)};

    _surface.Configure(&surface_config);

    _preferred_texture_format = surface_caps.formats[0];
    return true;
  }

  // Create a depth texture for depth testing
  void _create_depth_texture() {
    const auto depth_descriptor = wgpu::TextureDescriptor{
        .usage = wgpu::TextureUsage::RenderAttachment,
        .size = wgpu::Extent3D{.width = static_cast<uint32_t>(_width),
                               .height = static_cast<uint32_t>(_height)},
        .format = wgpu::TextureFormat::Depth24PlusStencil8,
    };

    _depth_texture = _device.CreateTexture(&depth_descriptor);
  }

  void _create_buffers() {
    // Uniform buffer for camera projection
    const auto camera_buffer_descriptor = wgpu::BufferDescriptor{
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = sizeof(_camera_uniform)};

    _camera_buffer = _device.CreateBuffer(&camera_buffer_descriptor);

    // Storage buffer for materials
    constexpr size_t max_materials = 64;
    const auto material_buffer_descriptor = wgpu::BufferDescriptor{
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        .size = sizeof(_material_uniform) * max_materials};
    _material_buffer = _device.CreateBuffer(&material_buffer_descriptor);

    // Uniform buffer for model transform
    constexpr size_t max_entities = 256;
    constexpr size_t model_uniform_alignment = 256;

    const auto model_buffer_descriptor = wgpu::BufferDescriptor{
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = max_entities * model_uniform_alignment,
    };

    _model_buffer = _device.CreateBuffer(&model_buffer_descriptor);

    constexpr auto max_lights = 256;

    // Storage buffer for lights
    const auto light_buffer_descriptor = wgpu::BufferDescriptor{
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        .size = sizeof(_light_uniform) * max_lights};
    _light_buffer = _device.CreateBuffer(&light_buffer_descriptor);

    // Storage buffer for tiles
    const auto tile_buffer_descriptor = wgpu::BufferDescriptor{
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        .size = sizeof(_tile_uniform) * (16 * 16)};
    _tile_buffer = _device.CreateBuffer(&tile_buffer_descriptor);
  }

  // Create the shader module from WGSL code
  void _create_shader() {
    auto shader_file = std::fstream("resources/shaders/forward_renderer.wgsl");
    auto shader_src = [&shader_file]() {
      auto sstr = std::ostringstream();
      sstr << shader_file.rdbuf();
      return sstr.str();
    }();

    const auto wgsl_desc = wgpu::ShaderModuleWGSLDescriptor({
        .code = shader_src.c_str(),
    });

    const auto module_descriptor = wgpu::ShaderModuleDescriptor{
        .nextInChain =
            reinterpret_cast<const wgpu::ChainedStruct *>(&wgsl_desc),
    };

    _shader_module = _device.CreateShaderModule(&module_descriptor);

    _instance.WaitAny(_shader_module.GetCompilationInfo(
                          wgpu::CallbackMode::AllowProcessEvents,
                          [](wgpu::CompilationInfoRequestStatus status,
                             const wgpu::CompilationInfo *info) {
                            switch (status) {
                            case wgpu::CompilationInfoRequestStatus::Success:
                              std::cout << "SUCCESS!\n";
                              break;
                            default:
                              throw std::exception();
                            }

                            for (auto i = 0; i < info->messageCount; ++i) {
                              std::cout << info->messages[i].message.data
                                        << '\n';
                            }
                          }),
                      UINT64_MAX);
  }

  void _create_bind_group_layout() {
    // Group 0: Per-frame data
    const wgpu::BindGroupLayoutEntry group0_entries[] = {
        // Camera entry
        {.binding = 0,
         .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
         .buffer = {.type = wgpu::BufferBindingType::Uniform}},
        // Light entry
        {.binding = 1,
         .visibility = wgpu::ShaderStage::Fragment,
         .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}}};

    wgpu::BindGroupLayoutDescriptor group0_descriptor = {
        .entryCount = std::size(group0_entries), .entries = group0_entries};
    _bind_group_layout_per_frame =
        _device.CreateBindGroupLayout(&group0_descriptor);

    // Group 1: Per-object data
    const wgpu::BindGroupLayoutEntry group1_entries[] = {
        // Model entry with dynamic offset
        {.binding = 0,
         .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
         .buffer = {
             .type = wgpu::BufferBindingType::Uniform,
             .hasDynamicOffset = true,
             .minBindingSize = sizeof(_model_uniform),
         }}};

    wgpu::BindGroupLayoutDescriptor group1_descriptor = {
        .entryCount = std::size(group1_entries), .entries = group1_entries};
    _bind_group_layout_per_object =
        _device.CreateBindGroupLayout(&group1_descriptor);

    // Group 2: Shared resources
    const wgpu::BindGroupLayoutEntry group2_entries[] = {
        // Material entry
        {.binding = 0,
         .visibility = wgpu::ShaderStage::Fragment,
         .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}}};

    wgpu::BindGroupLayoutDescriptor group2_descriptor = {
        .entryCount = std::size(group2_entries), .entries = group2_entries};
    _bind_group_layout_shared =
        _device.CreateBindGroupLayout(&group2_descriptor);
  }

  void _create_bind_group() {
    // Group 0: Per-frame bind group
    const wgpu::BindGroupEntry group0_entries[] = {
        // Camera entry
        {.binding = 0,
         .buffer = _camera_buffer,
         .size = sizeof(_camera_uniform)},
        // Light entry
        {.binding = 1,
         .buffer = _light_buffer,
         .size = sizeof(_light_uniform) * 256}};

    wgpu::BindGroupDescriptor group0_descriptor = {
        .layout = _bind_group_layout_per_frame,
        .entryCount = std::size(group0_entries),
        .entries = group0_entries};
    _bind_group_per_frame = _device.CreateBindGroup(&group0_descriptor);

    // Group 1: Per-object bind group
    const wgpu::BindGroupEntry group1_entries[] = {
        // Model entry
        {.binding = 0,
         .buffer = _model_buffer,
         .size = sizeof(_model_uniform)}};

    wgpu::BindGroupDescriptor group1_descriptor = {
        .layout = _bind_group_layout_per_object,
        .entryCount = std::size(group1_entries),
        .entries = group1_entries};
    _bind_group_per_object = _device.CreateBindGroup(&group1_descriptor);

    // Group 2: Shared resources bind group
    const wgpu::BindGroupEntry group2_entries[] = {
        // Material entry
        {.binding = 0,
         .buffer = _material_buffer,
         .size = sizeof(_material_uniform) * 64}};

    wgpu::BindGroupDescriptor group2_descriptor = {
        .layout = _bind_group_layout_shared,
        .entryCount = std::size(group2_entries),
        .entries = group2_entries};
    _bind_group_shared = _device.CreateBindGroup(&group2_descriptor);
  }

  void _create_pipeline_layout() {
    wgpu::BindGroupLayout layouts[3] = {_bind_group_layout_per_frame,
                                        _bind_group_layout_per_object,
                                        _bind_group_layout_shared};

    wgpu::PipelineLayoutDescriptor layout_descriptor = {
        .bindGroupLayoutCount = 3, .bindGroupLayouts = layouts};
    _pipeline_layout = _device.CreatePipelineLayout(&layout_descriptor);
  }

  // Create the render pipeline
  void _create_pipeline() {
    // Define the vertex attributes
    wgpu::VertexAttribute vertex_attributes[3]{
        // Position (vec3<f32>)
        {
            .format = wgpu::VertexFormat::Float32x3,
            .offset = 0,
            .shaderLocation = 0,
        },
        // Normal (vec3<f32>)
        {
            .format = wgpu::VertexFormat::Float32x3,
            .offset = sizeof(float) * 3,
            .shaderLocation = 1,
        },
        // UV (vec2<f32>)
        {
            .format = wgpu::VertexFormat::Float32x2,
            .offset = sizeof(float) * 6,
            .shaderLocation = 2,
        }};

    // Describe the vertex buffer layout
    const auto vertex_buffer_layout = wgpu::VertexBufferLayout{
        .arrayStride = sizeof(float) * 8,
        .stepMode = wgpu::VertexStepMode::Vertex,
        .attributeCount = std::size(vertex_attributes),
        .attributes = vertex_attributes,
    };

    // Set up the vertex state
    const auto vertex_state = wgpu::VertexState{
        .module = _shader_module,
        .entryPoint = "vs",
        .bufferCount = 1,
        .buffers = &vertex_buffer_layout,
    };

    // Define the fragment output target
    const auto color_target = wgpu::ColorTargetState{
        .format = _preferred_texture_format,
        .blend = nullptr,
        .writeMask = wgpu::ColorWriteMask::All,
    };

    // Set up the fragment state
    const auto fragment_state = wgpu::FragmentState{
        .module = _shader_module,
        .entryPoint = "fs",
        .targetCount = 1,
        .targets = &color_target,
    };

    // Configure depth and stencil state
    const auto depth_stencil_state = wgpu::DepthStencilState{
        .format = wgpu::TextureFormat::Depth24PlusStencil8,
        .depthWriteEnabled = true,
        .depthCompare = wgpu::CompareFunction::Less,
        .stencilReadMask = 0xff,
        .stencilWriteMask = 0xff,
    };

    // Set up the render pipeline descriptor
    const auto pipeline_descriptor = wgpu::RenderPipelineDescriptor{
        .nextInChain = nullptr,
        .label = wgpu::StringView("Forward Rendering Pipeline"),
        .layout = _pipeline_layout,
        .vertex = vertex_state,
        .primitive =
            {
                .topology = wgpu::PrimitiveTopology::TriangleList,
                .stripIndexFormat = wgpu::IndexFormat::Undefined,
                .frontFace = wgpu::FrontFace::CCW,
                .cullMode = wgpu::CullMode::Back,
                .unclippedDepth = false,
            },
        .depthStencil = &depth_stencil_state,
        .multisample =
            {
                .count = 1,
                .mask = ~0u,
                .alphaToCoverageEnabled = false,
            },
        .fragment = &fragment_state,
    };

    // Create the render pipeline
    _pipeline = _device.CreateRenderPipeline(&pipeline_descriptor);
  }

  void update_light_buffer(const std::vector<_light_uniform> &lights) {
    // Check if the light buffer has enough capacity
    size_t size = sizeof(_light_uniform) * lights.size();

    // Make sure the data fits into the buffer size (here, 256 lights)
    assert(lights.size() <= 256);

    // Write the light data to the GPU buffer
    _queue.WriteBuffer(_light_buffer, // Destination buffer
                       0,             // Offset in bytes (start of the buffer)
                       lights.data(), // Pointer to the data array
                       size);         // Size of the data in bytes
  }

  void
  _update_material_buffer(const std::vector<_material_uniform> &materials) {
    // Check if the material buffer has enough capacity
    size_t size = sizeof(_material_uniform) * materials.size();

    // Make sure the data fits into the buffer size (here, 64 materials)
    assert(materials.size() <= 64);

    // Write the material data to the GPU buffer
    _queue.WriteBuffer(_material_buffer, // Destination buffer
                       0, // Offset in bytes (start of the buffer)
                       materials.data(), // Pointer to the data array
                       size);            // Size of the data in bytes
  }

  // Convert degrees to radians
  constexpr static float radians(float degrees) {
    return degrees * (std::numbers::pi_v<float> / 180.0f);
  }

  // WebGPU instance, adapter, device, and queue
  wgpu::Instance _instance;
  wgpu::Adapter _adapter;
  wgpu::Device _device;
  wgpu::Queue _queue;

  // Surface and preferred texture format
  wgpu::Surface _surface;
  wgpu::TextureFormat _preferred_texture_format;

  // GPU resources
  wgpu::Buffer _camera_buffer;  // Uniform buffer for matrices
  wgpu::Buffer _model_buffer;   // Uniform buffer for matrices
  wgpu::Buffer _light_buffer;   // Storage buffer for lights
  wgpu::Buffer _tile_buffer;    // Storage buffer for tiles
  wgpu::Texture _depth_texture; // Depth texture
  // wgpu::BindGroup _bind_group;              // Bind group for uniform buffer
  // wgpu::BindGroupLayout _bind_group_layout; // Bind group layout

  wgpu::BindGroupLayout _bind_group_layout_per_frame;
  wgpu::BindGroupLayout _bind_group_layout_per_object;
  wgpu::BindGroupLayout _bind_group_layout_shared;
  wgpu::BindGroup _bind_group_per_frame;
  wgpu::BindGroup _bind_group_per_object;
  wgpu::BindGroup _bind_group_shared;

  wgpu::ShaderModule _shader_module;     // Shader module
  wgpu::PipelineLayout _pipeline_layout; // Pipeline layout
  wgpu::RenderPipeline _pipeline;        // Render pipeline

  wgpu::Buffer _material_buffer;

  // Application data
  int _width, _height; // Window dimensions
  GLFWwindow *_window; // GLFW window handle
};
} // namespace scene3d