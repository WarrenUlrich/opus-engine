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

#include <opus/scene3d/camera.hpp>
#include <opus/scene3d/deferred_renderer.hpp>
#include <opus/scene3d/forward_renderer.hpp>
#include <opus/scene3d/mesh_instance.hpp>
#include <opus/scene3d/transform.hpp>
#include <opus/scene3d/light.hpp>

#include <opus/ecs/ecs.hpp>

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

    return app;
  }

  void render(auto &ctx) {
    std::vector<_light_uniform> lights(256);

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Populate light data with time-based randomness
    float timeFactor = static_cast<float>(std::clock()) / CLOCKS_PER_SEC;

    for (int i = 0; i < 256; ++i) {
      // Generate a random offset for the light position
      float randomOffsetX = static_cast<float>(std::rand()) / RAND_MAX;
      float randomOffsetY = static_cast<float>(std::rand()) / RAND_MAX;
      float randomOffsetZ = static_cast<float>(std::rand()) / RAND_MAX;

      // Assign a random light type for variation
      if (i % 3 == 0) {
        lights[i].type = _light_uniform::point;
        lights[i].color = math::vector3<float>{0.0f, 1.0f, 0.0f};
      } else if (i % 3 == 1) {
        // lights[i].type = _light_uniform::directional;
        // lights[i].direction = math::vector3<float>{
        //     0.0f, -1.0f, 0.0f}; // Default downward direction
        // lights[i].color = math::vector3<float>{0.0f, 0.0f, 1.0f};
      } else {
        lights[i].type = _light_uniform::spot;
        lights[i].direction =
            math::vector3<float>{0.0f, -1.0f, 0.0f}; // Spotlight direction
        lights[i].cutoff_angle = 40.0f; // Spotlights' cutoff angle in degrees
        lights[i].color = math::vector3<float>{1.0f, 0.0f, 0.0f};
      }

      // Modulate position, intensity, and color with time for randomness
      lights[i].position = math::vector3<float>{
          i * 0.1f + randomOffsetX * std::sin(timeFactor + i),
          i * 0.1f + randomOffsetY * std::cos(timeFactor + i),
          i * 0.1f + randomOffsetZ * std::sin(timeFactor * 0.5f + i)};

      // Randomly vary intensity between 0.5 and 1.5 over time
      lights[i].intensity = 1.5f + std::sin(timeFactor + i) * 0.5f;

      // Keep the radius constant for point lights or spotlights
      lights[i].radius = (lights[i].type == _light_uniform::point ||
                          lights[i].type == _light_uniform::spot)
                             ? 10.0f
                             : 0.0f;
    }

    // Update buffers
    update_light_buffer(lights);

    // Get the camera transform and camera
    ctx.template for_each_entity<scene3d::transform<float>,
                                 scene3d::camera<float>>(
        [&](const scene3d::transform<float> &xform,
            const scene3d::camera<float> &camera) {
          // Update the camera uniform buffer
          _camera_uniform camera_data;
          camera_data.mvp = camera.get_projection_matrix() * xform.to_matrix();
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

          // Calculate offset for this entity (must be 256-byte aligned)
          size_t model_data_offset = entity_index * model_uniform_alignment;

          // Write data to the buffer at the calculated offset
          _queue.WriteBuffer(_model_buffer, model_data_offset, &model_data,
                             sizeof(_model_uniform));

          // Set pipeline and bind group
          render_pass.SetPipeline(_pipeline);

          // Dynamic offset for this draw call
          uint32_t dynamic_offset = static_cast<uint32_t>(model_data_offset);

          // Set bind group with dynamic offset
          render_pass.SetBindGroup(0, _bind_group, 1, &dynamic_offset);

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
    math::matrix4x4<float> mvp;
    math::vector3<float> position;
  };

  struct alignas(16) _model_uniform {
    math::matrix4x4<float> transform;
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

    constexpr size_t max_entities = 256;
    constexpr size_t model_uniform_alignment = 256;

    // Uniform buffer for model transform
    const auto model_buffer_descriptor = wgpu::BufferDescriptor{
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = max_entities * model_uniform_alignment,
    };

    _model_buffer = _device.CreateBuffer(&model_buffer_descriptor);

    // Storage buffer for lights
    const auto light_buffer_descriptor = wgpu::BufferDescriptor{
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        .size = sizeof(_light_uniform) * 256};
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

  // Create the bind group layout for uniform and storage buffers
  void _create_bind_group_layout() {
    const wgpu::BindGroupLayoutEntry bgl_entries[] = {
        // Camera entry
        {.binding = 0,
         .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
         .buffer = {.type = wgpu::BufferBindingType::Uniform}},
        // Model entry
        {.binding = 1,
         .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
         .buffer =
             {
                 .type = wgpu::BufferBindingType::Uniform,
                 .hasDynamicOffset = true,
                 .minBindingSize = sizeof(_model_uniform),
             }},
        // Light entry
        {.binding = 2,
         .visibility = wgpu::ShaderStage::Fragment,
         .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}},
        // Tile entry
        {.binding = 3,
         .visibility = wgpu::ShaderStage::Fragment,
         .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}},
    };

    wgpu::BindGroupLayoutDescriptor bgl_descriptor = {
        .entryCount = std::size(bgl_entries), .entries = bgl_entries};

    _bind_group_layout = _device.CreateBindGroupLayout(&bgl_descriptor);
  }

  // Create the bind group for uniform and storage buffers
  void _create_bind_group() {
    const wgpu::BindGroupEntry bg_entries[] = {
        // Camera entry
        {.binding = 0,
         .buffer = _camera_buffer,
         .size = sizeof(_camera_uniform)},
        // Model entry
        {.binding = 1, .buffer = _model_buffer, .size = sizeof(_model_uniform)},
        // Light entry
        {.binding = 2,
         .buffer = _light_buffer,
         .size = sizeof(_light_uniform) * 256},
        // Tile entry
        {.binding = 3,
         .buffer = _tile_buffer,
         .size = sizeof(_tile_uniform) * 16 * 16},
    };

    wgpu::BindGroupDescriptor bg_descriptor = {.layout = _bind_group_layout,
                                               .entryCount =
                                                   std::size(bg_entries),
                                               .entries = bg_entries};

    _bind_group = _device.CreateBindGroup(&bg_descriptor);
  }

  // Create the pipeline layout
  void _create_pipeline_layout() {
    wgpu::PipelineLayoutDescriptor layout_descriptor = {
        .bindGroupLayoutCount = 1, .bindGroupLayouts = &_bind_group_layout};
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

  // Update the uniform buffer with the latest MVP matrix
  // Inside the 'application' class
  void update_uniform_buffer(const scene3d::camera<float> &camera,
                             const math::matrix4x4<float> &view_matrix,
                             scene3d::transform<float> &obj_transform) {
    // Step 1: Calculate the current rotation angle based on elapsed time
    double currentTime = glfwGetTime(); // Get the elapsed time in seconds
    float angle = static_cast<float>(currentTime);

    // Optional: Update object rotation if needed
    // obj_transform.rotation = math::quaternion<float>::from_axis_angle(
    //     math::vector3<float>(0.0f, 1.0f, 0.0f), std::sin(angle));

    // Step 4: Generate the Model matrix from the updated transform
    const auto model_matrix = obj_transform.to_matrix();

    // Step 5: Compute the MVP matrix
    const auto projection_matrix = camera.get_projection_matrix();
    const auto mvp = projection_matrix * view_matrix * model_matrix;

    // Step 6: Populate the UniformsData struct without transposing
    _camera_uniform camera_data;
    camera_data.mvp = mvp;
    camera_data.position = {0, 0, -5};

    // Step 7: Write the UniformsData to the uniform buffer
    _queue.WriteBuffer(_camera_buffer,         // Destination buffer
                       0,                      // Offset in bytes
                       &camera_data,           // Pointer to the data
                       sizeof(_camera_uniform) // Size of the data in bytes
    );

    _model_uniform model_data;
    model_data.transform = model_matrix;

    _queue.WriteBuffer(_model_buffer,         // Destination buffer
                       0,                     // Offset in bytes
                       &model_data,           // Pointer to the data
                       sizeof(_model_uniform) // Size of the data in bytes
    );
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
  wgpu::Buffer _camera_buffer;              // Uniform buffer for matrices
  wgpu::Buffer _model_buffer;               // Uniform buffer for matrices
  wgpu::Buffer _light_buffer;               // Storage buffer for lights
  wgpu::Buffer _tile_buffer;                // Storage buffer for tiles
  wgpu::Texture _depth_texture;             // Depth texture
  wgpu::BindGroup _bind_group;              // Bind group for uniform buffer
  wgpu::BindGroupLayout _bind_group_layout; // Bind group layout
  wgpu::ShaderModule _shader_module;        // Shader module
  wgpu::PipelineLayout _pipeline_layout;    // Pipeline layout
  wgpu::RenderPipeline _pipeline;           // Render pipeline

  // Application data
  int _width, _height; // Window dimensions
  GLFWwindow *_window; // GLFW window handle
};

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

  auto renderer = forward_renderer::create(window);
  if (!renderer) {
    std::cerr << "Failed to create application" << std::endl;
    return 1;
  }

  ecs_context ctx;

  auto camera_entity = ctx.new_entity();
  ctx.add_component(camera_entity, scene3d::transform<float>{
                                       .position = {0.0f, 0.0f, -10.0f}});

  ctx.add_component(camera_entity, [&]() {
    auto camera = scene3d::camera<float>();
    camera.set_fov(60.0f * (std::numbers::pi_v<float> / 180.0f));
    camera.set_aspect_ratio(static_cast<float>(width) / height);
    camera.set_far_plane(1000.0f);

    return camera;
  }());

  auto sphere1 = ctx.new_entity();
  ctx.add_component(sphere1, scene3d::transform<float>{.position = {0, -2, 0}});
  ctx.add_component(sphere1, scene3d::mesh_instance::create_sphere(
                                 renderer->get_device(), 1.0f, 10, 10));

  auto sphere2 = ctx.new_entity();
  ctx.add_component(sphere2, scene3d::transform<float>{.position = {0, 2, 0}});
  ctx.add_component(sphere2, scene3d::mesh_instance::create_sphere(
                                 renderer->get_device(), 1.0f, 10, 10));

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    double currentTime = glfwGetTime(); // Get the elapsed time in seconds
    float angle = static_cast<float>(currentTime);

    // Optional: Update object rotation if needed
    ctx.for_each_entity<scene3d::transform<float>, scene3d::mesh_instance>(
        [&](scene3d::transform<float> &xform, const auto &_) {
          auto xr = math::quaternion<float>::from_axis_angle(
              math::vector3<float>(1.0f, 0.0f, 0.0f), std::sin(angle));

          auto yr = math::quaternion<float>::from_axis_angle(
              math::vector3<float>(0.0f, 1.0f, 0.0f), std::sin(-angle));

          xform.rotation = xr * yr;
        });

    renderer->render(ctx);

    std::this_thread::sleep_for(std::chrono::milliseconds(4));
  }

  return 0;
}