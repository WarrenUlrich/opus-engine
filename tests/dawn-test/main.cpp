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

class application {
public:
  // Factory method to create an instance of the application
  static std::optional<application> create(int width, int height) {
    dawnProcSetProcs(&dawn::native::GetProcs());

    application app;
    app._width = width;
    app._height = height;

    // Initialize WebGPU components
    if (!app.initialize_webgpu()) {
      return std::nullopt;
    }

    // Initialize GLFW and create a window
    if (!app.initialize_glfw()) {
      return std::nullopt;
    }

    // Create depth texture for rendering
    app.create_depth_texture();

    // Create sphere mesh using mesh_instance
    app._cube_mesh =
        scene3d::mesh_instance::create_sphere(app._device, 0.5f, 20, 20);

    // Create uniform buffer
    app.create_buffers();

    // Create shader modules
    app.create_shader();

    // Set up bind group layout and bind group
    app.create_bind_group_layout();
    app.create_bind_group();

    // Set up pipeline layout and create the render pipeline
    app.create_pipeline_layout();
    app.create_pipeline();

    return app;
  }

  // Main application loop
  bool run() {
    // Set up the camera
    auto camera = scene3d::camera<float>();
    camera.set_fov(radians(60.0f));
    camera.set_aspect_ratio(static_cast<float>(_width) / _height);
    camera.set_far_plane(100.0f);

    auto camera_transform = scene3d::transform<float>();
    camera_transform.position.z = -5.0;

    // Transform for the sphere
    auto transform = scene3d::transform<float>();

    while (!glfwWindowShouldClose(this->_window)) {
      glfwPollEvents();

      double currentTime = glfwGetTime(); // Get the elapsed time in seconds
      float angle = static_cast<float>(
          currentTime); // You can adjust the speed multiplier if desired

      // Update application state (uniform buffers)
      update_uniform_buffer(camera, camera_transform.to_matrix(), transform);

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
          lights[i].type = _light_uniform::directional;
          lights[i].direction = math::vector3<float>{
              0.0f, -1.0f, 0.0f}; // Default downward direction
          lights[i].color = math::vector3<float>{0.0f, 0.0f, 1.0f};
        } else {
          lights[i].type = _light_uniform::spot;
          lights[i].direction =
              math::vector3<float>{0.0f, -1.0f, 0.0f}; // Spotlight direction
          lights[i].cutoff_angle = 25.0f; // Spotlights' cutoff angle in degrees
          lights[i].color = math::vector3<float>{1.0f, 0.0f, 0.0f};
        }

        // Modulate position, intensity, and color with time for randomness
        lights[i].position = math::vector3<float>{
            i * 0.1f + randomOffsetX * std::sin(timeFactor + i),
            i * 0.1f + randomOffsetY * std::cos(timeFactor + i),
            i * 0.1f + randomOffsetZ * std::sin(timeFactor * 0.5f + i)};

        // Randomly vary intensity between 0.5 and 1.5 over time
        lights[i].intensity = 1.0f + std::sin(timeFactor + i) * 0.5f;

        // Keep the radius constant for point lights or spotlights
        lights[i].radius = (lights[i].type == _light_uniform::point ||
                            lights[i].type == _light_uniform::spot)
                               ? 10.0f
                               : 0.0f;
      }

      // Update buffers
      update_light_buffer(lights);

      // Render a frame
      if (!render_frame()) {
        return false;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return true;
  }

private:
  struct alignas(16) _camera_uniform {
    math::matrix4x4<float> mvp;
    math::matrix4x4<float> model;
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

  application() = default;

  // Initialize WebGPU instance, adapter, and device
  bool initialize_webgpu() {
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
    auto adapter_options = wgpu::RequestAdapterOptions();
    adapter_options.backendType = wgpu::BackendType::Vulkan;
    adapter_options.powerPreference = wgpu::PowerPreference::HighPerformance;

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
  bool initialize_glfw() {
    glfwSetErrorCallback([](int code, const char *message) {
      std::cerr << "GLFW error: " << code << " - " << message << '\n';
    });

    if (!glfwInit()) {
      return false;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    _window =
        glfwCreateWindow(_width, _height, "WebGPU 3D Sphere", nullptr, nullptr);

    if (!_window) {
      std::cerr << "Failed to create GLFW window.\n";
      return false;
    }

    // Create a surface for the window
    _surface = wgpu::glfw::CreateSurfaceForWindow(_instance, _window);

    // Configure the surface
    auto surface_caps = wgpu::SurfaceCapabilities();
    _surface.GetCapabilities(_adapter, &surface_caps);

    auto surface_config =
        wgpu::SurfaceConfiguration{.device = _device,
                                   .format = surface_caps.formats[0],
                                   .width = static_cast<uint32_t>(_width),
                                   .height = static_cast<uint32_t>(_height)};

    _surface.Configure(&surface_config);

    _preferred_texture_format = surface_caps.formats[0];
    return true;
  }

  // Create a depth texture for depth testing
  void create_depth_texture() {
    const auto depth_descriptor = wgpu::TextureDescriptor{
        .usage = wgpu::TextureUsage::RenderAttachment,
        .size = wgpu::Extent3D{.width = static_cast<uint32_t>(_width),
                               .height = static_cast<uint32_t>(_height)},
        .format = wgpu::TextureFormat::Depth24PlusStencil8,
    };

    _depth_texture = _device.CreateTexture(&depth_descriptor);
  }

  void create_buffers() {
    // Uniform buffer for matrices
    wgpu::BufferDescriptor uniformBufferDescriptor = {
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = sizeof(math::matrix4x4<float>) * 2};

    _camera_buffer = _device.CreateBuffer(&uniformBufferDescriptor);

    // Storage buffer for lights
    wgpu::BufferDescriptor lightBufferDescriptor = {
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        .size = sizeof(_light_uniform) * 256};
    _light_buffer = _device.CreateBuffer(&lightBufferDescriptor);

    // Storage buffer for tiles
    wgpu::BufferDescriptor tileBufferDescriptor = {
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        .size = sizeof(_tile_uniform) * (16 * 16)};
    _tile_buffer = _device.CreateBuffer(&tileBufferDescriptor);
  }

  // Create the shader module from WGSL code
  void create_shader() {
    const char *shader_src = R"(
struct CameraUniform {
  mvp: mat4x4<f32>,
  model: mat4x4<f32>,
};

struct LightUniform {
  position: vec3<f32>,    // Position for point/spot lights, ignored for directional lights
  intensity: f32,         // Intensity of the light
  color: vec3<f32>,       // Light color
  radius: f32,            // Radius for point/spot lights, 0 for directional lights
  direction: vec3<f32>,   // Light direction for directional/spotlights, ignored for point lights
  cutoff_angle: f32,      // Cutoff angle for spotlights, 0 for other lights
  light_type: u32,        // 0 = Point light, 1 = Directional light, 2 = Spotlight
};

@group(0) @binding(0) var<uniform> uniforms: CameraUniform;
@group(0) @binding(1) var<storage, read> lights: array<LightUniform, 256>;

struct VertexInput {
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) Position: vec4<f32>,
  @location(0) fragUV: vec2<f32>,
  @location(1) fragNormal: vec3<f32>,
  @location(2) fragPos: vec3<f32>, // Pass the fragment position to the fragment shader
};

@vertex
fn vs(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.Position = calculateMVP(input.pos);
  output.fragUV = input.uv;
  output.fragNormal = calculateNormal(input.normal);
  output.fragPos = calculateFragPosition(input.pos);
  return output;
}

fn calculateMVP(pos: vec3<f32>) -> vec4<f32> {
  return uniforms.mvp * vec4<f32>(pos, 1.0);
}

fn calculateNormal(normal: vec3<f32>) -> vec3<f32> {
  return normalize((uniforms.model * vec4<f32>(normal, 0.0)).xyz);
}

fn calculateFragPosition(pos: vec3<f32>) -> vec3<f32> {
  return (uniforms.model * vec4<f32>(pos, 1.0)).xyz;
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
  let checkerColor = calculateCheckerPattern(input.fragUV);
  let normal = normalize(input.fragNormal);
  var finalColor = vec3<f32>(0.1); // Ambient light base

  for (var i = 0u; i < 256u; i = i + 1u) {
    let light = lights[i];
    finalColor += calculateLighting(light, input.fragPos, normal);
  }

  let baseColor = vec3<f32>(checkerColor, checkerColor, checkerColor);
  finalColor = baseColor * finalColor;

  return vec4<f32>(finalColor, 1.0); // Output color with full opacity
}

fn calculateCheckerPattern(uv: vec2<f32>) -> f32 {
  let scaledUV = floor(30.0 * uv);
  return 0.2 + 0.5 * ((scaledUV.x + scaledUV.y) - 2.0 * floor((scaledUV.x + scaledUV.y) / 2.0));
}

fn calculateLighting(light: LightUniform, fragPos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
  var lightColor = vec3<f32>(0.0);

  if (light.light_type == 0u) {
    lightColor = calculatePointLight(light, fragPos, normal);
  } else if (light.light_type == 1u) {
    lightColor = calculateDirectionalLight(light, normal);
  } else if (light.light_type == 2u) {
    lightColor = calculateSpotLight(light, fragPos, normal);
  }

  return lightColor;
}

fn calculatePointLight(light: LightUniform, fragPos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
  let lightDir = normalize(light.position - fragPos);
  let distance = length(light.position - fragPos);
  let attenuation = light.intensity / (1.0 + light.radius * distance * distance);
  let diffuse = max(dot(normal, lightDir), 0.0);
  return light.color * diffuse * attenuation;
}

fn calculateDirectionalLight(light: LightUniform, normal: vec3<f32>) -> vec3<f32> {
  let lightDir = normalize(light.direction);
  let diffuse = max(dot(normal, lightDir), 0.0);
  return light.color * diffuse * light.intensity;
}

fn calculateSpotLight(light: LightUniform, fragPos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
  let lightDir = normalize(light.position - fragPos);
  let distance = length(light.position - fragPos);
  let attenuation = light.intensity / (1.0 + light.radius * distance * distance);
  let spotEffect = dot(normalize(light.direction), -lightDir);

  if (spotEffect > cos(radians(light.cutoff_angle))) {
    let diffuse = max(dot(normal, lightDir), 0.0);
    return light.color * diffuse * attenuation * spotEffect;
  }

  return vec3<f32>(0.0);
}
)";

    wgpu::ShaderModuleWGSLDescriptor wgsl_desc = {};
    wgsl_desc.code = shader_src;

    wgpu::ShaderModuleDescriptor module_descriptor = {};
    module_descriptor.nextInChain =
        reinterpret_cast<const wgpu::ChainedStruct *>(&wgsl_desc);

    _shader_module = _device.CreateShaderModule(&module_descriptor);
  }

  // Create the bind group layout for uniform and storage buffers
  void create_bind_group_layout() {
    const wgpu::BindGroupLayoutEntry bgl_entries[] = {
        // Camera entry
        {.binding = 0,
         .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
         .buffer = {.type = wgpu::BufferBindingType::Uniform}},
        // Light entry
        {.binding = 1,
         .visibility = wgpu::ShaderStage::Fragment,
         .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}},
        // Tile entry
        {.binding = 2,
         .visibility = wgpu::ShaderStage::Fragment,
         .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}},
    };

    wgpu::BindGroupLayoutDescriptor bgl_descriptor = {
        .entryCount = std::size(bgl_entries), .entries = bgl_entries};

    _bind_group_layout = _device.CreateBindGroupLayout(&bgl_descriptor);
  }

  // Create the bind group for uniform and storage buffers
  void create_bind_group() {
    const wgpu::BindGroupEntry bg_entries[] = {
        // Camera entry
        {.binding = 0,
         .buffer = _camera_buffer,
         .size = sizeof(_camera_uniform)},
        // Light entry
        {.binding = 1,
         .buffer = _light_buffer,
         .size = sizeof(_light_uniform) * 256},
        // Tile entry
        {.binding = 2,
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
  void create_pipeline_layout() {
    wgpu::PipelineLayoutDescriptor layout_descriptor = {
        .bindGroupLayoutCount = 1, .bindGroupLayouts = &_bind_group_layout};
    _pipeline_layout = _device.CreatePipelineLayout(&layout_descriptor);
  }

  // Create the render pipeline
  void create_pipeline() {
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
        .label = nullptr,
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
                             scene3d::transform<float> &transform) {
    // Step 1: Calculate the current rotation angle based on elapsed time
    double currentTime = glfwGetTime(); // Get the elapsed time in seconds
    float angle = static_cast<float>(
        currentTime); // You can adjust the speed multiplier if desired

    // Step 2: Create rotation quaternions around the Y and Z axes
    auto yr = math::quaternion<float>::from_axis_angle(
        math::vector3<float>(0.0f, 1.0f, 0.0f), std::sin(angle));
    auto zr = math::quaternion<float>::from_axis_angle(
        math::vector3<float>(0.0f, 0.0f, 1.0f), std::cos(angle));

    // Step 3: Update the transform's rotation by combining Y and Z rotations
    transform.rotation = yr * zr;

    // Step 4: Generate the Model matrix from the updated transform
    math::matrix4x4<float> model_matrix = transform.to_matrix();

    // Step 5: Compute the MVP matrix
    math::matrix4x4<float> projection_matrix = camera.get_projection_matrix();
    math::matrix4x4<float> mvp = projection_matrix * view_matrix * model_matrix;

    // Step 6: Populate the UniformsData struct
    _camera_uniform data;
    data.mvp = mvp.transpose(); // Transpose if your math library uses row-major
    data.model = model_matrix.transpose(); // Transpose if necessary

    // Step 7: Write the UniformsData to the uniform buffer
    _queue.WriteBuffer(_camera_buffer,         // Destination buffer
                       0,                      // Offset in bytes
                       &data,                  // Pointer to the data
                       sizeof(_camera_uniform) // Size of the data in bytes
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

  // Render a single frame
  bool render_frame() {
    // Acquire the next texture from the surface
    wgpu::SurfaceTexture texture;
    _surface.GetCurrentTexture(&texture);

    if (!texture.texture) {
      std::cerr << "Failed to acquire next surface texture.\n";
      return false;
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
    const auto render_pass = encoder.BeginRenderPass(&render_pass_desc);

    // Set pipeline and vertex buffer
    render_pass.SetPipeline(_pipeline);
    render_pass.SetVertexBuffer(0, _cube_mesh.get_vertex_buffer());
    render_pass.SetIndexBuffer(_cube_mesh.get_index_buffer(),
                               wgpu::IndexFormat::Uint16);

    // Bind the bind group
    render_pass.SetBindGroup(0, _bind_group);

    // Issue draw call
    render_pass.DrawIndexed(_cube_mesh.get_index_count(), 1, 0, 0, 0);

    // End render pass
    render_pass.End();

    // Finish encoding commands
    const auto command_buffer = encoder.Finish();

    // Submit commands to the GPU queue
    _queue.Submit(1, &command_buffer);

    // Present the rendered image to the surface
    _surface.Present();

    return true;
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
  wgpu::Buffer _light_buffer;               // Storage buffer for lights
  wgpu::Buffer _tile_buffer;                // Storage buffer for tiles
  wgpu::Texture _depth_texture;             // Depth texture
  wgpu::BindGroup _bind_group;              // Bind group for uniform buffer
  wgpu::BindGroupLayout _bind_group_layout; // Bind group layout
  wgpu::ShaderModule _shader_module;        // Shader module
  wgpu::PipelineLayout _pipeline_layout;    // Pipeline layout
  wgpu::RenderPipeline _pipeline;           // Render pipeline

  // Application data
  int _width, _height;               // Window dimensions
  scene3d::mesh_instance _cube_mesh; // Cube mesh
  GLFWwindow *_window;               // GLFW window handle
};

int main(int argc, char **args) {
  auto app = application::create(1920 / 2, 1080);
  if (!app) {
    std::cerr << "Failed to create application" << std::endl;
    return 1;
  }

  app->run();
  return 0;
}
