#include <X11/Xlib.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numbers>
#include <sstream>
#include <thread>
#include <vector>

#include <GL/glew.h>

#include <opus/gfx/gfx.hpp>

#include <opus/math/math.hpp>

#include <opus/scene/scene.hpp>

#include <opus/ecs/ecs.hpp>

#include <opus/asset/obj_model.hpp>

#include <opus/syswin/window.hpp>

constexpr float radians(float degrees) {
  return degrees * (std::numbers::pi_v<float> / 180.0f);
}

int main() {
  constexpr auto window_width = 1920 / 2;
  constexpr auto window_height = 1080;

  auto window = syswin::window_builder()
                    .with_title("OpenGL 3D with G-buffer")
                    .with_size(window_width, window_height)
                    .build();
  if (!window)
    return -1;

  gfx::feature_set features; // Use an empty feature set for now
  auto gl_context = window->get_context();
  if (!gl_context.init(features)) {
    std::cerr << "Failed to initialize OpenGL context!" << std::endl;
    return -1;
  }

  GLenum err = glewInit();
  glGetError();
  if (err != GLEW_OK) {
    std::cerr << "Error initializing GLEW: " << glewGetErrorString(err)
              << std::endl;
    return -1;
  }

  // Enable depth test
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  // First Pass Shaders (G-buffer shaders)

  // Vertex Shader
  std::string gbuffer_vertex_shader_src = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec2 aTexCoord;
        layout(location = 2) in vec3 aNormal;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        out vec2 TexCoord;
        out vec3 FragPos;
        out vec3 Normal;

        void main() {
            FragPos = vec3(model * vec4(aPos, 1.0));
            gl_Position = projection * view * vec4(FragPos, 1.0);
            TexCoord = aTexCoord;
            Normal = mat3(transpose(inverse(model))) * aNormal;
        }
    )";

  // Fragment Shader
  std::string gbuffer_fragment_shader_src = R"(
        #version 330 core
        layout(location = 0) out vec3 gPosition;
        layout(location = 1) out vec3 gNormal;
        layout(location = 2) out vec4 gAlbedoSpec;

        in vec2 TexCoord;
        in vec3 FragPos;
        in vec3 Normal;

        void main() {
            gPosition = FragPos;
            gNormal = normalize(Normal);
            gAlbedoSpec.rgb = vec3(1.0, 0.5, 0.31); // Placeholder albedo
            gAlbedoSpec.a = 1.0; // Specular intensity, placeholder
        }
    )";

  auto gbuffer_shader = gfx::shader_program(gbuffer_vertex_shader_src,
                                            gbuffer_fragment_shader_src);

  // Second Pass Shaders (Display the G-buffer textures)

  // Vertex Shader
  std::string quad_vertex_shader_src = R"(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aTexCoord;

        out vec2 TexCoord;

        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
    )";

  // Fragment Shader
  std::string quad_fragment_shader_src = R"(
    #version 330 core
    out vec4 FragColor;

    in vec2 TexCoord;

    uniform sampler2D gPosition;
    uniform sampler2D gNormal;
    uniform sampler2D gAlbedoSpec;
    uniform sampler2D depthMap;

    uniform int displayMode; // 0 = position, 1 = normal, 2 = albedo, 3 = depth, 4 = lighting

    uniform float near;
    uniform float far;

    // Light properties
    uniform vec3 lightDirection;
    uniform vec3 lightColor;
    uniform float lightIntensity;

    // Camera position
    uniform vec3 cameraPos;

    float LinearizeDepth(float depth) {
        float z = depth * 2.0 - 1.0; // Back to NDC
        return (2.0 * near * far) / (far + near - z * (far - near));
    }

    void main() {
        if (displayMode == 0) {
            // Display Position
            vec3 position = texture(gPosition, TexCoord).rgb;
            FragColor = vec4(position / 10.0, 1.0); // Scale for visualization
        } else if (displayMode == 1) {
            // Display Normal
            vec3 normal = texture(gNormal, TexCoord).rgb;
            FragColor = vec4(normal * 0.5 + 0.5, 1.0); // Map normals to [0,1]
        } else if (displayMode == 2) {
            // Display Albedo
            vec4 albedoSpec = texture(gAlbedoSpec, TexCoord);
            FragColor = vec4(albedoSpec.rgb, 1.0);
        } else if (displayMode == 3) {
            // Display Depth
            float depth = texture(depthMap, TexCoord).r;
            float linearDepth = LinearizeDepth(depth) / far; // Normalize to [0,1]
            FragColor = vec4(vec3(1.0 - linearDepth), 1.0);
        } else if (displayMode == 4) {
            // Lighting pass
            // Retrieve data from G-buffer
            vec3 FragPos = texture(gPosition, TexCoord).rgb;
            vec3 Normal = normalize(texture(gNormal, TexCoord).rgb);
            vec3 Albedo = texture(gAlbedoSpec, TexCoord).rgb;
            float SpecularStrength = texture(gAlbedoSpec, TexCoord).a;

            // Ambient lighting
            vec3 ambient = 0.1 * Albedo;

            // Diffuse lighting
            vec3 lightDir = normalize(-lightDirection); // Direction from fragment to light
            float diff = max(dot(Normal, lightDir), 0.0);
            vec3 diffuse = diff * Albedo * lightColor * lightIntensity;

            // Specular lighting
            vec3 viewDir = normalize(cameraPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, Normal);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16.0); // 16 is shininess
            vec3 specular = spec * SpecularStrength * lightColor * lightIntensity;

            vec3 result = ambient + diffuse + specular;
            FragColor = vec4(result, 1.0);
        }
    }
)";

  auto quad_shader =
      gfx::shader_program(quad_vertex_shader_src, quad_fragment_shader_src);

  // Load the model
  std::fstream cube_fstream("/home/frodo/teapot.obj");

  auto model_opt = asset::obj_model::load_from(cube_fstream);

  if (!model_opt) {
    std::cerr << "Failed to load cube model!" << std::endl;
    return -1;
  }

  auto model = model_opt.value();
  std::vector<float> vertex_data = model.flatten();

  // Build the vertex layout
  gfx::vertex_layout layout =
      gfx::vertex_layout_builder()
          .with_attribute(gfx::vertex_attribute_semantic::position,
                          gfx::vertex_attribute_type::float32, 3)
          .with_attribute(gfx::vertex_attribute_semantic::tex_coord0,
                          gfx::vertex_attribute_type::float32, 2)
          .with_attribute(gfx::vertex_attribute_semantic::normal,
                          gfx::vertex_attribute_type::float32, 3)
          .build();

  auto cube_mesh = std::make_shared<gfx::mesh>(
      gfx::vbuffer(vertex_data.data(), vertex_data.size() * sizeof(float),
                   layout),
      vertex_data.size() * sizeof(float));

  // Record the start time
  auto start_time = std::chrono::high_resolution_clock::now();

  // Create a camera
  auto camera = scene::camera3d<float>();
  camera.set_fov(radians(60.0f));
  camera.set_aspect_ratio(static_cast<float>(window_width) / window_height);
  camera.set_far_plane(1000.0f);

  // Assuming camera has get_view_matrix(), otherwise use predefined view_matrix
  math::matrix4x4<float> view_matrix = math::matrix4x4<float>::translation(
      math::vector3<float>(0.0f, 0.0f, -5.0f));

  // Create ECS context with defined components
  auto ctx = ecs::context<scene::transform3d<float>, scene::mesh_instance3d>();

  // Create entity for the cube
  const auto cube_entity = ctx.new_entity();
  ctx.add_component(cube_entity, scene::transform3d<float>());
  ctx.add_component(cube_entity, scene::mesh_instance3d(std::move(cube_mesh)));

  auto pipeline = gfx::render_pipeline();

  window->show();

  // Create G-buffer Textures

  // Position color buffer
  auto gPosition_opt = gfx::texture2d::create_empty(
      window_width, window_height, GL_RGB16F, GL_RGB, GL_FLOAT);
  if (!gPosition_opt) {
    std::cerr << "Failed to create gPosition texture.\n";
    return -1;
  }
  auto gPosition = std::move(gPosition_opt.value());

  // Normal color buffer
  auto gNormal_opt = gfx::texture2d::create_empty(window_width, window_height,
                                                  GL_RGB16F, GL_RGB, GL_FLOAT);
  if (!gNormal_opt) {
    std::cerr << "Failed to create gNormal texture.\n";
    return -1;
  }
  auto gNormal = std::move(gNormal_opt.value());

  // Albedo + Specular color buffer
  auto gAlbedoSpec_opt = gfx::texture2d::create_empty(
      window_width, window_height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE);
  if (!gAlbedoSpec_opt) {
    std::cerr << "Failed to create gAlbedoSpec texture.\n";
    return -1;
  }
  auto gAlbedoSpec = std::move(gAlbedoSpec_opt.value());

  // Depth buffer
  auto depth_texture_opt = gfx::texture2d::create_empty(
      window_width, window_height, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT,
      GL_FLOAT);
  if (!depth_texture_opt) {
    std::cerr << "Failed to create depth texture.\n";
    return -1;
  }
  auto depth_texture = std::move(depth_texture_opt.value());

  // Create framebuffer
  auto gbuffer = gfx::frame_buffer();

  // Attach G-buffer textures
  if (!gbuffer.attach_texture(GL_COLOR_ATTACHMENT0, gPosition)) {
    std::cerr << "Failed to attach gPosition texture to framebuffer."
              << std::endl;
    return -1;
  }

  if (!gbuffer.attach_texture(GL_COLOR_ATTACHMENT1, gNormal)) {
    std::cerr << "Failed to attach gNormal texture to framebuffer."
              << std::endl;
    return -1;
  }

  if (!gbuffer.attach_texture(GL_COLOR_ATTACHMENT2, gAlbedoSpec)) {
    std::cerr << "Failed to attach gAlbedoSpec texture to framebuffer."
              << std::endl;
    return -1;
  }

  // Attach depth texture
  if (!gbuffer.attach_texture(GL_DEPTH_ATTACHMENT, depth_texture)) {
    std::cerr << "Failed to attach depth texture to framebuffer." << std::endl;
    return -1;
  }

  // Set draw buffers
  gbuffer.set_draw_buffers();

  if (!gbuffer.is_complete()) {
    std::cerr << "Framebuffer is not complete!" << std::endl;
    return -1;
  }

  // Create Fullscreen Quad
  float quadVertices[] = {
      // positions   // texCoords
      -1.0f, 1.0f,  0.0f, 1.0f, // Top-left
      -1.0f, -1.0f, 0.0f, 0.0f, // Bottom-left
      1.0f,  -1.0f, 1.0f, 0.0f, // Bottom-right

      -1.0f, 1.0f,  0.0f, 1.0f, // Top-left
      1.0f,  -1.0f, 1.0f, 0.0f, // Bottom-right
      1.0f,  1.0f,  1.0f, 1.0f  // Top-right
  };

  // Build the vertex layout for the quad
  gfx::vertex_layout quadLayout =
      gfx::vertex_layout_builder()
          .with_attribute(gfx::vertex_attribute_semantic::position,
                          gfx::vertex_attribute_type::float32, 2)
          .with_attribute(gfx::vertex_attribute_semantic::tex_coord0,
                          gfx::vertex_attribute_type::float32, 2)
          .build();

  // Create the mesh for the quad
  auto quadMesh = std::make_shared<gfx::mesh>(
      gfx::vbuffer(quadVertices, sizeof(quadVertices), quadLayout),
      sizeof(quadVertices));

  // Define the camera position
  math::vector3<float> cameraPos(0.0f, 0.0f, -5.0f);

  // Create a directional light
  auto light = scene::directional_light3d<float>(
      math::vector3<float>(0.0f, -1.0f,
                           -1.0f), // Direction pointing towards the scene
      math::vector3<float>(0.0f, 1.0f, 0.0f), // White light
      0.2f                                    // Intensity
  );

  // Main rendering loop
  bool running = true;
  while (running) {
    window->process_events<syswin::button_press, syswin::motion_event>(
        [&](const auto &event) { return true; });

    // Begin rendering frame
    pipeline.begin_frame();

    // First Pass: Render to G-buffer
    pipeline.submit(gfx::bind_framebuffer_cmd(&gbuffer));

    pipeline.submit(gfx::set_viewport_cmd(0, 0, window_width, window_height));

    pipeline.submit(gfx::clear_cmd(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    const auto current_time = std::chrono::high_resolution_clock::now();
    const auto timeValue =
        std::chrono::duration<float>(current_time - start_time).count();

    ctx.for_each_entity<scene::transform3d<float>, scene::mesh_instance3d>(
        [&](ecs::entity_id id, scene::transform3d<float> &transform,
            scene::mesh_instance3d &mesh_instance) {
          transform.rotation = math::quaternion<float>::from_axis_angle(
              math::vector3<float>(0.0f, 1.0f, 0.0f), timeValue);

          // Compute the model matrix
          const auto model_matrix = transform.to_matrix();

          // Compute the MVP matrices
          const auto projection_matrix = camera.get_projection_matrix();
          const auto view_matrix = math::matrix4x4<float>::translation(
              math::vector3<float>(0.0f, 0.0f, -5.0f));

          pipeline.submit(gfx::use_shader_cmd(gbuffer_shader));

          pipeline.submit(gfx::set_uniform_cmd<math::matrix4x4<float>>(
              "model", model_matrix, gbuffer_shader));

          pipeline.submit(gfx::set_uniform_cmd<math::matrix4x4<float>>(
              "view", view_matrix, gbuffer_shader));

          pipeline.submit(gfx::set_uniform_cmd<math::matrix4x4<float>>(
              "projection", projection_matrix, gbuffer_shader));

          pipeline.submit(gfx::draw_mesh_cmd(*mesh_instance.mesh));
        });

    // Unbind the framebuffer to render to the default framebuffer
    pipeline.submit(gfx::unbind_framebuffer_cmd());

    // Second Pass: Render the fullscreen quad to display the G-buffer textures
    pipeline.submit(gfx::set_viewport_cmd(0, 0, window_width, window_height));

    // Clear the default framebuffer
    pipeline.submit(gfx::clear_cmd(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    // Use the quad shader
    pipeline.submit(gfx::use_shader_cmd(quad_shader));

    // Bind G-buffer textures to texture units
    pipeline.submit(gfx::bind_texture_cmd(gPosition, 0));     // Texture unit 0
    pipeline.submit(gfx::bind_texture_cmd(gNormal, 1));       // Texture unit 1
    pipeline.submit(gfx::bind_texture_cmd(gAlbedoSpec, 2));   // Texture unit 2
    pipeline.submit(gfx::bind_texture_cmd(depth_texture, 3)); // Texture unit 3

    // Set the sampler uniforms
    pipeline.submit(gfx::set_uniform_cmd<int>("gPosition", 0, quad_shader));
    pipeline.submit(gfx::set_uniform_cmd<int>("gNormal", 1, quad_shader));
    pipeline.submit(gfx::set_uniform_cmd<int>("gAlbedoSpec", 2, quad_shader));
    pipeline.submit(gfx::set_uniform_cmd<int>("depthMap", 3, quad_shader));

    // Set the near and far plane uniforms
    float near_plane = 0.1f;
    float far_plane = 1000.0f;
    pipeline.submit(
        gfx::set_uniform_cmd<float>("near", near_plane, quad_shader));
    pipeline.submit(gfx::set_uniform_cmd<float>("far", far_plane, quad_shader));

    // Set the display mode to lighting pass
    int displayMode =
        4; // 0 = position, 1 = normal, 2 = albedo, 3 = depth, 4 = lighting
    pipeline.submit(
        gfx::set_uniform_cmd<int>("displayMode", displayMode, quad_shader));

    // Set the light uniforms
    pipeline.submit(gfx::set_uniform_cmd<math::vector3<float>>(
        "lightDirection", light.direction, quad_shader));
    pipeline.submit(gfx::set_uniform_cmd<math::vector3<float>>(
        "lightColor", light.color, quad_shader));
    pipeline.submit(gfx::set_uniform_cmd<float>("lightIntensity",
                                                light.intensity, quad_shader));

    // Set the camera position uniform
    pipeline.submit(gfx::set_uniform_cmd<math::vector3<float>>(
        "cameraPos", cameraPos, quad_shader));

    // Draw the quad
    pipeline.submit(gfx::draw_mesh_cmd(*quadMesh));

    // End rendering frame
    pipeline.end_frame();

    if (!gl_context.swap_buffers()) {
      std::cerr << "Failed to swap buffers!" << std::endl;
      running = false;
    }
  }

  return 0;
}
