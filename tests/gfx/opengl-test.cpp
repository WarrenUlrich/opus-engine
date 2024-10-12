#include <X11/Xlib.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <sstream>
#include <vector>

// Include GLEW
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

struct ShaderComponent {
  gfx::shader_program<gfx::render_backend::opengl> *shaderProgram;
};

// Function to create an X11 window
Window create_window(Display *display, int width, int height) {
  int screen = DefaultScreen(display);
  Window root_window = RootWindow(display, screen);

  // Create an X11 window
  Window window = XCreateSimpleWindow(display, root_window, 10, 10, width,
                                      height, 1, BlackPixel(display, screen),
                                      WhitePixel(display, screen));

  // Select the kind of events we want to listen to
  XSelectInput(display, window, ExposureMask | KeyPressMask);

  // Map the window to the display
  XMapWindow(display, window);

  return window;
}

int main() {
  Display *display = XOpenDisplay(nullptr);
  if (!display) {
    std::cerr << "Failed to open X display!" << std::endl;
    return -1;
  }

  int window_width = 1920 / 2;
  int window_height = 1080 / 2;
  Window window = create_window(display, window_width, window_height);

  gfx::feature_set features; // Use an empty feature set for now
  auto gl_context = gfx::renderer_context<gfx::render_backend::opengl>(display, window);
  if (!gl_context.init(features)) {
    std::cerr << "Failed to initialize OpenGL context!" << std::endl;
    return -1;
  }

  auto win = syswin::window<gfx::render_backend::opengl>::create(
      "opengl example", 1920 / 2, 1080 / 2);
  
  GLenum err = glewInit();
  // GLEW may cause an OpenGL error, so clear it
  glGetError();
  if (err != GLEW_OK) {
    std::cerr << "Error initializing GLEW: " << glewGetErrorString(err)
              << std::endl;
    return -1;
  }

  // Enable depth test
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  std::string vertex_shader_src = R"(
    #version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec2 aTexCoord;
    layout(location = 2) in vec3 aNormal;

    uniform mat4 MVP;

    out vec2 TexCoord;
    out vec3 Normal; // Pass normal to fragment shader

    void main() {
        gl_Position = MVP * vec4(aPos, 1.0);
        TexCoord = aTexCoord;
        Normal = aNormal; // Assign normal
    }
)";

  std::string fragment_shader_src = R"(
    #version 330 core
    out vec4 FragColor;

    in vec2 TexCoord;
    in vec3 Normal; // Receive normal from vertex shader

    void main() {
        // Normalize the normal vector
        vec3 norm = normalize(Normal);
        // Visualize normals by mapping them to colors
        FragColor = vec4(norm * 0.5 + 0.5, 1.0); // Colors range from (0,0,0) to (1,1,1)
    }
)";

  auto shader = gfx::shader_program<gfx::render_backend::opengl>(
      vertex_shader_src, fragment_shader_src);

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

  auto cube_mesh = std::make_shared<gfx::mesh<gfx::render_backend::opengl>>(
      gfx::vbuffer<gfx::render_backend::opengl>(
          vertex_data.data(), vertex_data.size() * sizeof(float), layout),
      vertex_data.size() * sizeof(float));

  // Record the start time
  auto start_time = std::chrono::high_resolution_clock::now();

  // Create a camera using the new scene::camera3d
  auto camera = scene::camera3d<float>();
  camera.set_fov(radians(60.0f));
  camera.set_aspect_ratio(static_cast<float>(window_width) / window_height);

  math::matrix4x4<float> view_matrix = math::matrix4x4<float>::translation(
      math::vector3<float>(0.0f, 0.0f, -5.0f));

  // Create ECS context with defined components
  auto ctx = ecs::context<scene::transform3d<float>,
                          scene::mesh_instance3d<gfx::render_backend::opengl>,
                          ShaderComponent>();

  // Shared components
  ShaderComponent shader_component = {&shader};

  // Create entity for the cube
  auto cube_entity = ctx.new_entity();

  ctx.add_component(cube_entity, scene::transform3d<float>());
  ctx.add_component(
      cube_entity,
      scene::mesh_instance3d<gfx::render_backend::opengl>(cube_mesh));
  ctx.add_component(cube_entity, std::move(shader_component));

  // Set up your basic render loop

  auto pipeline = gfx::render_pipeline2<gfx::render_backend::opengl>();
  bool running = true;
  while (running) {
    // Handle X11 events
    XEvent event;
    while (XPending(display)) {
      XNextEvent(display, &event);
    }

    pipeline.begin_frame();

    // Update time
    auto current_time = std::chrono::high_resolution_clock::now();
    float timeValue =
        std::chrono::duration<float>(current_time - start_time).count();

    ctx.for_each_entity<scene::transform3d<float>,
                        scene::mesh_instance3d<gfx::render_backend::opengl>,
                        ShaderComponent>(
        [&](ecs::entity_id id, scene::transform3d<float> &transform,
            scene::mesh_instance3d<gfx::render_backend::opengl> &mesh_instance,
            ShaderComponent &shaderComp) {
          transform.rotation = math::quaternion<float>::from_axis_angle(
              math::vector3<float>(1.0f, 0.0f, 0.0f), timeValue);

          // Compute the model matrix
          math::matrix4x4<float> model_matrix = transform.to_matrix();

          // Compute the MVP matrix
          math::matrix4x4<float> MVP =
              camera.get_projection_matrix() * view_matrix * model_matrix;

          // Use the shader program
          shaderComp.shaderProgram->use();

          pipeline.submit(gfx::set_uniform_cmd<gfx::render_backend::opengl,
                                               math::matrix4x4<float>>(
              "MVP", MVP, *shaderComp.shaderProgram));

          pipeline.submit(gfx::draw_mesh_cmd<gfx::render_backend::opengl>(
              *mesh_instance.mesh));
        });

    pipeline.end_frame();

    if (!gl_context.swap_buffers()) {
      std::cerr << "Failed to swap buffers!" << std::endl;
      return -1;
    }
  }

  // Close X11 display
  XCloseDisplay(display);

  return 0;
}
