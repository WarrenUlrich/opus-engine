cmake_minimum_required(VERSION 3.28)

project(opus-engine)

# Add Dawn as a subdirectory
add_subdirectory(dawn)

# Create opus-engine library
add_library(${PROJECT_NAME} INTERFACE)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_23)

# Include directories for opus-engine
target_include_directories(
  ${PROJECT_NAME} INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link dependencies
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(X11 REQUIRED)

target_link_libraries(
  ${PROJECT_NAME} INTERFACE 
  X11 OpenGL::GL GLEW         
  dawn::dawn_proc
  dawn::dawn_glfw
  dawn::dawn_native
  glfw
)

# Add subdirectories (for tests, etc.)
add_subdirectory(tests)
