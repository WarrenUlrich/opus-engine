cmake_minimum_required(VERSION 3.5)

# opengl-test
add_executable(
  opengl-test
    gfx/opengl-test.cpp
)

target_compile_features(
  opengl-test PUBLIC
   cxx_std_23
)

target_link_libraries(
  opengl-test PUBLIC
    opus-engine
)

add_executable(
  dawn-test
    dawn-test/main.cpp
)

target_compile_features(
  dawn-test PUBLIC
   cxx_std_23
)

target_link_libraries(
  dawn-test PUBLIC
    opus-engine
)