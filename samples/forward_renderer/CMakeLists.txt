cmake_minimum_required(VERSION 3.5)

# Define your executable
add_executable(
  forward-renderer
    src/main.cpp
)

# Set the C++ standard
target_compile_features(
  forward-renderer PUBLIC
   cxx_std_23
)

# Link libraries
target_link_libraries(
  forward-renderer PUBLIC
    opus-engine
)

# Set the runtime output directory (where the executable will be placed)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)