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

# Define the path to the assets directory
set(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")

# Custom command to copy the assets directory next to the binary after build
add_custom_command(
    TARGET forward-renderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${ASSETS_DIR} $<TARGET_FILE_DIR:forward-renderer>/resources
    COMMENT "Copying assets directory next to the binary"
)

