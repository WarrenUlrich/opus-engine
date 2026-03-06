cmake_minimum_required(VERSION 3.14)
project(opus-engine C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Fetch the latest Sokol headers automatically
include(FetchContent)
FetchContent_Declare(
    sokol
    GIT_REPOSITORY https://github.com/floooh/sokol.git
    GIT_TAG master
)
FetchContent_MakeAvailable(sokol)

# 2. Define the executable
add_executable(opus-engine src/main.cpp)

# 3. Include the Sokol headers
target_include_directories(opus-engine PRIVATE include ${sokol_SOURCE_DIR})

# 4. Set the backend (GL Core 3.3 for modern desktop Linux)
target_compile_definitions(opus-engine PRIVATE SOKOL_GLCORE)

# 5. Link required system libraries
target_link_libraries(opus-engine PRIVATE
    GL
    X11
    Xcursor
    Xi
    Xrandr
    dl
    m
    pthread
)