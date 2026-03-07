cmake_minimum_required(VERSION 3.14)
project(opus-engine C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    sokol
    GIT_REPOSITORY https://github.com/floooh/sokol.git
    GIT_TAG master
)
FetchContent_MakeAvailable(sokol)

add_executable(opus-engine src/main.cpp)

target_include_directories(opus-engine PRIVATE include ${sokol_SOURCE_DIR})

target_compile_definitions(opus-engine PRIVATE SOKOL_GLCORE)

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