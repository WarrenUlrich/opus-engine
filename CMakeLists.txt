cmake_minimum_required(VERSION 3.14)
project(opus-engine C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)

# ---------------------------------------------------------------------------
# Build option: editor UI (ImGui) or standalone game runtime
# ---------------------------------------------------------------------------
option(OPUS_EDITOR "Build with the editor UI (ImGui docking, panels, grid)" ON)

include(FetchContent)
FetchContent_Declare(
    sokol
    GIT_REPOSITORY https://github.com/floooh/sokol.git
    GIT_TAG master
)

if(OPUS_EDITOR)
    # Dear ImGui – docking branch for dockspace / viewport support
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG docking
    )
    FetchContent_MakeAvailable(sokol imgui)

    # Build Dear ImGui as a static library
    add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    )
    target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR})
else()
    FetchContent_MakeAvailable(sokol)
endif()

# ---------------------------------------------------------------------------
# Engine / Editor executable
# ---------------------------------------------------------------------------
set(ENGINE_SOURCES src/main.cpp src/sokol_impl.cpp)
if(OPUS_EDITOR)
    list(APPEND ENGINE_SOURCES src/sokol_imgui_impl.cpp)
endif()

add_executable(opus-engine ${ENGINE_SOURCES})

target_include_directories(opus-engine PRIVATE
    include
    ${sokol_SOURCE_DIR}
    ${sokol_SOURCE_DIR}/util
)

target_compile_definitions(opus-engine PRIVATE SOKOL_GLCORE)

if(OPUS_EDITOR)
    target_compile_definitions(opus-engine PRIVATE OPUS_EDITOR)
    target_include_directories(opus-engine PRIVATE ${imgui_SOURCE_DIR})
    target_link_libraries(opus-engine PRIVATE imgui)
endif()

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

# Export symbols so the hot-reloaded game DLL can resolve engine functions
target_link_options(opus-engine PRIVATE -rdynamic)

# ---------------------------------------------------------------------------
# Hot-reloadable game DLL (scripts)
# ---------------------------------------------------------------------------
add_library(opus-game SHARED
    game/register.cpp
)

target_include_directories(opus-game PRIVATE
    include
    ${sokol_SOURCE_DIR}
    ${sokol_SOURCE_DIR}/util
)

if(OPUS_EDITOR)
    target_compile_definitions(opus-game PRIVATE OPUS_EDITOR)
    target_include_directories(opus-game PRIVATE ${imgui_SOURCE_DIR})
endif()

target_compile_definitions(opus-game PRIVATE SOKOL_GLCORE)

set_target_properties(opus-game PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)