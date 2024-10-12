cmake_minimum_required(VERSION 3.28)

project(opus-engine)

add_library(
  ${PROJECT_NAME} INTERFACE
)

target_compile_features(
  ${PROJECT_NAME} INTERFACE 
    cxx_std_23
)

target_include_directories(
  ${PROJECT_NAME} INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)

find_package(X11 REQUIRED)


target_link_libraries(${PROJECT_NAME} INTERFACE X11 OpenGL::GL GLEW)

add_subdirectory(tests)