cmake_minimum_required(VERSION 3.10)
project(PhysicsEngine)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(PhysicsEngine main.cpp shader_utils.hpp)
target_link_libraries(PhysicsEngine PRIVATE glfw GLEW OpenGL::GL)
