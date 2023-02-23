cmake_minimum_required(VERSION 3.22)
project(heightmap CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "-G")

add_executable(heightmap main.cu noise.cu PerlinNoiseGPU.cuh mapGen.cu mapGen.cuh)

find_package(OpenGL REQUIRED)
find_package(GLU REQUIRED)
find_package(GLUT REQUIRED)

target_link_libraries(heightmap OpenGL glut GLU)

set_target_properties(heightmap PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
