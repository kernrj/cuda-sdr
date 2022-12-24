cmake_minimum_required(VERSION 3.14)

project(sdrtest)

add_executable(sdrtest sdrtest.cpp)
set_target_properties(sdrtest PROPERTIES CUDA_ARCHITECTURES "75")
target_link_libraries(sdrtest PRIVATE gpusdr)
