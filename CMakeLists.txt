cmake_minimum_required(VERSION 3.19)

set(LIB_PROJECT_NAME gpusdrpipeline)
set(LIB_TARGET_NAME ${LIB_PROJECT_NAME})

set(CMAKE_CXX_COMPILER /usr/bin/clang++)
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_C_COMPILER /usr/bin/clang)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#set(PROJ_CUDA_ARCHITECTURES "75")

set(VERBOSE 1)

include_directories(
    SYSTEM
    /usr/include/c++/11
    /usr/include/x86_64-linux-gnu/c++/11
)

project(${LIB_PROJECT_NAME})

set(LIB_PROJECT_ROOT_DIR "${PROJECT_SOURCE_DIR}")

#enable_language(CUDA)
#include(CheckLanguage)
#check_language(CUDA)

#if (NOT DEFINED CMAKE_CUDA_STANDARD)
#    set(CMAKE_CUDA_STANDARD 17)
#    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#endif ()

add_subdirectory(src)

option(USE_TESTS "Enable unit testing" ON)
if (USE_TESTS)
    message("Tests enabled.")
    enable_testing()
    add_subdirectory(tests)
endif ()
