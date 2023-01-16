cmake_minimum_required(VERSION 3.14)

set( LIB_PROJECT_NAME gpusdrpipeline )
set( LIB_TARGET_NAME ${LIB_PROJECT_NAME} )
set( PROJECT_MACRO_PREFIX GPUSDR )

project(${LIB_PROJECT_NAME})

set( LIB_PROJECT_ROOT_DIR "${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

add_subdirectory(src)

option(USE_TESTS "Enable unit testing" ON)
if (USE_TESTS)
    message("Tests enabled.")
    enable_testing()
    add_subdirectory(tests)
endif ()
