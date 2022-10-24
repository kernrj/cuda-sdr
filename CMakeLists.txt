cmake_minimum_required(VERSION 3.14)

project(sdrtest)

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

set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig")

set(VERBOSE 1)

find_package(PkgConfig)
pkg_search_module(HACKRF REQUIRED IMPORTED_TARGET libhackrf)

set(ENV{PKG_CONFIG_PATH} "/usr/local/lib/pkgconfig")

add_executable(sdrtest
        src/AddConst.cu
        src/AddConst.h
        src/AddConstToVectorLength.cu
        src/AddConstToVectorLength.h
        src/am_test.cpp
        src/Buffer.h
        src/CosineSource.cu
        src/CosineSource.h
        src/cuComplexOperatorOverloads.cuh
        src/cuda_util.h
        src/CudaDevicePushPop.h
        src/fir.cu
        src/fir.h
        src/fm.h
        src/prototype/fm_pipeline.cpp
        src/prototype/fm_pipeline.h
        src/CudaBuffers.cpp
        src/CudaBuffers.h
        src/HackrfSession.cpp
        src/HackrfSession.h
        src/HackrfSource.cpp
        src/HackrfSource.h
        src/Magnitude.cu
        src/Magnitude.h
        src/Multiply.cu
        src/Multiply.h
        src/S8ToFloat.cu
        src/S8ToFloat.h
        src/ScopeExit.h
        src/Thread.cpp
        src/Thread.h
        src/fir.h
        src/Filter.h)

target_include_directories(sdrtest
        PRIVATE SYSTEM
        /usr/include
        /usr/local/include
        /usr/local/cuda/include
        )

set_target_properties(sdrtest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_directories(
        sdrtest PRIVATE
        /usr/local/lib/x86_64-linux-gnu
)

set_target_properties(sdrtest PROPERTIES CUDA_ARCHITECTURES "75")

target_link_libraries(sdrtest
        PRIVATE
        hackrf
        nvcuvid
        nvidia-encode
)
