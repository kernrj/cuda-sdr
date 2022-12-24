cmake_minimum_required(VERSION 3.14)

project(gpusdr)

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

set(GR_REQUIRED_COMPONENTS ANALOG BLOCKS DIGITAL FFT FILTER RUNTIME)
find_package(Gnuradio "3.10.1" REQUIRED COMPONENTS analog blocks digital fft filter runtime)

set(VERBOSE 1)
set(gnuradio-osmosdr_DIR "/lib/x86_64-linux-gnu/cmake/osmosdr")
find_package(gnuradio-osmosdr REQUIRED)

find_package(PkgConfig)
pkg_search_module(HACKRF REQUIRED IMPORTED_TARGET libhackrf)

set(ENV{PKG_CONFIG_PATH} "/usr/local/lib/pkgconfig")

add_library(gpusdr
        src/AddConst.cu
        src/AddConst.h
        src/AddConstToVectorLength.cu
        src/AddConstToVectorLength.h
        #src/am_test.cpp
        src/Buffer.h
        src/Buffer.cpp
        src/CosineSource.cu
        src/CosineSource.h
        src/cuComplexOperatorOverloads.cuh
        src/cuda_util.h
        src/CudaDevicePushPop.h
        src/fir.cu
        src/fir.h
        src/fm.h
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
        src/prototype/CudaLowPassBlock.cpp
        src/prototype/CudaLowPassBlock.h
        src/prototype/fm_deemph.cpp
        src/prototype/fm_deemph.h
        src/prototype/fm_pipeline.cpp
        src/prototype/fm_pipeline.h
        src/prototype/low_pass_filter.cpp
        src/prototype/low_pass_filter.h
        src/prototype/nbfm_rx.cpp
        src/prototype/nbfm_rx.h
        src/prototype/wbfm_rx.cpp
        src/prototype/wbfm_rx.h
        src/remez.h
        src/remez.cpp
        src/remez-orig.c
        src/S8ToFloat.cu
        src/S8ToFloat.h
        src/ScopeExit.h
        src/Thread.cpp
        src/Thread.h
        src/fir.h
        src/Filter.h)


target_include_directories(gpusdr
        SYSTEM PRIVATE
        /usr/include
        /usr/local/include
        /usr/local/cuda/include
        ${GNURADIO_ALL_INCLUDE_DIRS}
        )

#message("[${HACKRF_LIBRARY_DIRS}] [${HACKRF_LIBRARY_DIR}]")
#target_link_directories( sdrtest
#  PRIVATE
#  ${HACKRF_LIBRARY_DIRS}
#)

set_target_properties(gpusdr PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_directories(
        gpusdr PRIVATE
        /usr/local/lib/x86_64-linux-gnu
        /lib/x86_64-linux-gnu/
        ${GNURADIO_RUNTIME_LIBRARY_DIRS}
)

set_target_properties(gpusdr PROPERTIES CUDA_ARCHITECTURES "75")

target_link_libraries(gpusdr
        PRIVATE
        hackrf
        nvcuvid
        nvidia-encode
        gnuradio::gnuradio-analog
        gnuradio::gnuradio-blocks
        gnuradio::gnuradio-digital
        gnuradio::gnuradio-fft
        gnuradio::gnuradio-filter
        gnuradio::gnuradio-runtime
        sndfile
        gnuradio-osmosdr
)

add_subdirectory(src/applications)

option(USE_TESTS "Enable unit testing" ON)
if (USE_TESTS)
    message("Tests enabled.")
    enable_testing()
    add_subdirectory(tests)
endif ()
