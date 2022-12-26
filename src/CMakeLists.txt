

cmake_minimum_required(VERSION 3.14)

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

set(GPUSDR_HEADER_DIR "${PROJECT_SOURCE_DIR}/include")

FILE(GLOB headers "${GPUSDR_HEADER_DIR}/*")
add_library(gpusdr
        ${headers}
        AddConst.cu
        AddConstToVectorLength.cu
        #am_test.cpp
        Buffer.cpp
        CosineSource.cu
        fir.cu
        CudaBuffers.cpp
        HackrfSession.cpp
        HackrfSource.cpp
        Magnitude.cu
        Multiply.cu
        prototype/CudaLowPassBlock.cpp
        prototype/fm_deemph.cpp
        prototype/fm_pipeline.cpp
        prototype/low_pass_filter.cpp
        prototype/nbfm_rx.cpp
        prototype/wbfm_rx.cpp
        remez.cpp
        remez-orig.c
        S8ToFloat.cu
        Thread.cpp
)


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
        gpusdr SYSTEM PRIVATE
        /usr/local/lib/x86_64-linux-gnu
        /lib/x86_64-linux-gnu/
        ${GNURADIO_RUNTIME_LIBRARY_DIRS}
)

target_include_directories(
        gpusdr PUBLIC ${GPUSDR_HEADER_DIR}
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

add_subdirectory(applications)

