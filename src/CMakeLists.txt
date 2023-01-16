cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
    gsdr
    #    GIT_REPOSITORY https://github.com/kernrj/gsdr.git
    #    GIT_TAG main
    SOURCE_DIR /home/rick/projects/gsdr
)
FetchContent_MakeAvailable(gsdr)

FetchContent_Declare(
    remez
    GIT_REPOSITORY https://github.com/kernrj/remez-exchange.git
    GIT_TAG main
)
FetchContent_MakeAvailable(remez)

set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig")

set(GR_REQUIRED_COMPONENTS ANALOG BLOCKS DIGITAL FFT FILTER RUNTIME)
find_package(Gnuradio "3.10.1" REQUIRED COMPONENTS analog blocks digital fft filter runtime)

set(VERBOSE 1)
set(gnuradio-osmosdr_DIR "/lib/x86_64-linux-gnu/cmake/osmosdr")
find_package(gnuradio-osmosdr REQUIRED)

find_package(PkgConfig)
pkg_search_module(HACKRF REQUIRED IMPORTED_TARGET libhackrf)

set(ENV{PKG_CONFIG_PATH} "/usr/local/lib/pkgconfig")

set(LIB_DEV_PUBLIC_HEADER_DIR "${PROJECT_SOURCE_DIR}/include")

FILE(GLOB_RECURSE PUBLIC_HEADERS "${LIB_DEV_PUBLIC_HEADER_DIR}/${LIB_TARGET_NAME}" *.h)
FILE(GLOB_RECURSE PRIVATE_HEADERS "${PROJECT_SOURCE_DIR}/src" *.h)

add_library(${LIB_TARGET_NAME}
    ${PUBLIC_HEADERS}
    ${PRIVATE_HEADERS}
    #am_test.cpp
    buffers/BufferPool.cpp
    buffers/BufferPoolFactory.cpp
    buffers/BufferRange.cpp
    buffers/BufferRangeFactory.cpp
    buffers/BufferSlice.cpp
    buffers/BufferSliceFactory.cpp
    buffers/BufferUtil.cpp
    buffers/CudaAllocator.cpp
    buffers/CudaAllocatorFactory.cpp
    buffers/CudaBufferCopier.cpp
    buffers/CudaBufferCopierFactory.cpp
    buffers/CudaMemSet.cpp
    buffers/OwnedBuffer.cpp
    buffers/RelocatableResizableBufferFactory.cpp
    buffers/RelocatableCudaBufferFactory.cpp
    buffers/RelocatableResizableBuffer.cpp
    buffers/ResizableBuffer.cpp
    buffers/ResizableBufferFactory.cpp
    buffers/SysMemAllocator.cpp
    buffers/SysMemBufferCopier.cpp
    buffers/BufferFactory.cpp
    HackrfSession.cpp
    Factories.cpp
    filters/AacFileWriter.cpp
    filters/AacFileWriter.h
    filters/AddConst.cpp
    filters/AddConst.h
    filters/AddConstToVectorLength.cpp
    filters/AddConstToVectorLength.h
    filters/BaseFilter.cpp
    filters/BaseSink.cpp
    filters/CosineSource.cpp
    filters/CosineSource.h
    filters/CudaMemcpyFilter.cpp
    filters/CudaMemcpyFilter.h
    filters/factories/AacFileWriterFactory.h
    filters/factories/AddConstFactory.h
    filters/factories/AddConstToVectorLengthFactory.h
    filters/factories/CosineSourceFactory.h
    filters/factories/CudaMemcpyFilterFactory.h
    filters/factories/FileReaderFactory.h
    filters/factories/FirFactory.h
    filters/factories/Int8ToFloatFactory.h
    filters/factories/MagnitudeFactory.h
    filters/factories/MultiplyFactory.h
    filters/factories/QuadDemodFactory.h
    filters/FileReader.cpp
    filters/FileReader.h
    filters/Fir.cpp
    filters/Fir.h
    filters/HackrfSource.cpp
    filters/HackrfSource.h
    filters/Int8ToFloat.cpp
    filters/Int8ToFloat.h
    filters/Magnitude.cpp
    filters/Magnitude.h
    filters/Multiply.cpp
    filters/Multiply.h
    filters/QuadDemod.cpp
    filters/QuadDemod.h
    prototype/CudaLowPassBlock.cpp
    prototype/CudaLowPassBlock.h
    prototype/fm_deemph.cpp
    prototype/fm_deemph.h
    prototype/fm_pipeline.cpp
    prototype/fm_pipeline.h
    prototype/low_pass_filter.cpp
    prototype/low_pass_filter.h
    prototype/nbfm_rx.cpp
    prototype/nbfm_rx.h
    prototype/wbfm_rx.cpp
    prototype/wbfm_rx.h
    util/CudaUtil.cpp
    util/Thread.cpp
    util/Window.cpp
    )

target_include_directories(${LIB_TARGET_NAME}
    SYSTEM PRIVATE
    /usr/include
    /usr/local/include
    /usr/local/cuda/include
    "${GNURADIO_ALL_INCLUDE_DIRS}"
    "${gsdr_SOURCE_DIR}/include"
    "${remez_SOURCE_DIR}/include"
    )

target_include_directories(${LIB_TARGET_NAME}
    BEFORE PRIVATE
    ${LIB_DEV_PUBLIC_HEADER_DIR}/${LIB_TARGET_NAME}
    )

target_include_directories(${LIB_TARGET_NAME}
    PRIVATE ${LIB_DEV_PUBLIC_HEADER_DIR}
    )

target_include_directories(
    ${LIB_TARGET_NAME}
    SYSTEM INTERFACE
    "$<INSTALL_INTERFACE:include>"
)
target_include_directories(${LIB_TARGET_NAME} INTERFACE "$<INSTALL_INTERFACE:include>")

#message("[${HACKRF_LIBRARY_DIRS}] [${HACKRF_LIBRARY_DIR}]")
#target_link_directories( sdrtest
#  PRIVATE
#  ${HACKRF_LIBRARY_DIRS}
#)

set_target_properties(${LIB_TARGET_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_link_directories(
    ${PROJECT_NAME} SYSTEM PRIVATE
    /usr/local/lib/x86_64-linux-gnu
    /lib/x86_64-linux-gnu/
    ${GNURADIO_RUNTIME_LIBRARY_DIRS}
)

set_target_properties(${LIB_TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "75")

target_link_libraries(${LIB_TARGET_NAME}
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
    avcodec
    avformat
    avutil
    swresample
    swscale
    opus
    x264
    x265
    lzma
    z
    gsdr
    remez
    )

target_compile_definitions(${LIB_TARGET_NAME} PRIVATE $<$<CONFIG:Debug>:DEBUG>>)

include(GenerateExportHeader)

set(EXPORT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/${LIB_TARGET_NAME}/${LIB_TARGET_NAME}_export.h")

generate_export_header(
    ${LIB_TARGET_NAME}
    BASE_NAME ${PROJECT_MACRO_PREFIX}
    EXPORT_MACRO_NAME ${PROJECT_MACRO_PREFIX}_PUBLIC
    NO_EXPORT_MACRO_NAME ${PROJECT_MACRO_PREFIX}_PRIVATE
    STATIC_DEFINE ${PROJECT_MACRO_PREFIX}_STATIC_BUILD
    EXPORT_FILE_NAME ${EXPORT_FILE}
)

if (WIN32)
    install(
        TARGETS ${LIB_TARGET_NAME}
        EXPORT ${LIB_TARGET_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
        ARCHIVE DESTINATION lib
    )

    install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION bin OPTIONAL)
else ()
    install(
        TARGETS ${LIB_TARGET_NAME}
        EXPORT ${LIB_TARGET_NAME}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif ()

install(DIRECTORY ${LIB_DEV_PUBLIC_HEADER_DIR}/${LIB_TARGET_NAME} DESTINATION include/)
install(EXPORT ${LIB_TARGET_NAME} DESTINATION lib/cmake/${LIB_TARGET_NAME} FILE ${LIB_TARGET_NAME}-config.cmake)

add_subdirectory(applications)
