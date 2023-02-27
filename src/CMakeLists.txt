cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PROJECT_MACRO_PREFIX GS)

include(FetchContent)

message("src '${CMAKE_CXX_COMPILER}' '${CMAKE_C_COMPILER}'")

FetchContent_Declare(
    gsdr
    #    GIT_REPOSITORY https://github.com/kernrj/gsdr.git
    #    GIT_TAG main
    SOURCE_DIR /home/rick/projects/gsdr
)
FetchContent_MakeAvailable(gsdr)

FetchContent_Declare(
    remez
    #GIT_REPOSITORY https://github.com/kernrj/remez-exchange.git
    #GIT_TAG main
    SOURCE_DIR /home/rick/projects/remez
)
FetchContent_MakeAvailable(remez)

set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig")

find_package(CUDAToolkit REQUIRED)

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
    buffers/CudaMemory.cpp
    buffers/CudaMemory.h
    buffers/CudaMemSet.cpp
    buffers/Memory.cpp
    buffers/Memory.h
    buffers/OwnedBuffer.cpp
    buffers/RelocatableResizableBufferFactory.cpp
    buffers/RelocatableCudaBufferFactory.cpp
    buffers/RelocatableResizableBuffer.cpp
    buffers/ResizableBuffer.cpp
    buffers/ResizableBufferFactory.cpp
    buffers/SysMemAllocator.cpp
    buffers/SysMemBufferCopier.cpp
    buffers/BufferFactory.cpp
    driver/DriverToDot.cpp
    driver/DriverToDot.h
    driver/DriverToDotFactory.h
    driver/DriverTool.h
    driver/FilterDriver.cpp
    driver/FilterDriver.h
    driver/FilterDriverFactory.h
    driver/SteppingDriver.cpp
    driver/SteppingDriver.h
    driver/SteppingDriverFactory.h
    Factories.cpp
    filters/AacFileWriter.cpp
    filters/AacFileWriter.h
    filters/AddConst.cpp
    filters/AddConst.h
    filters/AddConstToVectorLength.cpp
    filters/AddConstToVectorLength.h
    filters/BaseFilter.cpp
    filters/BaseSink.cpp
    filters/ComplexCosineSource.cpp
    filters/ComplexCosineSource.h
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
    filters/factories/RfToPcmAudioFactory.h
    filters/factories/QuadDemodFactory.h
    filters/FileReader.cpp
    filters/FileReader.h
    filters/FilterFactories.cpp
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
    filters/PortRemappingSink.cpp
    filters/PortRemappingSink.h
    filters/QuadAmDemod.cpp
    filters/QuadAmDemod.h
    filters/QuadFmDemod.cpp
    filters/QuadFmDemod.h
    filters/ReadByteCountMonitor.cpp
    filters/ReadByteCountMonitor.h
    filters/factories/ReadByteCountMonitorFactory.h
    filters/RfToPcmAudio.cpp
    filters/RfToPcmAudio.h
    filters/Waiter.cpp
    FileLogger.cpp
    FileLogger.h
    GSLog.cpp
    HackrfSession.cpp
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
    util/Window.cpp)

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

set_target_properties(${LIB_TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${PROJ_CUDA_ARCHITECTURES}")

target_link_directories(
    ${PROJECT_NAME} SYSTEM PRIVATE
    /usr/local/lib/x86_64-linux-gnu
    /lib/x86_64-linux-gnu/
    ${GNURADIO_RUNTIME_LIBRARY_DIRS}
)

get_target_property(LIB_TARGET_TYPE ${LIB_TARGET_NAME} TYPE)
if (LIB_TARGET_TYPE STREQUAL STATIC_LIBRARY)
    set(DEPENDENCY_VISIBILITY PUBLIC)
else()
    set(DEPENDENCY_VISIBILITY PRIVATE)
endif()

target_link_libraries(${LIB_TARGET_NAME}
    ${DEPENDENCY_VISIBILITY}
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
    CUDA::cudart
)

proj_set_default_warning_flags(${LIB_TARGET_NAME} PRIVATE)

if (CMAKE_BUILD_TYPE EQUAL "DEBUG" OR CMAKE_BUILD_TYPE EQUAL "RELWITHDEBINFO")
    target_compile_definitions(${LIB_TARGET_NAME} PRIVATE DEBUG)
endif()

include(GenerateExportHeader)

set(GENERATED_HEADER_DIR "${PROJECT_BINARY_DIR}/include")
set(EXPORT_FILE "${GENERATED_HEADER_DIR}/${LIB_PROJECT_NAME}/${LIB_PROJECT_NAME}_export.h")
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

generate_export_header(
    ${LIB_TARGET_NAME}
    BASE_NAME ${PROJECT_MACRO_PREFIX}
    EXPORT_MACRO_NAME ${PROJECT_MACRO_PREFIX}_PUBLIC
    NO_EXPORT_MACRO_NAME ${PROJECT_MACRO_PREFIX}_PRIVATE
    STATIC_DEFINE ${PROJECT_MACRO_PREFIX}_STATIC_BUILD
    EXPORT_FILE_NAME ${EXPORT_FILE}
)

target_include_directories(${LIB_TARGET_NAME} SYSTEM PRIVATE "${GENERATED_HEADER_DIR}")

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
install(DIRECTORY "${GENERATED_HEADER_DIR}/${LIB_PROJECT_NAME}" DESTINATION include/)
install(EXPORT ${LIB_TARGET_NAME} DESTINATION lib/cmake/${LIB_TARGET_NAME} FILE ${LIB_TARGET_NAME}-config.cmake)

add_subdirectory(applications)
