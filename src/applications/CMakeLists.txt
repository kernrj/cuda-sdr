cmake_minimum_required(VERSION 3.19)

set(APPLICATIONS_PROJECT_NAME ${LIB_TARGET_NAME}_applications)
project(${APPLICATIONS_PROJECT_NAME})

include(FetchContent)

#FetchContent_Declare(
#    gsdr
#    #    GIT_REPOSITORY https://github.com/kernrj/gsdr.git
#    #    GIT_TAG main
#    SOURCE_DIR /home/rick/projects/gsdr
#    CMAKE_ARGS
#        -DCMAKE_CXX_COMPILER="${LIB_CXX_COMPILER}"
#        -DCMAKE_C_COMPILER="${LIB_C_COMPILER}"
#        -DCMAKE_CUDA_COMPILER="${LIB_CUDA_COMPILER}"
#)
#FetchContent_MakeAvailable(gsdr)

FetchContent_Declare(
    remez
    GIT_REPOSITORY https://github.com/kernrj/remez-exchange.git
    GIT_TAG main
)
FetchContent_MakeAvailable(remez)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)
FetchContent_MakeAvailable(nlohmann_json)

set(GENERATED_HEADER_DIR "${CMAKE_BINARY_DIR}/include")

add_executable(sdrtest sdrtest.cpp)
set_target_properties(sdrtest PROPERTIES CUDA_ARCHITECTURES "${PROJ_CUDA_ARCHITECTURES}")
target_link_libraries(sdrtest PRIVATE ${LIB_TARGET_NAME})
proj_set_default_warning_flags(sdrtest PRIVATE)
target_include_directories(
    sdrtest
    SYSTEM PRIVATE
    "${LIB_PROJECT_ROOT_DIR}/include"
    "${remez_SOURCE_DIR}/include"
    "${nlohmann_json_SOURCE_DIR}/include"
    "${GENERATED_HEADER_DIR}"
)

add_executable(nbfm_test nbfm_test.cpp)
set_target_properties(nbfm_test PROPERTIES CUDA_ARCHITECTURES "${PROJ_CUDA_ARCHITECTURES}")
target_link_libraries(nbfm_test PRIVATE ${LIB_TARGET_NAME})
proj_set_default_warning_flags(nbfm_test PRIVATE)
target_include_directories(
    nbfm_test
    SYSTEM PRIVATE
    "${LIB_DEV_PUBLIC_HEADER_DIR}"
    "${remez_SOURCE_DIR}/include"
    "${nlohmann_json_SOURCE_DIR}/include"
    "${GENERATED_HEADER_DIR}"
)

add_executable(am_test am_test.cpp)
set_target_properties(am_test PROPERTIES CUDA_ARCHITECTURES "${PROJ_CUDA_ARCHITECTURES}")
target_link_libraries(am_test PRIVATE ${LIB_TARGET_NAME})
proj_set_default_warning_flags(am_test PRIVATE)
target_include_directories(
    am_test
    SYSTEM PRIVATE
    "${LIB_DEV_PUBLIC_HEADER_DIR}"
    "${remez_SOURCE_DIR}/include"
    "${nlohmann_json_SOURCE_DIR}/include"
    "${GENERATED_HEADER_DIR}"
)

add_executable(digital_to_audio 145.45_digital_signal_to_audio.cpp)
set_target_properties(digital_to_audio PROPERTIES CUDA_ARCHITECTURES "${PROJ_CUDA_ARCHITECTURES}")
target_link_libraries(digital_to_audio PRIVATE ${LIB_TARGET_NAME})
proj_set_default_warning_flags(digital_to_audio PRIVATE)
target_include_directories(
    digital_to_audio
    SYSTEM PRIVATE
    "${LIB_DEV_PUBLIC_HEADER_DIR}"
    "${remez_SOURCE_DIR}/include"
    "${nlohmann_json_SOURCE_DIR}/include"
    "${GENERATED_HEADER_DIR}"
)

add_executable(server server.cpp)
set_target_properties(server PROPERTIES CUDA_ARCHITECTURES all)
target_link_libraries(server PRIVATE ${LIB_TARGET_NAME} nghttp2)
proj_set_default_warning_flags(server PRIVATE)
target_include_directories(
    server
    SYSTEM PRIVATE
    "${LIB_DEV_PUBLIC_HEADER_DIR}"
    "${nlohmann_json_SOURCE_DIR}/include"
    "${GENERATED_HEADER_DIR}"
)

add_executable(fm_simpletest fm_simpletest.cpp)
set_target_properties(fm_simpletest PROPERTIES CUDA_ARCHITECTURES "${PROJ_CUDA_ARCHITECTURES}")
proj_set_default_warning_flags(fm_simpletest PRIVATE)
target_include_directories(
    fm_simpletest
    SYSTEM PRIVATE
    "${LIB_PROJECT_ROOT_DIR}/include"
    "${remez_SOURCE_DIR}/include"
    "${gsdr_SOURCE_DIR}/include"
    "${nlohmann_json_SOURCE_DIR}/include"
    "${GENERATED_HEADER_DIR}"
)

find_package(PkgConfig)
pkg_search_module(HACKRF REQUIRED IMPORTED_TARGET libhackrf)

target_link_libraries(fm_simpletest PRIVATE gsdr ${LIB_TARGET_NAME})

include_directories("/usr/local/cuda/include")
