cmake_minimum_required(VERSION 3.14)

set(APPLICATIONS_PROJECT_NAME ${LIB_TARGET_NAME}_applications)
project(${APPLICATIONS_PROJECT_NAME})

include(FetchContent)
FetchContent_Declare(
    remez
    GIT_REPOSITORY https://github.com/kernrj/remez-exchange.git
    GIT_TAG main
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(remez)

add_executable(sdrtest sdrtest.cpp)
set_target_properties(sdrtest PROPERTIES CUDA_ARCHITECTURES "75")
target_link_libraries(sdrtest PRIVATE ${LIB_TARGET_NAME})
target_include_directories(
    sdrtest
    SYSTEM PRIVATE
    "${LIB_PROJECT_ROOT_DIR}/include"
    "${remez_SOURCE_DIR}/include")

add_executable(nbfm_test nbfm_test.cpp)
set_target_properties(nbfm_test PROPERTIES CUDA_ARCHITECTURES "75")
target_link_libraries(nbfm_test PRIVATE ${LIB_TARGET_NAME})
target_include_directories(
    nbfm_test
    SYSTEM BEFORE PRIVATE
    "${LIB_DEV_PUBLIC_HEADER_DIR}"
    "${remez_SOURCE_DIR}/include")

include_directories("/usr/local/cuda/include")
