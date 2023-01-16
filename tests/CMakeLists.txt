cmake_minimum_required(VERSION 3.14)

set(TEST_PROJECT_NAME ${LIB_TARGET_NAME}_tests)
set(TEST_TARGET_NAME ${TEST_PROJECT_NAME})

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

project(${TEST_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(
    ${TEST_PROJECT_NAME}
    CosineSourceTests.cpp
    FirTests.cpp
)

set_target_properties(${TEST_TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "75")

set(GTEST_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/googletest-src/googletest/include")

target_include_directories(
    ${TEST_TARGET_NAME}
    SYSTEM PRIVATE
    /usr/local/cuda/include
    "${GTEST_INCLUDE_DIR}"
    "${LIB_PROJECT_ROOT_DIR}/include"
)

target_link_libraries(
    ${TEST_TARGET_NAME}
    ${LIB_TARGET_NAME}
    gtest_main
)

add_test(NAME ${TEST_TARGET_NAME}_tests1 COMMAND ${TEST_TARGET_NAME})
