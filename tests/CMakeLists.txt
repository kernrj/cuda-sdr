cmake_minimum_required(VERSION 3.19)

set(TEST_PROJECT_NAME ${LIB_TARGET_NAME}_tests)
set(TEST_TARGET_NAME ${TEST_PROJECT_NAME})

message("tests '${CMAKE_CXX_COMPILER}' '${CMAKE_C_COMPILER}'")

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

project(${TEST_PROJECT_NAME})

add_executable(
    ${TEST_PROJECT_NAME}
    CosineSourceTests.cpp
    FirTests.cpp
    SteppingDriverTest.cpp
)

proj_set_default_warning_flags(${TEST_TARGET_NAME} PRIVATE)
#set_target_properties(${TEST_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "${PROJ_CUDA_ARCHITECTURES}")

set(GENERATED_HEADER_DIR "${CMAKE_BINARY_DIR}/include")
target_include_directories(${TEST_TARGET_NAME} SYSTEM PRIVATE "${GENERATED_HEADER_DIR}")

set(GTEST_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/googletest-src/googletest/include")

target_include_directories(
    ${TEST_TARGET_NAME}
    SYSTEM PRIVATE
    "${GTEST_INCLUDE_DIR}"
    "${LIB_PROJECT_ROOT_DIR}/include"
)

#target_link_directories(${TEST_TARGET_NAME} PRIVATE /usr/local/cuda/lib64)

target_link_libraries(
    ${TEST_TARGET_NAME}
    ${LIB_TARGET_NAME}
    gtest_main
)

add_test(NAME ${TEST_TARGET_NAME}_tests1 COMMAND ${TEST_TARGET_NAME})
