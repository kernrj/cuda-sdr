cmake_minimum_required(VERSION 3.14)

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

project(gpusdr_tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


add_executable(
        gpusdr_tests
        FirTests.cpp
)

set_target_properties(gpusdr_tests PROPERTIES CUDA_ARCHITECTURES "75")

set(GTEST_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/googletest-src/googletest/include")

target_include_directories(
        gpusdr_tests
        SYSTEM PUBLIC
        /usr/local/cuda/include
        "${GTEST_INCLUDE_DIR}"
)

target_link_libraries(
        gpusdr_tests
        gpusdr
        gtest_main
)

add_test(NAME gpusdr_tests1 COMMAND gpusdr_tests)
